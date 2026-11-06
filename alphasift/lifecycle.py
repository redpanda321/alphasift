"""Multi-year A/H price-lifecycle recognition.

The model deliberately keeps A (a base/panic low before a new cycle) separate
from H (the terminal low after a completed boom and a sequence of lower highs).
It uses daily indicators and weekly pivots; no single oversold/new-low flag can
produce a high score on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import pandas as pd

from alphasift.daily import (
    _normalize_daily_history,
    cn_code_to_yfinance_symbol,
    fetch_daily_history,
)

DEFAULT_PROFILE = {
    "mode": "both",
    "lookback_days": 0,  # all available history
    "min_history_days": 504,
    "max_candidates": 0,
    "max_workers": 12,
    "min_score": 45.0,
    "top_per_stage": 20,
    "market_cap_min_us": 1_000_000_000,
    "near_52w_low_pct": 10.0,
    "panic_20d_pct": -12.0,
    "panic_60d_pct": -22.0,
    "h_min_prior_runup_pct": 100.0,
    "h_min_drawdown_pct": 50.0,
    "pivot_window_weeks": 4,
    # Slow-cycle data priority for A/H decisions.  Full available history is
    # requested by default; a caller constrained to a window can explicitly
    # use five_year_monthly, and short histories degrade to weekly bars.
    "monthly_min_bars": 24,
    "five_year_days": 1260,
    "weekly_min_bars": 52,
}


def enrich_lifecycle_features(
    candidates: pd.DataFrame,
    *,
    market: str,
    profile: dict | None = None,
    source: str = "auto",
    cache_dir=None,
    cache_ttl_seconds: float | None = None,
    fetcher: Callable[..., pd.DataFrame] = fetch_daily_history,
    expected_session: str | None = None,
) -> pd.DataFrame:
    """Fetch multi-year histories and attach A/H scores to a snapshot frame."""
    cfg = {**DEFAULT_PROFILE, **(profile or {})}
    frame = candidates.copy()
    if market not in {"cn", "us"}:
        raise ValueError("lifecycle market must be cn or us")
    if market == "us" and "total_mv" not in frame.columns:
        raise ValueError("US lifecycle screening requires market capitalization in USD")
    if market == "us":
        mv = pd.to_numeric(frame["total_mv"], errors="coerce")
        frame = frame[mv.ge(float(cfg["market_cap_min_us"]))].copy()
    limit = int(cfg["max_candidates"])
    if limit > 0:
        frame = frame.head(limit).copy()

    requests = [
        (idx, str(row.get("code", "")).strip()) for idx, row in frame.iterrows()
    ]
    errors: list[str] = []

    def analyze_one(item):
        idx, code = item
        try:
            fetch_code = code
            effective_profile = dict(cfg)
            history_source = (
                "yfinance"
                if market == "us" or int(cfg["lookback_days"]) == 0
                else source
            )
            if market == "cn" and history_source == "yfinance" and code.isdigit():
                fetch_code = cn_code_to_yfinance_symbol(code)
            try:
                hist = fetcher(
                    fetch_code,
                    lookback_days=int(cfg["lookback_days"]),
                    source=history_source,
                    retries=1,
                    cache_dir=None,  # every scan fetches afresh, with no stale fallback
                    cache_ttl_seconds=0,
                )
            except Exception as full_error:
                # A provider can reject its max/all-history query while still
                # serving a normal bounded request.  Preserve the requested
                # priority by retrying only the five-year fallback.
                if int(cfg["lookback_days"]) != 0:
                    raise
                effective_profile["lookback_days"] = int(cfg["five_year_days"])
                hist = fetcher(
                    fetch_code,
                    lookback_days=effective_profile["lookback_days"],
                    source=history_source,
                    retries=1,
                    cache_dir=None,
                    cache_ttl_seconds=0,
                )
                hist.attrs["all_history_fetch_error"] = str(full_error)
            retrieved_at = pd.Timestamp.now(tz="UTC").isoformat()
            if expected_session is not None:
                normalized = _normalize_daily_history(hist)
                if "date" not in normalized:
                    raise ValueError("dated history required for freshness check")
                normalized = normalized[
                    pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
                    <= expected_session
                ].copy()
                normalized.attrs.update(hist.attrs)
                hist = normalized
            features = compute_lifecycle_features(hist, profile=effective_profile)
            if (
                expected_session is not None
                and features["data_as_of"] != expected_session
            ):
                raise RuntimeError(
                    f"outdated history: {features['data_as_of']}; required {expected_session}"
                )
            features["history_retrieved_at"] = retrieved_at
            features["history_scope"] = features["slow_cycle_timeframe"]
            features["snapshot_price"] = frame.loc[idx].get("price")
            # A daily bar supplies a session date, not an observed quote time.
            features["data_timestamp"] = str(hist.attrs.get("data_timestamp", ""))
            features["data_granularity"] = "daily"
            features["data_source"] = str(hist.attrs.get("data_source", history_source))
            if bool(hist.attrs.get("daily_stale")):
                raise RuntimeError("stale history excluded from current ranking")
            row = frame.loc[idx]
            fatal = any(
                str(row.get(key, "")).lower() in {"true", "1", "yes"}
                for key in (
                    "fatal_risk",
                    "delisting_risk",
                    "bankruptcy_risk",
                    "fraud_risk",
                )
            )
            if market == "cn":
                fatal = fatal or "ST" in str(row.get("name", "")).upper()
            if fatal:
                raise RuntimeError("fatal fundamental risk excluded")
            features["fundamental_risk_status"] = "unverified"
            features["lifecycle_action"] = (
                "③ A/H附近但尚未止跌" if not features["stabilized"] else "① 左侧观察"
            )
            return idx, features, None
        except Exception as exc:  # noqa: BLE001 - row-level degradation is reported
            return idx, None, f"{code}: {exc}"

    workers = max(1, min(int(cfg["max_workers"]), len(requests) or 1))
    if workers == 1:
        analyzed = [analyze_one(item) for item in requests]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            analyzed = list(pool.map(analyze_one, requests))

    successful = []
    for idx, features, error in analyzed:
        if error:
            errors.append(error)
            continue
        if features is None:
            continue
        for key, value in features.items():
            if key not in frame.columns:
                frame[key] = None
            frame.at[idx, key] = value
        frame.at[idx, "price"] = features["lifecycle_price"]
        successful.append(idx)

    frame = frame.loc[successful].copy() if successful else frame.iloc[0:0].copy()
    if not frame.empty:
        mode = str(cfg["mode"]).lower()
        if mode == "a":
            frame = frame[frame["a_eligible"].eq(True)].copy()
            frame["lifecycle_score"] = frame["a_score"]
            frame["lifecycle_stage"] = frame["a_stage"]
        elif mode == "h":
            frame = frame[frame["h_eligible"].eq(True)].copy()
            frame["lifecycle_score"] = frame["h_score"]
            frame["lifecycle_stage"] = frame["h_stage"]
        else:
            frame = frame[
                frame["a_eligible"].eq(True) | frame["h_eligible"].eq(True)
            ].copy()
            choose_a = frame["a_eligible"].eq(True)
            frame["lifecycle_score"] = frame["a_score"].where(
                choose_a, frame["h_score"]
            )
            frame["lifecycle_stage"] = frame["h_stage"].where(
                ~choose_a, frame["a_stage"]
            )
        frame = frame[frame["lifecycle_score"] >= float(cfg["min_score"])].copy()
        if mode == "both" and not frame.empty:
            top_n = int(cfg["top_per_stage"])
            top_a = (
                frame[frame["a_eligible"].eq(True)]
                .sort_values("a_score", ascending=False)
                .head(top_n)
            )
            top_h = (
                frame[frame["h_eligible"].eq(True)]
                .sort_values("h_score", ascending=False)
                .head(top_n)
            )
            frame = (
                pd.concat([top_a, top_h]).loc[lambda x: ~x.index.duplicated()].copy()
            )

    frame.attrs.update(candidates.attrs)
    frame.attrs["lifecycle_errors"] = errors
    frame.attrs["lifecycle_attempted"] = len(requests)
    frame.attrs["lifecycle_success_count"] = len(successful)
    return frame


def _select_slow_cycle_frame(
    df: pd.DataFrame, weekly: pd.DataFrame, cfg: dict
) -> dict[str, float | int | str]:
    """Return the slow-cycle regime using the documented data priority.

    Daily OHLCV is resampled locally so CN and US providers have identical
    semantics.  A full-history request uses every available monthly bar.  A
    deliberately bounded request (the five-year fallback) uses its monthly
    bars.  When fewer than two years of month bars are present, weekly bars
    are the only sufficiently granular fallback.
    """
    monthly = (
        df.set_index("_date")
        .resample("ME")
        .agg({"high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    min_months = int(cfg["monthly_min_bars"])
    if len(monthly) >= min_months:
        bars = monthly
        timeframe = (
            "all_available_history_monthly"
            if int(cfg["lookback_days"]) == 0
            else "five_year_monthly"
        )
        fast, slow, return_period = 6, 12, 3
    else:
        if len(weekly) < int(cfg["weekly_min_bars"]):
            raise RuntimeError(
                "insufficient monthly history and weekly fallback history"
            )
        bars = weekly
        timeframe = "weekly_fallback"
        # Roughly mirrors the six/twelve-month regime using 13/26 weeks.
        fast, slow, return_period = 13, 26, 13

    close = pd.to_numeric(bars["close"], errors="coerce").dropna()
    low = float(close.min())
    high = float(close.max())
    position = float((close.iloc[-1] - low) / (high - low)) if high > low else 0.5
    return {
        "timeframe": timeframe,
        "bars": len(close),
        "position": position,
        "fast_ma": float(close.tail(min(fast, len(close))).mean()),
        "slow_ma": float(close.tail(min(slow, len(close))).mean()),
        "return_pct": (
            float((close.iloc[-1] / close.iloc[-return_period - 1] - 1) * 100)
            if len(close) > return_period
            else 0.0
        ),
    }


def compute_lifecycle_features(
    hist: pd.DataFrame, *, profile: dict | None = None
) -> dict:
    """Compute interpretable A-Score/H-Score features from one adjusted history."""
    cfg = {**DEFAULT_PROFILE, **(profile or {})}
    if str(cfg["mode"]).lower() not in {"a", "h", "both"}:
        raise ValueError("lifecycle mode must be a, h or both")
    if int(cfg["min_history_days"]) < 252 or (
        int(cfg["lookback_days"]) != 0
        and int(cfg["lookback_days"]) < int(cfg["min_history_days"])
    ):
        raise ValueError(
            "history must cover at least 252 sessions and fit lookback_days"
        )
    if int(cfg["pivot_window_weeks"]) < 1:
        raise ValueError("pivot_window_weeks must be positive")
    if int(cfg["monthly_min_bars"]) < 1 or int(cfg["weekly_min_bars"]) < 1:
        raise ValueError("slow-cycle minimum bar counts must be positive")
    df = _normalize_daily_history(hist)
    if len(df) < int(cfg["min_history_days"]):
        raise RuntimeError(f"insufficient history: {len(df)} rows")
    df = (
        df.tail(int(cfg["lookback_days"])).copy()
        if int(cfg["lookback_days"])
        else df.copy()
    )
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    volume = pd.to_numeric(
        df.get("volume", pd.Series(index=df.index, dtype=float)), errors="coerce"
    )
    if "date" not in df and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("dated history required; dates cannot be inferred")
    dates = pd.to_datetime(df.get("date", df.index), errors="coerce")
    if dates.isna().any() or (close <= 0).any():
        raise ValueError("history contains invalid dates or nonpositive prices")
    df = (
        df.assign(_date=dates.values)
        .dropna(subset=["_date", "close"])
        .sort_values("_date")
    )
    if df["_date"].duplicated().any():
        raise ValueError("duplicate history dates")
    df = df.reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(
        df.get("volume", pd.Series(index=df.index, dtype=float)), errors="coerce"
    )

    current = float(close.iloc[-1])
    low_52w = float(df["low"].tail(252).min())
    distance_52w = _pct(current, low_52w)
    cycle_low = float(df["low"].min())
    distance_cycle_low = _pct(current, cycle_low)
    distance_3y_low = _pct(current, float(df["low"].tail(756).min()))
    d_pos = int(df["high"].to_numpy().argmax())
    cycle_high = float(df["high"].iloc[d_pos])
    d_date = df["_date"].iloc[d_pos]
    drawdown = max(0.0, (1 - current / cycle_high) * 100)
    pre_d_low = float(df["low"].iloc[: d_pos + 1].min())
    prior_runup = max(0.0, _pct(cycle_high, pre_d_low))

    ret20 = _return(close, 20)
    ret60 = _return(close, 60)
    ret120 = _return(close, 120)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    slope200 = _return(close.rolling(200).mean().dropna(), 60)
    rsi = _rsi_series(close)
    rsi_now = _last_valid(rsi, 50.0)
    rsi_min20 = _last_valid(rsi.tail(20).min(), rsi_now)
    rsi_rebound = max(0.0, rsi_now - rsi_min20)
    macd_hist = _macd_hist(close)
    macd_improving = bool(
        len(macd_hist) >= 5 and macd_hist.iloc[-1] > macd_hist.iloc[-5]
    )
    macd_cross = bool(
        len(macd_hist) >= 2 and macd_hist.iloc[-1] > 0 >= macd_hist.iloc[-2]
    )
    stabilized = bool(current >= float(close.tail(10).min()) * 1.02 or current >= ma20)

    weekly = (
        df.set_index("_date")
        .resample("W-FRI")
        .agg({"high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    # Never use the uncompleted calendar week to confirm a pivot.
    weekly = weekly[weekly.index <= df["_date"].iloc[-1]]
    slow = _select_slow_cycle_frame(df, weekly, cfg)
    piv_hi, _ = _pivots(weekly["high"], int(cfg["pivot_window_weeks"]))
    _, piv_lo = _pivots(weekly["low"], int(cfg["pivot_window_weeks"]))
    post_d_week = pd.Timestamp(d_date).to_period("W-FRI").end_time.normalize()
    post_highs = [(d, v) for d, v in piv_hi if d > post_d_week]
    post_lows = [(d, v) for d, v in piv_lo if d > post_d_week]
    lower_high_count = _descending_count([v for _, v in post_highs])
    lower_low_count = _descending_count([v for _, v in post_lows])
    bottom_divergence = _bottom_divergence(close, rsi)

    flush20 = float(close.pct_change(20).tail(60).min() * 100)
    flush60 = float(close.pct_change(60).tail(60).min() * 100)
    panic = flush20 <= float(cfg["panic_20d_pct"]) or flush60 <= float(
        cfg["panic_60d_pct"]
    )
    acceleration = ret20 < min(-8.0, ret60 / 3.0) or ret60 < -20.0
    vol_panic_ratio, vol_base_ratio, vol_rebound_ratio = _volume_structure(
        volume, close
    )
    volume_pattern = vol_panic_ratio >= 1.35 and vol_base_ratio <= 1.05
    right_confirmed = current > ma20 and macd_improving and ret20 > -3
    post_d_low = float(df["low"].iloc[d_pos:].min())
    distance_post_d_low = _pct(current, post_d_low)
    max_drawdown = float((1 - close / close.cummax()).max() * 100)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = float(true_range.tail(14).mean() / current * 100)
    prior_base = close.iloc[-252:-60]
    base_range = _pct(float(prior_base.max()), float(prior_base.min()))
    pre_flush_return = _return(close.iloc[:-60], 120)
    h_structure = (
        prior_runup >= float(cfg["h_min_prior_runup_pct"])
        and drawdown >= float(cfg["h_min_drawdown_pct"])
        and lower_high_count >= 2
        and slope200 < 0
        and len(close) - d_pos >= 126
    )
    near_low = distance_52w <= float(cfg["near_52w_low_pct"]) * 2
    h_eligible = h_structure and near_low and distance_post_d_low <= 20 and acceleration
    a_eligible = (
        not h_structure
        and near_low
        and distance_3y_low <= 35
        and panic
        and pre_flush_return <= 10
        and (base_range <= 60 or pre_flush_return < -10)
    )

    # A/H are long-cycle labels.  Prefer all-available-history monthly bars,
    # then a caller-provided five-year monthly window, and only then weekly
    # bars when monthly history is unavailable.  This keeps a short provider
    # response from silently treating daily noise as a cycle boundary.
    slow_position = slow["position"]
    slow_bearish_or_flat = slow["fast_ma"] <= slow["slow_ma"] or slow["return_pct"] <= 0
    a_eligible = a_eligible and slow_position <= 0.40
    h_eligible = h_eligible and slow_position <= 0.35 and slow_bearish_or_flat

    # A: low location + a recent flush + exhaustion/turn, penalized when the
    # chart clearly contains the completed D->E->F->G structure required by H.
    a_score = 0.0
    a_score += 15 * _descending(distance_3y_low, 0, 35)
    a_score += 15 * _descending(distance_52w, 0, float(cfg["near_52w_low_pct"]) * 2)
    a_score += 10 * _descending(max(ret120, slope200), -35, 10)
    a_score += 15 * max(
        _descending(flush20, float(cfg["panic_20d_pct"]), 0),
        _descending(flush60, float(cfg["panic_60d_pct"]), 0),
    )
    a_score += 10 * (0.35 + 0.65 * float(stabilized))
    a_score += 8 * min(
        1.0, (_descending(rsi_now, 25, 55) + min(rsi_rebound / 12, 1)) / 2
    )
    a_score += 8 * (0.35 + 0.45 * float(macd_improving) + 0.20 * float(macd_cross))
    a_score += 8 * (
        0.25 + 0.50 * float(volume_pattern) + 0.25 * min(vol_rebound_ratio / 1.5, 1)
    )
    a_score += 6 * float(bottom_divergence)
    a_score += 8 * (1 - slow_position)
    # Fundamental safety cannot be inferred from OHLCV. No free risk points.
    if (
        prior_runup >= float(cfg["h_min_prior_runup_pct"])
        and lower_high_count >= 2
        and drawdown >= 45
    ):
        a_score -= 18

    # H has hard structural gates: prior boom, D top, drawdown, and repeated
    # lower highs. Oversold alone cannot compensate for missing cycle history.
    h_score = 0.0
    h_score += 15 * _ascending(prior_runup, 60, 180)
    h_score += 15 * _ascending(drawdown, 35, 75)
    h_score += 10 * _descending(slope200, -30, 5)
    h_score += 15 * min(lower_high_count / 3, 1)
    h_score += 8 * min(lower_low_count / 3, 1)
    h_score += 12 * _ascending(drawdown, float(cfg["h_min_drawdown_pct"]), 85)
    h_score += 8 * float(acceleration)
    h_score += 5 * _descending(distance_post_d_low, 0, 20)
    h_score += 7 * min(1.0, (_descending(rsi_now, 20, 45) + float(macd_improving)) / 2)
    h_score += 5 * float(bottom_divergence)
    h_score += 8 * (1 - slow_position)
    if prior_runup < float(cfg["h_min_prior_runup_pct"]):
        h_score = min(h_score, 49)
    if drawdown < float(cfg["h_min_drawdown_pct"]):
        h_score = min(h_score, 54)
    if lower_high_count < 2:
        h_score = min(h_score, 59)
    if distance_52w > float(cfg["near_52w_low_pct"]) * 2:
        h_score = min(h_score, 54)

    a_score = round(_clip(a_score), 2)
    h_score = round(_clip(h_score), 2)
    a_stage = (
        "A右侧启动"
        if right_confirmed
        else ("A附近止跌" if stabilized else "A附近尚未止跌")
    )
    h_stage = (
        "H右侧确认"
        if right_confirmed
        else ("H附近动能衰竭" if macd_improving else "H附近尚未止跌")
    )
    if not a_eligible:
        a_stage = "非A结构"
    if not h_eligible:
        h_stage = "非H结构"
    if (a_eligible or h_eligible) and not stabilized:
        action = "③ A/H附近但尚未止跌"
    else:
        action = "① 左侧观察"

    reasons = [
        f"距52周低点{distance_52w:.1f}%",
        f"距D高点回撤{drawdown:.1f}%",
        f"D前涨幅{prior_runup:.1f}%",
        f"Lower High/Low={lower_high_count}/{lower_low_count}",
        f"RSI14={rsi_now:.1f}",
    ]
    if bottom_divergence:
        reasons.append("RSI底背离")
    if volume_pattern:
        reasons.append("放量杀跌后缩量")

    return {
        "a_eligible": bool(a_eligible),
        "h_eligible": bool(h_eligible),
        "stabilized": stabilized,
        "right_confirmed": bool(right_confirmed),
        "history_sessions": len(df),
        "history_start": df["_date"].iloc[0].date().isoformat(),
        "price_basis": "adjusted OHLC; same basis for all price comparisons",
        "max_drawdown_pct": round(max_drawdown, 2),
        "max_drawdown_from_d_pct": round((1 - post_d_low / cycle_high) * 100, 2),
        "distance_post_d_low_pct": round(distance_post_d_low, 2),
        "atr14_pct": round(atr_pct, 2),
        "ma20": round(ma20, 4),
        "ma50": round(ma50, 4),
        "ma200": round(ma200, 4),
        "ma200_slope_60d_pct": round(slope200, 2),
        "weekly_pivot_highs": [
            {"date": d.date().isoformat(), "price": v} for d, v in post_highs
        ],
        "weekly_pivot_lows": [
            {"date": d.date().isoformat(), "price": v} for d, v in post_lows
        ],
        "slow_cycle_timeframe": slow["timeframe"],
        "slow_cycle_bars": slow["bars"],
        "slow_cycle_position": round(slow_position, 4),
        "slow_cycle_fast_ma": round(slow["fast_ma"], 4),
        "slow_cycle_slow_ma": round(slow["slow_ma"], 4),
        "slow_cycle_return_pct": round(slow["return_pct"], 2),
        "a_score": a_score,
        "h_score": h_score,
        "a_stage": a_stage,
        "h_stage": h_stage,
        "lifecycle_action": action,
        "data_as_of": df["_date"].iloc[-1].date().isoformat(),
        "lifecycle_price": round(current, 4),
        "low_52w": round(low_52w, 4),
        "distance_52w_low_pct": round(distance_52w, 2),
        "cycle_high": round(cycle_high, 4),
        "cycle_high_date": pd.Timestamp(d_date).date().isoformat(),
        "drawdown_from_cycle_high_pct": round(drawdown, 2),
        "distance_cycle_low_pct": round(distance_cycle_low, 2),
        "distance_3y_low_pct": round(distance_3y_low, 2),
        "prior_runup_pct": round(prior_runup, 2),
        "lower_high_count": lower_high_count,
        "lower_low_count": lower_low_count,
        "bottom_divergence": bottom_divergence,
        "rsi14": round(rsi_now, 2),
        "macd_status": "improving" if macd_improving else "weakening",
        "change_60d": round(ret60, 2),
        "lifecycle_reasons": reasons,
    }


def _pivots(series: pd.Series, window: int):
    highs, lows = [], []
    values = series.to_numpy()
    for i in range(window, len(values) - window):
        block = values[i - window : i + window + 1]
        if (
            values[i] == block.max()
            and values[i] > values[i - 1]
            and values[i] >= values[i + 1]
        ):
            highs.append((series.index[i], float(values[i])))
        if (
            values[i] == block.min()
            and values[i] < values[i - 1]
            and values[i] <= values[i + 1]
        ):
            lows.append((series.index[i], float(values[i])))
    return highs, lows


def _descending_count(values: list[float], tolerance: float = 0.02) -> int:
    count = 0
    for a, b in reversed(list(pairwise(values))):
        if b >= a * (1 - tolerance):
            break
        count += 1
    return count


def _bottom_divergence(close: pd.Series, rsi: pd.Series) -> bool:
    recent = close.tail(160)
    lows = []
    for i in range(5, len(recent) - 5):
        if recent.iloc[i] == recent.iloc[i - 5 : i + 6].min():
            lows.append(recent.index[i])
    if len(lows) < 2:
        return False
    i1, i2 = lows[-2], lows[-1]
    return bool(close.loc[i2] < close.loc[i1] * 0.99 and rsi.loc[i2] > rsi.loc[i1] + 2)


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).copy()
    loss = -delta.clip(upper=0).copy()
    for series in (gain, loss):
        series.iloc[:period] = float("nan")
    if len(close) > period:
        gain.iloc[period] = delta.clip(lower=0).iloc[1 : period + 1].mean()
        loss.iloc[period] = -delta.clip(upper=0).iloc[1 : period + 1].mean()
    gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    result = 100 - 100 / (1 + gain / loss.where(loss.ne(0)))
    return result.mask(loss.eq(0) & gain.gt(0), 100).mask(loss.eq(0) & gain.eq(0), 50)


def _macd_hist(close: pd.Series) -> pd.Series:
    diff = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )
    return diff - diff.ewm(span=9, adjust=False).mean()


def _volume_structure(volume: pd.Series, close: pd.Series):
    if volume.notna().sum() < 80:
        return 1.0, 1.0, 1.0
    down = close.pct_change() < -0.03
    base = float(volume.tail(120).median()) or 1.0
    panic = _last_valid(volume.tail(60)[down.tail(60)].max(), 0.0) / base
    bottom = float(volume.tail(10).mean()) / base
    rebound_days = close.pct_change().tail(10) > 0.02
    rebound = (
        float(volume.tail(10)[rebound_days].mean()) / base
        if rebound_days.any()
        else 0.0
    )
    return panic, bottom, rebound


def _return(series: pd.Series, periods: int) -> float:
    if len(series) <= periods or float(series.iloc[-periods - 1]) <= 0:
        return 0.0
    return (float(series.iloc[-1]) / float(series.iloc[-periods - 1]) - 1) * 100


def _pct(value: float, base: float) -> float:
    return (value / base - 1) * 100 if base > 0 else 0.0


def _ascending(value: float, low: float, high: float) -> float:
    return _clip((value - low) / max(high - low, 1e-9), 0, 1)


def _descending(value: float, low: float, high: float) -> float:
    return 1 - _ascending(value, low, high)


def _clip(value: float, low: float = 0, high: float = 100) -> float:
    return float(min(max(value, low), high))


def _last_valid(value, default: float) -> float:
    if isinstance(value, pd.Series):
        value = value.dropna()
        return float(value.iloc[-1]) if not value.empty else float(default)
    return float(value) if pd.notna(value) else float(default)

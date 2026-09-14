"""Independent five-calendar-year / all-available-history lifecycle comparison.

Run with explicit symbols (sample) or --market cn/us (filtered market universe).
All prices in both windows share the adjustment basis of a single fresh download.
Agreement measures temporal robustness, not independent-provider verification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path

import pandas as pd

from alphasift.daily import _normalize_daily_history, cn_code_to_yfinance_symbol
from alphasift.lifecycle import compute_lifecycle_features, _pivots, _select_slow_cycle_frame
from alphasift.lifecycle_contract import FALLBACK_WINDOW_YEARS, STAGES, strategy_contract


def classify_window(history: pd.DataFrame, *, min_history_days: int | None = None) -> dict:
    df = _normalize_daily_history(history).reset_index(drop=True)
    profile = {"lookback_days": 0}
    if min_history_days is not None:
        profile["min_history_days"] = min_history_days
    f = compute_lifecycle_features(df, profile=profile)
    c = df.close
    price = float(c.iloc[-1])
    dates = pd.to_datetime(df.date)
    weekly = (
        df.assign(date=dates)
        .set_index("date")
        .resample("W-FRI")
        .agg({"high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    weekly = weekly[weekly.index <= dates.iloc[-1]]
    highs, _ = _pivots(weekly.high, 4)
    _, lows = _pivots(weekly.low, 4)
    monthly = (
        df.assign(date=dates)
        .set_index("date")
        .resample("ME")
        .agg({"high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    slow = _select_slow_cycle_frame(df.assign(_date=dates), weekly, {"lookback_days": 0, "monthly_min_bars": 24, "weekly_min_bars": 52})
    month_position = slow["position"]
    month_ma6 = slow["fast_ma"]
    month_ma12 = slow["slow_ma"]
    month_return_3 = slow["return_pct"]
    month_highs, _ = _pivots(monthly.high, 1)
    d_date = pd.Timestamp(f["cycle_high_date"])
    d_age = int((dates > d_date).sum())
    dd = f["drawdown_from_cycle_high_pct"]
    scores = {
        "A": f["a_score"],
        "H": f["h_score"],
        "B": 0.0,
        "C": 0.0,
        "D": 0.0,
        "E": 0.0,
        "F": 0.0,
        "G": 0.0,
    }
    eligible = {
        "A": f["a_eligible"],
        "H": f["h_eligible"],
        "B": False,
        "C": False,
        "D": False,
        "E": False,
        "F": False,
        "G": False,
    }
    evidence = {
        "D_date": f["cycle_high_date"],
        "D_high": f["cycle_high"],
        "D_age_sessions": d_age,
        "drawdown_pct": dd,
        "monthly": {
            "timeframe": slow["timeframe"],
            "bars": slow["bars"],
            "position_60m": round(month_position, 4),
            "ma6": round(month_ma6, 4),
            "ma12": round(month_ma12, 4),
            "return_3m_pct": round(month_return_3, 2),
        },
    }
    # A/H need a depressed monthly price regime, while D needs a high-zone
    # monthly regime. This prevents daily/weekly noise from assigning a stage
    # that contradicts the slower chart.
    eligible["A"] = bool(f["a_eligible"] and month_position <= 0.40)
    eligible["H"] = bool(
        f["h_eligible"]
        and month_position <= 0.35
        and (month_ma6 <= month_ma12 or month_return_3 <= 0)
    )
    scores["A"] = round(min(100, f["a_score"] + 8 * (1 - month_position)), 2)
    scores["H"] = round(min(100, f["h_score"] + 8 * (1 - month_position)), 2)
    # D is only a high-zone candidate; a final top cannot be confirmed in real time.
    eligible["D"] = (
        f["prior_runup_pct"] >= 100
        and dd <= 12
        and d_age <= 63
        and month_position >= 0.75
        and month_ma6 >= month_ma12
    )
    if eligible["D"]:
        scores["D"] = round(
            min(
                100,
                60
                + 20 * (1 - dd / 12)
                + 10 * (f["rsi14"] >= 65)
                + 10 * (f["macd_status"] == "weakening"),
            ),
            2,
        )
    # E: first material post-D trough and its rebound; no later confirmed
    # rebound peak is allowed (otherwise this may already be F/G).
    d_week = d_date.to_period("W-FRI").end_time.normalize()
    after_lows = [(d, v) for d, v in lows if d > d_week]
    after_highs = [(d, v) for d, v in highs if d > d_week]
    if after_lows:
        trough_date, trough = after_lows[0]
        rebound = (price / trough - 1) * 100
        first_decline = (1 - trough / f["cycle_high"]) * 100
        eligible["E"] = bool(
            f["prior_runup_pct"] >= 100
            and first_decline >= 20
            and 5 <= dd <= 50
            and 5 <= rebound <= 50
            and d_age <= 252
            and not after_highs
            and len(after_lows) == 1
            and float(c.pct_change(20).iloc[-1]) > 0
            and price < f["ma200"]
            and month_ma6 < month_ma12
            and month_return_3 > 0
        )
        if eligible["E"]:
            scores["E"] = round(
                65
                + 15 * min(rebound / 20, 1)
                + 10 * (f["ma200_slope_60d_pct"] < 0)
                + 10 * (f["macd_status"] == "improving"),
                2,
            )
            evidence["E_trough"] = {"date": str(trough_date.date()), "price": trough}
    # F: exactly one confirmed post-D lower high followed by a second, deeper
    # trough than the first; currently rebounding without a second confirmed
    # peak yet. One decline leg further down the cycle than E.
    if len(after_highs) == 1 and len(after_lows) == 2:
        peak_date, peak = after_highs[0]
        trough1_date, trough1 = after_lows[0]
        trough2_date, trough2 = after_lows[1]
        rebound = (price / trough2 - 1) * 100
        decline_from_peak = (1 - trough2 / peak) * 100
        eligible["F"] = bool(
            f["prior_runup_pct"] >= 100
            and peak < f["cycle_high"]
            and trough2 < trough1
            and decline_from_peak >= 15
            and 20 <= dd <= 65
            and 5 <= rebound <= 50
            and d_age <= 504
            and float(c.pct_change(20).iloc[-1]) > 0
            and price < f["ma200"]
            and month_ma6 < month_ma12
        )
        if eligible["F"]:
            scores["F"] = round(
                60
                + 15 * min(rebound / 20, 1)
                + 10 * (f["ma200_slope_60d_pct"] < 0)
                + 10 * (f["macd_status"] == "improving")
                + 5 * (dd >= 35),
                2,
            )
            evidence["F_trough"] = {"date": str(trough2_date.date()), "price": trough2}
            evidence["F_lower_high"] = {"date": str(peak_date.date()), "price": peak}
    # G: two confirmed post-D lower highs and a third, deeper trough; the
    # decline is maturing toward acceleration but has not yet met H's near-low
    # and acceleration gates. One decline leg further down than F.
    if len(after_highs) == 2 and len(after_lows) == 3:
        peak1_date, peak1 = after_highs[0]
        peak2_date, peak2 = after_highs[1]
        trough2_date, trough2 = after_lows[1]
        trough3_date, trough3 = after_lows[2]
        rebound = (price / trough3 - 1) * 100
        decline_from_peak = (1 - trough3 / peak2) * 100
        eligible["G"] = bool(
            f["prior_runup_pct"] >= 100
            and peak2 < peak1 < f["cycle_high"]
            and trough3 < trough2
            and decline_from_peak >= 15
            and 35 <= dd <= 80
            and 5 <= rebound <= 50
            and d_age <= 756
            and not f["h_eligible"]
            and float(c.pct_change(20).iloc[-1]) > 0
            and price < f["ma200"]
            and month_ma6 < month_ma12
        )
        if eligible["G"]:
            scores["G"] = round(
                55
                + 15 * min(rebound / 20, 1)
                + 10 * (f["ma200_slope_60d_pct"] < 0)
                + 10 * (f["macd_status"] == "improving")
                + 10 * (dd >= 50),
                2,
            )
            evidence["G_trough"] = {"date": str(trough3_date.date()), "price": trough3}
            evidence["G_lower_highs"] = [
                {"date": str(peak1_date.date()), "price": peak1},
                {"date": str(peak2_date.date()), "price": peak2},
            ]
    # B requires an A that was identifiable using only information then
    # available, followed by the FIRST confirmed weekly rally high.
    for low_date, low in reversed(lows):
        if (dates.iloc[-1] - low_date).days > 730:
            break
        rally_highs = [(d, v) for d, v in highs if d > low_date]
        if len(rally_highs) != 1:
            continue
        prefix = df[dates <= low_date]
        if len(prefix) < 504:
            continue
        anchor = compute_lifecycle_features(prefix, profile={"lookback_days": 0})
        if not anchor["a_eligible"] or len(rally_highs) != 1:
            continue
        peak_date, peak = rally_highs[0]
        pullback = (1 - price / peak) * 100
        rise = (peak / low - 1) * 100
        subsequent_low = float(df.loc[dates > peak_date, "low"].min())
        if (
            rise >= 30
            and 5 <= pullback <= 25
            and subsequent_low > low
            and price > f["ma200"]
            and f["ma200_slope_60d_pct"] > 0
            and f["ma50"] > f["ma200"]
            and month_ma6 > month_ma12
            and month_position >= 0.45
        ):
            eligible["B"] = True
            scores["B"] = round(
                65
                + 15 * min(rise / 60, 1)
                + 10 * f["stabilized"]
                + 10 * (f["macd_status"] == "improving"),
                2,
            )
            evidence["B_anchor_A"] = {
                "date": str(low_date.date()),
                "price": low,
                "rally_high_date": str(peak_date.date()),
                "pullback_pct": pullback,
            }
            break
    # C: same A anchor as B, but a SECOND confirmed rally high above the
    # first with a higher-low pullback in between — the uptrend has continued
    # one leg further than B — while price has pulled back from that second
    # high without yet meeting D's high-zone criteria.
    for low_date, low in reversed(lows):
        if (dates.iloc[-1] - low_date).days > 1460:
            break
        rally_highs = [(d, v) for d, v in highs if d > low_date]
        if len(rally_highs) != 2:
            continue
        prefix = df[dates <= low_date]
        if len(prefix) < 504:
            continue
        anchor = compute_lifecycle_features(prefix, profile={"lookback_days": 0})
        if not anchor["a_eligible"]:
            continue
        peak1_date, peak1 = rally_highs[0]
        peak2_date, peak2 = rally_highs[1]
        mid_lows = [(d, v) for d, v in lows if peak1_date < d < peak2_date]
        if not mid_lows:
            continue
        mid_low_date, mid_low = mid_lows[-1]
        pullback = (1 - price / peak2) * 100
        rise = (peak2 / low - 1) * 100
        subsequent_low = float(df.loc[dates > peak2_date, "low"].min())
        if (
            peak2 > peak1
            and mid_low > low
            and rise >= 50
            and 5 <= pullback <= 25
            and subsequent_low > mid_low
            and price > f["ma200"]
            and f["ma200_slope_60d_pct"] > 0
            and f["ma50"] > f["ma200"]
            and month_ma6 > month_ma12
            and month_position >= 0.55
            and not eligible["D"]
        ):
            eligible["C"] = True
            scores["C"] = round(
                65
                + 15 * min(rise / 100, 1)
                + 10 * f["stabilized"]
                + 10 * (f["macd_status"] == "improving"),
                2,
            )
            evidence["C_anchor_A"] = {
                "date": str(low_date.date()),
                "price": low,
                "first_rally_high": {"date": str(peak1_date.date()), "price": peak1},
                "higher_low": {"date": str(mid_low_date.date()), "price": mid_low},
                "second_rally_high_date": str(peak2_date.date()),
                "pullback_pct": pullback,
            }
            break
    post_d_month_highs = [v for d, v in month_highs if d > d_date.to_period("M").end_time]
    evidence["monthly"]["post_D_lower_high_count"] = sum(
        current < previous
        for previous, current in zip(post_d_month_highs, post_d_month_highs[1:])
    )
    matches = [s for s in scores if eligible[s]]
    return {
        "stage": matches[0]
        if len(matches) == 1
        else "UNCLASSIFIED"
        if not matches
        else "AMBIGUOUS",
        "matches": matches,
        "scores": scores,
        "evidence": evidence,
        "history_start": f["history_start"],
        "history_sessions": f["history_sessions"],
        "as_of": f["data_as_of"],
        "price": f["lifecycle_price"],
        "fundamentals": "unverified",
    }


# The A/H feature math (MA200, multi-year cycle-high lookback, etc.) needs a
# real minimum of daily history to mean anything; this is the absolute floor
# the model itself validates (see lifecycle.py's `min_history_days < 252`
# guard). The production default for the primary 5-year/full pair stays a
# more conservative 504 sessions (~2 years); this lower floor is only used
# for the 1-year fallback pair below, so young listings can still be
# classified instead of raising outright.
MODEL_MIN_HISTORY_DAYS = 252
DEFAULT_MIN_HISTORY_DAYS = 504


def _pick_short_window(df: pd.DataFrame, dates: pd.Series, cutoff: pd.Timestamp):
    """Pick the best available short window to cross-check against full history.

    Tries FALLBACK_WINDOW_YEARS in preference order (five years) and
    uses the first one for which the stock has both a complete span of that
    length *and* additional history beyond it (so the short window and full
    history are meaningfully distinct, not the same data twice). This keeps
    the scan while still requiring at least two independent windows to agree
    before a stage is confirmed.  Shorter histories remain visible through
    the classifier's weekly fallback but cannot claim two-window agreement.
    Returns None if no window (including the shortest fallback) is usable.
    """
    for years in FALLBACK_WINDOW_YEARS:
        start = cutoff - pd.DateOffset(years=years)
        # Weekend/holiday allowance at the window boundary. Does not establish
        # completeness of every session, or availability back to the IPO.
        span_available = dates.min() <= start + pd.Timedelta(days=7)
        extra_history = bool((dates < start).any())
        if span_available and extra_history:
            min_history_days = DEFAULT_MIN_HISTORY_DAYS if years == FALLBACK_WINDOW_YEARS[0] else MODEL_MIN_HISTORY_DAYS
            return years, start, min_history_days
    return None


def crosscheck(history: pd.DataFrame, *, as_of: str) -> dict:
    df = _normalize_daily_history(history)
    if "date" not in df:
        raise ValueError("dated OHLCV required")
    dates = pd.to_datetime(df.date)
    cutoff = pd.Timestamp(as_of)
    df = df.loc[dates <= cutoff].copy()
    dates = pd.to_datetime(df.date)
    if df.empty or dates.max().date().isoformat() != as_of:
        raise ValueError(f"latest daily bar must equal {as_of}")

    if len(df) < MODEL_MIN_HISTORY_DAYS:
        # Too little history for the model to compute anything at all (a very
        # recent IPO) — report it as an honest exclusion rather than raising.
        return {
            "strategy": strategy_contract(),
            "as_of": as_of,
            "window_years": None,
            "five_year_boundary": None,
            "five_year_span_available": False,
            "full_has_older_data": False,
            "status": "INSUFFICIENT_DISTINCT_HISTORY",
            "consensus_stage": None,
            "consensus_score": None,
            "five_year": None,
            "full_history": None,
            "validation_kind": "same-provider different-window robustness; not predictive validation",
        }

    picked = _pick_short_window(df, dates, cutoff)
    if picked is None:
        # Enough history to classify once, but not enough to split into two
        # meaningfully distinct windows for cross-checking.
        window_years, start, comparable = None, None, False
        full = classify_window(df, min_history_days=MODEL_MIN_HISTORY_DAYS)
        short = None
    else:
        window_years, start, min_history_days = picked
        comparable = True
        full = classify_window(df, min_history_days=min_history_days)
        short = classify_window(df.loc[dates >= start], min_history_days=min_history_days)

    same = comparable and short["stage"] == full["stage"]
    status = (
        "INSUFFICIENT_DISTINCT_HISTORY"
        if not comparable
        else "AGREEMENT"
        if same and short["stage"] in STAGES
        else "NO_MATCH"
        if same and short["stage"] == "UNCLASSIFIED"
        else "CONFLICT"
    )
    stage = short["stage"] if status == "AGREEMENT" else None
    return {
        "strategy": strategy_contract(),
        "as_of": as_of,
        "window_years": window_years,
        "five_year_boundary": str(start.date()) if start is not None else None,
        "five_year_span_available": comparable,
        "full_has_older_data": comparable,
        "status": status,
        "consensus_stage": stage,
        "consensus_score": min(short["scores"][stage], full["scores"][stage])
        if stage
        else None,
        "five_year": short if short is not None else full,
        "full_history": full,
        "validation_kind": "same-provider different-window robustness; not predictive validation",
    }


def main(argv=None):
    from alphasift.snapshot_us import fetch_daily_history_yfinance
    from alphasift.freshness import latest_completed_session
    from alphasift.snapshot import fetch_snapshot_with_fallback
    from alphasift.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument(
        "--symbols", help="comma-separated Yahoo symbols; omitted means filtered market"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    as_of = latest_completed_session(args.market, pd.Timestamp.now(tz="UTC"))
    if args.symbols:
        symbols = args.symbols.split(",")
        scope = "explicit sample"
        snapshot_count = None
    else:
        config = Config.from_env()
        snapshot = fetch_snapshot_with_fallback(
            config.snapshot_source_priority,
            market=args.market,
            fallback_snapshot_path=None,
        )
        snapshot_count = len(snapshot)
        valid = snapshot.price.ge(1) & snapshot.amount.ge(20_000_000)
        if args.market == "us":
            valid &= snapshot.total_mv.ge(1_000_000_000)
        else:
            valid &= ~snapshot.name.str.contains("ST|退", case=False, na=False)
        codes = snapshot.loc[valid, "code"].astype(str).tolist()
        symbols = (
            codes
            if args.market == "us"
            else [cn_code_to_yfinance_symbol(c) for c in codes]
        )
        scope = "provider universe; price>=1, amount>=20m; CN excludes ST; US cap>=1b"

    def one(symbol):
        try:
            hist = fetch_daily_history_yfinance(symbol, lookback_days=0)
            return {
                "symbol": symbol,
                "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                **crosscheck(hist, as_of=as_of),
            }
        except Exception as exc:
            return {"symbol": symbol, "status": "FAILED", "error": str(exc)}

    print(f"{args.market}: {len(symbols)} histories, expected {as_of}", flush=True)
    # yfinance/curl_cffi maintains shared process state. High fan-out can
    # deadlock before the first ordered result is yielded, so keep the
    # production default deliberately bounded.
    workers = max(1, min(int(os.getenv("ALPHASIFT_HISTORY_WORKERS", "8")), 16))
    print(f"{args.market}: workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        futures = {pool.submit(one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows) % 250 == 0:
                print(f"{args.market}: {len(rows)}/{len(symbols)}", flush=True)
    payload = {
        "strategy": strategy_contract(),
        "scope": scope,
        "snapshot_count": snapshot_count,
        "attempted": len(symbols),
        "as_of": as_of,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "as_of": as_of,
                "statuses": pd.Series([r["status"] for r in rows])
                .value_counts()
                .to_dict(),
            }
        )
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Lightweight daily K-line enrichment for narrowed candidate pools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import pandas as pd
import requests

_DAILY_FEATURE_DEFAULTS = {
    "daily_data_points": pd.NA,
    "change_60d": pd.NA,
    "ma5": pd.NA,
    "ma20": pd.NA,
    "ma60": pd.NA,
    "ma_bullish": pd.NA,
    "price_above_ma20": pd.NA,
    "macd_status": "",
    "rsi_status": "",
    "rsi14": pd.NA,
    "signal_score": pd.NA,
    "prev_high_20d": pd.NA,
    "range_20d_pct": pd.NA,
    "breakout_20d_pct": pd.NA,
    "volume_ratio_20d": pd.NA,
    "body_pct": pd.NA,
    "pullback_to_ma20_pct": pd.NA,
    "consolidation_days_20d": pd.NA,
}
_DAILY_ENRICH_MAX_WORKERS = 1
_DAILY_HISTORY_CACHE_VERSION = 1
_DAILY_HISTORY_CACHE_TTL_SECONDS = 24 * 60 * 60
_SOURCE_HEALTH_FAILURE_THRESHOLD = 3
_SOURCE_HEALTH_COOLDOWN_SECONDS = 5 * 60
_DEFAULT_TUSHARE_HTTP_URL = "http://api.waditu.com"
_BAOSTOCK_LOCK = threading.Lock()
_BAOSTOCK_OUTAGE_ERROR: str | None = None
_SOURCE_HEALTH: dict[str, dict[str, float]] = {}
_SOURCE_HEALTH_LOCK = threading.Lock()


def enrich_daily_features(
    df: pd.DataFrame,
    *,
    max_rows: int = 100,
    lookback_days: int = 120,
    source: str = "akshare",
    fetch_retries: int = 2,
    cache_dir: str | Path | None = None,
    cache_ttl_seconds: float | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Attach daily technical features to the first ``max_rows`` candidates.

    This intentionally runs after broad snapshot filtering; it is not a full
    market historical-data pass.
    """
    if df.empty or max_rows <= 0:
        return df.copy()

    result = df.copy()
    daily_errors: list[str] = []
    success_count = 0
    selected_index = list(result.index[:max_rows])
    fetch_requests: list[tuple[object, str]] = []
    for idx in selected_index:
        raw_code = str(result.at[idx, "code"] if "code" in result.columns else "").strip()
        if not raw_code:
            continue
        code = raw_code.zfill(6) if raw_code.isdigit() else raw_code
        fetch_requests.append((idx, code))

    def fetch_one(request: tuple[object, str]) -> tuple[object, dict[str, object], str | None]:
        idx, code = request
        try:
            hist = fetch_daily_history(
                code,
                lookback_days=lookback_days,
                source=source,
                retries=fetch_retries,
                cache_dir=cache_dir,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            return idx, compute_daily_features(hist), None
        except Exception as exc:
            return idx, dict(_DAILY_FEATURE_DEFAULTS), f"{code}: {exc}"

    if len(fetch_requests) <= 1:
        fetched_rows = [fetch_one(request) for request in fetch_requests]
    else:
        worker_limit = min(_normalize_max_workers(max_workers), len(fetch_requests))
        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            fetched_rows = list(executor.map(fetch_one, fetch_requests))

    for idx, features, error in fetched_rows:
        if error:
            daily_errors.append(error)
        else:
            success_count += 1
        for key, value in features.items():
            result.at[idx, key] = value

    result.attrs["daily_errors"] = daily_errors
    result.attrs["daily_success_count"] = success_count
    return result


def fetch_daily_history(
    code: str,
    *,
    lookback_days: int = 120,
    source: str = "akshare",
    retries: int = 2,
    cache_dir: str | Path | None = None,
    cache_ttl_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch daily history for one stock code.

    ``source`` accepts ``tencent``, ``akshare``, ``baostock``, ``tushare``,
    ``yfinance`` or ``auto``. ``auto`` prefers Tushare when a token is
    configured, then Tencent's direct HTTP K-line endpoint before wrapper-based
    free sources. Without a token it starts with Tencent. ``yfinance`` is
    explicit-only (never part of ``auto``) and expects a US ticker rather than
    an A-share code.
    """
    normalized_code = _normalize_daily_code(code)
    normalized_lookback_days = int(lookback_days)
    src = _normalize_daily_source(source)
    if src == "auto":
        sources: tuple[str, ...] = (
            ("tushare", "tencent", "akshare", "baostock")
            if _has_tushare_token()
            else ("tencent", "akshare", "baostock")
        )
    elif src in ("akshare", "baostock", "tushare", "tencent", "yfinance"):
        sources = (src,)
    else:
        raise ValueError(f"Unsupported daily source: {source}")

    cache_path = None
    if cache_dir is not None:
        cache_path = _daily_history_cache_path(
            cache_dir,
            code=normalized_code,
            source=src,
            lookback_days=normalized_lookback_days,
        )
        cached = _read_daily_history_cache(cache_path, ttl_seconds=cache_ttl_seconds)
        if cached is not None:
            return cached

    attempts = max(int(retries), 0) + 1
    errors: list[str] = []
    for current in sources:
        disabled_reason = _source_disabled_reason(current)
        if disabled_reason:
            errors.append(f"{current}: {disabled_reason}")
            continue
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if current == "yfinance":
                    from alphasift.snapshot_us import fetch_daily_history_yfinance
                    result = fetch_daily_history_yfinance(code, lookback_days=lookback_days)
                elif current == "tencent":
                    result = _fetch_daily_tencent(
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                elif current == "akshare":
                    result = _fetch_daily_akshare(
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                elif current == "tushare":
                    result = _fetch_daily_tushare(
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                else:
                    result = _fetch_daily_baostock(
                        normalized_code,
                        lookback_days=normalized_lookback_days,
                    )
                if cache_path is not None:
                    _write_daily_history_cache(
                        cache_path,
                        result,
                        code=normalized_code,
                        source=src,
                        lookback_days=normalized_lookback_days,
                    )
                _record_source_success(current)
                return result
            except Exception as exc:  # noqa: BLE001 - aggregated below
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(min(0.5 * (attempt + 1), 2.0))
        errors.append(f"{current} after {attempts} attempts: {last_error}")
        _record_source_failure(current)

    if cache_path is not None:
        stale = _read_daily_history_cache(
            cache_path,
            ttl_seconds=cache_ttl_seconds,
            allow_stale=True,
        )
        if stale is not None:
            stale.attrs["daily_stale"] = True
            stale.attrs["source_errors"] = list(errors)
            return stale

    raise RuntimeError(
        f"daily history fetch failed for {normalized_code}: {'; '.join(errors)}"
    )


def _normalize_daily_code(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)[-6:]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else text


def _normalize_daily_source(source: str | None) -> str:
    return (source or "akshare").strip().lower()


def _normalize_max_workers(value: int | None) -> int:
    if value is None:
        return _DAILY_ENRICH_MAX_WORKERS
    return max(1, int(value))


def _source_disabled_reason(source: str) -> str | None:
    now = time.monotonic()
    with _SOURCE_HEALTH_LOCK:
        state = _SOURCE_HEALTH.get(source)
        if not state:
            return None
        disabled_until = float(state.get("disabled_until", 0.0))
        if disabled_until <= now:
            if disabled_until:
                state["disabled_until"] = 0.0
            return None
        return f"temporarily disabled for {disabled_until - now:.1f}s after repeated failures"


def _record_source_success(source: str) -> None:
    with _SOURCE_HEALTH_LOCK:
        _SOURCE_HEALTH.pop(source, None)


def _record_source_failure(source: str) -> None:
    now = time.monotonic()
    with _SOURCE_HEALTH_LOCK:
        state = _SOURCE_HEALTH.setdefault(source, {"failures": 0.0, "disabled_until": 0.0})
        failures = float(state.get("failures", 0.0)) + 1.0
        state["failures"] = failures
        if failures >= _SOURCE_HEALTH_FAILURE_THRESHOLD:
            state["disabled_until"] = now + _SOURCE_HEALTH_COOLDOWN_SECONDS


def _daily_history_cache_path(
    cache_dir: str | Path,
    *,
    code: str,
    source: str,
    lookback_days: int,
) -> Path:
    key = f"{code}|{source}|{int(lookback_days)}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe_source = "".join(ch if ch.isalnum() else "-" for ch in source).strip("-") or "source"
    safe_code = "".join(ch if ch.isalnum() else "-" for ch in code).strip("-") or "code"
    return Path(cache_dir) / f"{safe_code}_{safe_source}_{int(lookback_days)}_{digest}.json"


def _read_daily_history_cache(
    path: Path,
    *,
    ttl_seconds: float | None,
    allow_stale: bool = False,
) -> pd.DataFrame | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    ttl = _DAILY_HISTORY_CACHE_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
    is_stale = ttl <= 0 or time.time() - stat.st_mtime > ttl
    if is_stale and not allow_stale:
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _DAILY_HISTORY_CACHE_VERSION:
            return None
        frame = payload.get("frame")
        if not isinstance(frame, dict):
            return None
        columns = frame.get("columns")
        data = frame.get("data")
        if not isinstance(columns, list) or not isinstance(data, list):
            return None
        df = pd.DataFrame(data, columns=columns)
        if is_stale:
            df.attrs["daily_stale"] = True
        return df
    except Exception:
        return None


def _write_daily_history_cache(
    path: Path,
    df: pd.DataFrame,
    *,
    code: str,
    source: str,
    lookback_days: int,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _DAILY_HISTORY_CACHE_VERSION,
            "key": {
                "code": code,
                "source": source,
                "lookback_days": int(lookback_days),
            },
            "created_at": datetime.now().isoformat(),
            "frame": json.loads(df.to_json(orient="split", date_format="iso", force_ascii=False)),
        }
        tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        return


def _fetch_daily_akshare(code: str, *, lookback_days: int) -> pd.DataFrame:
    import akshare as ak

    start_date = (datetime.now() - timedelta(days=max(lookback_days * 2, 90))).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=str(code).zfill(6),
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        raise RuntimeError(f"akshare daily history empty for {code}")
    return df.tail(max(lookback_days, 30)).copy()


def _fetch_daily_tencent(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch forward-adjusted daily history from Tencent's direct HTTP API.

    The endpoint is the same low-friction source recommended by a-stock-data for
    stable A-share market data access: no wrapper dependency, browser-like HTTP,
    and much lower IP-ban risk than Eastmoney-heavy endpoints. Tencent returns
    daily K-lines as rows shaped like ``date, open, close, high, low, volume``;
    amount is not always present, so it is exposed as ``NA`` when absent to keep
    the common daily schema stable.
    """
    symbol = _to_tencent_code(code)
    count = max(int(lookback_days), 30)
    response = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{symbol},day,,,{count},qfq"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code") not in (0, "0", None):
        message = payload.get("msg") if isinstance(payload, dict) else payload
        raise RuntimeError(f"tencent daily API error for {code}: {message}")
    data = payload.get("data") if isinstance(payload, dict) else None
    stock_data = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(stock_data, dict):
        raise RuntimeError(f"tencent daily history missing payload for {code}")
    rows = stock_data.get("qfqday") or stock_data.get("day") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"tencent daily history empty for {code}")

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        normalized_rows.append({
            "date": row[0],
            "open": row[1],
            "close": row[2],
            "high": row[3],
            "low": row[4],
            "volume": row[5],
            "amount": row[6] if len(row) > 6 else pd.NA,
        })
    if not normalized_rows:
        raise RuntimeError(f"tencent daily history malformed for {code}")
    df = pd.DataFrame(
        normalized_rows,
        columns=["date", "open", "close", "high", "low", "volume", "amount"],
    )
    for col in ("open", "close", "high", "low", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.tail(count).copy()


def _fetch_daily_tushare(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch forward-adjusted daily history via Tushare Pro."""
    token = _tushare_token()
    if not token:
        raise RuntimeError("tushare requires TUSHARE_TOKEN")

    import tushare as ts

    pro = ts.pro_api(token)
    _configure_tushare_client(pro, token=token)

    start_date = (datetime.now() - timedelta(days=max(lookback_days * 2, 90))).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    adj = _normalize_tushare_adj(os.getenv("TUSHARE_DAILY_ADJ", "qfq"))
    ts_code = _to_tushare_code(code)
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    if df is None or df.empty:
        raise RuntimeError(f"tushare daily history empty for {code}")
    if adj is not None:
        df = _apply_tushare_adjustment(
            df,
            pro=pro,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj=adj,
        )

    normalized = _normalize_tushare_daily_frame(df)
    return normalized.tail(max(lookback_days, 30)).copy()


def _tushare_token() -> str:
    return (
        os.getenv("TUSHARE_TOKEN", "").strip()
        or os.getenv("TUSHARE_API_TOKEN", "").strip()
    )


def _has_tushare_token() -> bool:
    return bool(_tushare_token())


def _configure_tushare_client(pro: object, *, token: str) -> None:
    try:
        setattr(pro, "_DataApi__token", token)
    except Exception:
        pass

    http_url = (
        os.getenv("TUSHARE_API_URL", "").strip()
        or os.getenv("TUSHARE_HTTP_URL", "").strip()
        or _DEFAULT_TUSHARE_HTTP_URL
    )
    try:
        setattr(pro, "_DataApi__http_url", http_url)
    except Exception:
        pass


def _normalize_tushare_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "trade_date": "date",
        "vol": "volume",
    }
    normalized = df.rename(columns=rename_map).copy()
    if "date" in normalized.columns:
        normalized["date"] = normalized["date"].astype(str)
        normalized = normalized.sort_values("date")
    return normalized


def _apply_tushare_adjustment(
    df: pd.DataFrame,
    *,
    pro: object,
    ts_code: str,
    start_date: str,
    end_date: str,
    adj: str,
) -> pd.DataFrame:
    factors = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="trade_date,adj_factor",
    )
    if factors is None or factors.empty:
        raise RuntimeError(f"tushare adj_factor empty for {ts_code}")

    merged = df.merge(factors, on="trade_date", how="left")
    merged = merged.sort_values("trade_date")
    merged["adj_factor"] = pd.to_numeric(merged["adj_factor"], errors="coerce").bfill()
    valid_factors = pd.to_numeric(factors["adj_factor"], errors="coerce").dropna()
    if valid_factors.empty:
        raise RuntimeError(f"tushare adj_factor invalid for {ts_code}")
    latest_factor = float(valid_factors.iloc[0])
    for col in ("open", "high", "low", "close"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if adj == "hfq":
            merged[col] = merged[col] * merged["adj_factor"]
        else:
            merged[col] = merged[col] * merged["adj_factor"] / latest_factor
    return merged.drop(columns=["adj_factor"])


def _normalize_tushare_adj(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    if text in {"", "none", "null", "no", "false", "0"}:
        return None
    if text not in {"qfq", "hfq"}:
        raise RuntimeError(f"unsupported TUSHARE_DAILY_ADJ: {value}")
    return text


def _fetch_daily_baostock(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch daily history via Baostock as a free fallback source.

    Baostock uses ``sh.600519`` / ``sz.000001`` style codes and exposes
    forward-adjusted prices via ``adjustflag='2'``.
    """
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("baostock not installed; pip install baostock") from exc

    global _BAOSTOCK_OUTAGE_ERROR
    if _BAOSTOCK_OUTAGE_ERROR is not None:
        raise RuntimeError(_BAOSTOCK_OUTAGE_ERROR)

    bs_code = _to_baostock_code(code)
    start_date = (datetime.now() - timedelta(days=max(lookback_days * 2, 90))).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    with _BAOSTOCK_LOCK:
        if _BAOSTOCK_OUTAGE_ERROR is not None:
            raise RuntimeError(_BAOSTOCK_OUTAGE_ERROR)

        login_result = bs.login()
        try:
            login_error_code = str(getattr(login_result, "error_code", "0"))
            if login_error_code not in {"", "0"}:
                login_error_msg = getattr(login_result, "error_msg", "")
                raise RuntimeError(f"baostock login error {login_error_code}: {login_error_msg}")

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                message = f"baostock error {rs.error_code}: {rs.error_msg}"
                if _is_baostock_network_outage(rs.error_code, rs.error_msg):
                    _BAOSTOCK_OUTAGE_ERROR = message
                raise RuntimeError(message)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    if not rows:
        raise RuntimeError(f"baostock daily history empty for {code}")

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df.tail(max(lookback_days, 30)).copy()


def _to_baostock_code(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("6", "9", "5")):
        return f"sh.{raw}"
    return f"sz.{raw}"


def _to_tushare_code(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("4", "8", "920")):
        return f"{raw}.BJ"
    if raw.startswith(("6", "9", "5")):
        return f"{raw}.SH"
    return f"{raw}.SZ"


def _to_tencent_code(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("4", "8", "920")):
        return f"bj{raw}"
    if raw.startswith(("6", "9", "5")):
        return f"sh{raw}"
    return f"sz{raw}"


def cn_code_to_yfinance_symbol(code: str) -> str:
    """Map an A-share code to its Yahoo Finance ticker (exchange suffix).

    Single source of truth for the exchange-prefix rule already used by the
    tushare/tencent/baostock code converters above: Beijing Exchange codes
    ("4"/"8"/"920"-prefixed) must be checked before the Shanghai check,
    since "920xxx" would otherwise also match a bare "9" prefix.
    """
    raw = str(code).strip().zfill(6)
    if raw.startswith(("4", "8", "920")):
        return f"{raw}.BJ"
    if raw.startswith(("6", "9", "5")):
        return f"{raw}.SS"
    return f"{raw}.SZ"


def _is_baostock_network_outage(error_code: object, error_msg: object) -> bool:
    code = str(error_code)
    message = str(error_msg)
    return code in {"10002007"} or "网络" in message or "接收" in message


def compute_daily_features(hist: pd.DataFrame) -> dict[str, object]:
    """Compute compact trend/reversal features from a daily K-line DataFrame."""
    df = _normalize_daily_history(hist)
    if df.empty:
        raise RuntimeError("daily history is empty after normalization")

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError("daily history has no valid close price")

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    last_close = float(close.iloc[-1])
    last_ma5 = _last_float(ma5)
    last_ma20 = _last_float(ma20)
    last_ma60 = _last_float(ma60)
    shape = _compute_shape_features(df, last_close=last_close, last_ma20=last_ma20)

    lookback_idx = max(0, len(close) - 61)
    base_close = float(close.iloc[lookback_idx])
    change_60d = (last_close / base_close - 1.0) * 100 if base_close > 0 else None

    macd_status = _compute_macd_status(close)
    rsi_value = _compute_rsi(close)
    rsi_status = _classify_rsi(rsi_value)
    ma_bullish = _is_true(last_ma5 is not None and last_ma20 is not None and last_ma60 is not None
                          and last_ma5 >= last_ma20 >= last_ma60)
    price_above_ma20 = _is_true(last_ma20 is not None and last_close >= last_ma20)
    signal_score = _compute_signal_score(
        change_60d=change_60d,
        ma_bullish=ma_bullish,
        price_above_ma20=price_above_ma20,
        macd_status=macd_status,
        rsi_status=rsi_status,
    )

    return {
        "daily_data_points": int(len(close)),
        "change_60d": None if change_60d is None else round(float(change_60d), 4),
        "ma5": last_ma5,
        "ma20": last_ma20,
        "ma60": last_ma60,
        "ma_bullish": ma_bullish,
        "price_above_ma20": price_above_ma20,
        "macd_status": macd_status,
        "rsi_status": rsi_status,
        "rsi14": None if rsi_value is None else round(float(rsi_value), 4),
        "signal_score": round(float(signal_score), 4),
        **shape,
    }


def _normalize_daily_history(hist: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = hist.rename(columns=rename_map).copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    if "close" not in df.columns:
        raise RuntimeError("daily history has no close column")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).copy()
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
        else:
            df[col] = df[col].fillna(df["close"])
    return df


def _compute_shape_features(
    df: pd.DataFrame,
    *,
    last_close: float,
    last_ma20: float | None,
) -> dict[str, object]:
    previous = df.iloc[:-1].tail(20)
    recent = df.tail(20)
    last = df.iloc[-1]

    prev_high_20d = _series_max(previous["high"]) if "high" in previous.columns else None
    range_20d_pct = _range_pct(recent)
    breakout_20d_pct = (
        (last_close / prev_high_20d - 1.0) * 100
        if prev_high_20d is not None and prev_high_20d > 0
        else None
    )
    volume_ratio_20d = _volume_ratio_20d(df)
    body_pct = _body_pct(last)
    pullback_to_ma20_pct = (
        (last_close / last_ma20 - 1.0) * 100
        if last_ma20 is not None and last_ma20 > 0
        else None
    )

    return {
        "prev_high_20d": _round_or_none(prev_high_20d),
        "range_20d_pct": _round_or_none(range_20d_pct),
        "breakout_20d_pct": _round_or_none(breakout_20d_pct),
        "volume_ratio_20d": _round_or_none(volume_ratio_20d),
        "body_pct": _round_or_none(body_pct),
        "pullback_to_ma20_pct": _round_or_none(pullback_to_ma20_pct),
        "consolidation_days_20d": _consolidation_days(previous),
    }


def _series_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _range_pct(df: pd.DataFrame) -> float | None:
    if "high" not in df.columns or "low" not in df.columns:
        return None
    high = pd.to_numeric(df["high"], errors="coerce").dropna()
    low = pd.to_numeric(df["low"], errors="coerce").dropna()
    if high.empty or low.empty:
        return None
    low_min = float(low.min())
    if low_min <= 0:
        return None
    return (float(high.max()) / low_min - 1.0) * 100


def _volume_ratio_20d(df: pd.DataFrame) -> float | None:
    if "volume" not in df.columns:
        return None
    volume = pd.to_numeric(df["volume"], errors="coerce")
    if len(volume) < 2 or pd.isna(volume.iloc[-1]):
        return None
    previous = volume.iloc[:-1].tail(20).dropna()
    if previous.empty:
        return None
    base = float(previous.mean())
    if base <= 0:
        return None
    return float(volume.iloc[-1]) / base


def _consolidation_days(previous: pd.DataFrame, *, max_range_pct: float = 12.0) -> int | None:
    if previous.empty or "high" not in previous.columns or "low" not in previous.columns:
        return None
    for days in range(min(len(previous), 20), 1, -1):
        window = previous.tail(days)
        range_pct = _range_pct(window)
        if range_pct is not None and range_pct <= max_range_pct:
            return int(days)
    return 0


def _body_pct(row: pd.Series) -> float | None:
    open_price = row.get("open")
    close_price = row.get("close")
    if pd.isna(open_price) or pd.isna(close_price) or float(open_price) <= 0:
        return None
    return (float(close_price) / float(open_price) - 1.0) * 100


def _round_or_none(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _compute_macd_status(close: pd.Series) -> str:
    if len(close) < 35:
        return "neutral"
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    last_diff = float(diff.iloc[-1])
    last_dea = float(dea.iloc[-1])
    if last_diff > last_dea and last_diff > 0:
        return "bullish"
    if last_diff < last_dea and last_diff < 0:
        return "bearish"
    return "neutral"


def _compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _classify_rsi(value: float | None) -> str:
    if value is None:
        return "neutral"
    if value <= 35:
        return "oversold"
    if value >= 70:
        return "overbought"
    return "neutral"


def _compute_signal_score(
    *,
    change_60d: float | None,
    ma_bullish: bool,
    price_above_ma20: bool,
    macd_status: str,
    rsi_status: str,
) -> float:
    score = 50.0
    if ma_bullish:
        score += 14
    if price_above_ma20:
        score += 10
    if macd_status == "bullish":
        score += 12
    elif macd_status == "bearish":
        score -= 12
    if change_60d is not None:
        if 0 <= change_60d <= 35:
            score += min(change_60d * 0.35, 12)
        elif change_60d > 60:
            score -= min((change_60d - 60) * 0.20, 12)
        elif change_60d < -25:
            score -= min(abs(change_60d + 25) * 0.25, 10)
    if rsi_status == "oversold":
        score += 4
    elif rsi_status == "overbought":
        score -= 6
    return max(0.0, min(score, 100.0))


def _last_float(series: pd.Series) -> float | None:
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _is_true(value: bool) -> bool:
    return bool(value)

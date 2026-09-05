# -*- coding: utf-8 -*-
"""US equity snapshot via yfinance.

Pluggable adapter for AlphaSift's L1 pipeline. Fetches a configurable
equity universe and returns the standard snapshot DataFrame schema.

HK is not supported yet: there is no HK universe source or ticker
configuration path, so ``market="hk"`` is rejected at the pipeline level
rather than silently screening the US pool.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

logger = logging.getLogger(__name__)

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_US_LISTED_EXCHANGES = ("NMS", "NGM", "NCM", "NYQ", "ASE")
_US_SCREEN_PAGE_SIZE = 250
_US_SCREEN_MAX_RETRIES = 3

_DEFAULT_US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "AVGO", "JPM", "LLY", "V", "MA", "UNH", "XOM", "COST", "HD", "PG",
    "JNJ", "ABBV", "WMT", "NFLX", "BAC", "KO", "CRM", "CVX", "MRK",
    "PEP", "AMD", "TMO", "LIN", "ACN", "CSCO", "MCD", "ABT", "ADBE",
    "WFC", "GE", "DHR", "TXN", "PM", "ISRG", "MS", "NEE", "INTU",
    "DIS", "QCOM", "CAT", "NOW",
]


def fetch_us_universe(source: str = "auto") -> list[str]:
    """Return US equity tickers from a full or explicitly limited source."""
    src = source.lower()
    if src in {"auto", "full", "yahoo_screen"}:
        quotes = _fetch_yahoo_screen_quotes()
        tickers = [str(item.get("symbol") or "").strip() for item in quotes]
        tickers = [ticker for ticker in tickers if ticker]
        if not tickers:
            raise RuntimeError("Yahoo listed-equity screen returned no US tickers")
        logger.info("US universe from yahoo_screen: %d tickers", len(tickers))
        return tickers

    if src == "sp500":
        return _fetch_sp500_tickers()
    elif src == "env":
        raw = os.getenv("ALPHASIFT_US_TICKERS", "").strip()
        if not raw:
            raise ValueError("ALPHASIFT_US_TICKERS not set")
        return [t.strip() for t in raw.split(",") if t.strip()]
    elif src == "default":
        return list(_DEFAULT_US_UNIVERSE)
    else:
        raise ValueError(f"Unknown US universe source: {source}")


def _fetch_yahoo_screen_quotes() -> list[dict]:
    """Fetch every exchange-listed US equity, rejecting partial pagination."""
    import yfinance as yf

    query = yf.EquityQuery("and", [
        yf.EquityQuery("eq", ["region", "us"]),
        yf.EquityQuery("is-in", ["exchange", *_US_LISTED_EXCHANGES]),
    ])
    quotes: list[dict] = []
    expected_total: int | None = None
    offset = 0

    while expected_total is None or offset < expected_total:
        response = None
        last_error: Exception | None = None
        for attempt in range(_US_SCREEN_MAX_RETRIES):
            try:
                response = yf.screen(
                    query,
                    offset=offset,
                    size=_US_SCREEN_PAGE_SIZE,
                    sortField="ticker",
                    sortAsc=True,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < _US_SCREEN_MAX_RETRIES:
                    time.sleep(0.25 * (attempt + 1))
        if response is None:
            raise RuntimeError(
                f"Yahoo US equity screen failed at offset {offset}: {last_error}"
            ) from last_error

        page = response.get("quotes") or []
        page_total = int(response.get("total") or 0)
        if expected_total is None:
            expected_total = page_total
        elif page_total != expected_total:
            logger.warning(
                "Yahoo US equity total changed during pagination: %d -> %d",
                expected_total,
                page_total,
            )
            expected_total = max(expected_total, page_total)
        if not page and offset < expected_total:
            raise RuntimeError(
                f"Yahoo US equity screen returned a partial result at offset {offset} "
                f"of {expected_total}"
            )
        quotes.extend(item for item in page if isinstance(item, dict))
        offset += len(page)

    deduplicated: dict[str, dict] = {}
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").strip()
        if symbol:
            deduplicated[symbol] = quote
    if expected_total and len(deduplicated) < expected_total:
        raise RuntimeError(
            "Yahoo US equity screen was incomplete after de-duplication: "
            f"expected {expected_total}, received {len(deduplicated)}"
        )
    return list(deduplicated.values())


def _screen_quotes_to_snapshot(quotes: list[dict]) -> pd.DataFrame:
    """Map Yahoo screener records to AlphaSift's snapshot schema."""
    rows = []
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").strip()
        price = quote.get("regularMarketPrice")
        if not symbol or price is None:
            continue
        volume = quote.get("regularMarketVolume") or 0
        average_volume = quote.get("averageDailyVolume3Month") or 0
        shares = quote.get("sharesOutstanding") or quote.get("impliedSharesOutstanding") or 0
        rows.append({
            "code": symbol,
            "name": quote.get("shortName") or quote.get("longName") or symbol,
            "price": price,
            "change_pct": quote.get("regularMarketChangePercent"),
            "amount": float(volume) * float(price),
            "total_mv": quote.get("marketCap"),
            "circ_mv": quote.get("marketCap"),
            "pe_ratio": quote.get("trailingPE"),
            "pb_ratio": quote.get("priceToBook"),
            "volume_ratio": round(float(volume) / float(average_volume), 2) if average_volume else None,
            "turnover_rate": round(float(volume) / float(shares) * 100, 4) if shares else None,
            "industry": quote.get("industry") or quote.get("sector") or "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Yahoo listed-equity screen returned no valid quote rows")
    numeric_cols = [
        "price", "change_pct", "amount", "total_mv", "circ_mv",
        "pe_ratio", "pb_ratio", "volume_ratio", "turnover_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].reset_index(drop=True)
    df.attrs["snapshot_source"] = "yahoo_screen"
    return df


def _fetch_sp500_tickers() -> list[str]:
    tables = pd.read_html(_SP500_WIKI_URL)
    for tbl in tables:
        if "Symbol" in tbl.columns:
            return sorted(tbl["Symbol"].dropna().str.strip().str.replace(".", "-", regex=False).tolist())
    raise RuntimeError("Could not find Symbol column in S&P 500 Wikipedia table")


def fetch_us_snapshot(
    tickers: list[str] | None = None,
    *,
    universe_source: str = "auto",
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch US equity snapshot in AlphaSift standard schema.

    The default path paginates Yahoo's complete listed-equity screen.
    Explicit tickers and explicitly limited universe sources retain the
    historical-price path. The returned frame uses the standard snapshot
    columns consumed by AlphaSift filters.
    """
    import yfinance as yf

    if tickers is None and universe_source.lower() in {"auto", "full", "yahoo_screen"}:
        df = _screen_quotes_to_snapshot(_fetch_yahoo_screen_quotes())
        logger.info("US snapshot: %d rows from yahoo_screen", len(df))
        return df

    if tickers is None:
        tickers = fetch_us_universe(universe_source)

    logger.info("Fetching US snapshot for %d tickers", len(tickers))

    hist_end = pd.Timestamp.now().normalize()
    hist_start = hist_end - pd.Timedelta(days=30)
    data = yf.download(
        tickers,
        start=hist_start.strftime("%Y-%m-%d"),
        end=hist_end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    rows = []

    def _process_ticker(ticker: str) -> dict | None:
        try:
            if len(tickers) == 1:
                hist = data.copy()
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel("Ticker")
            else:
                if ticker not in data.columns.get_level_values(0):
                    return None
                hist = data[ticker].copy()
            if hist.empty:
                return None

            hist = hist[hist["Close"].notna()]
            if len(hist) < 2:
                return None

            latest = hist.iloc[-1]
            prev = hist.iloc[-2]
            price = float(latest["Close"])
            prev_close = float(prev["Close"])
            volume = float(latest["Volume"])
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            vol_20d = float(hist["Volume"].tail(20).mean())
            volume_ratio = (volume / vol_20d) if vol_20d > 0 else 1.0

            info = yf.Ticker(ticker).fast_info
            market_cap = getattr(info, "market_cap", None) or 0
            shares = getattr(info, "shares", None) or 0
            turnover_rate = (volume / shares * 100) if shares > 0 else 0.0

            return {
                "code": ticker,
                "name": ticker,
                "price": price,
                "change_pct": round(change_pct, 2),
                "amount": round(volume * price, 0),
                "total_mv": market_cap,
                "circ_mv": market_cap,
                "pe_ratio": None,
                "pb_ratio": None,
                "volume_ratio": round(volume_ratio, 2),
                "turnover_rate": round(turnover_rate, 4),
                "industry": "",
            }
        except Exception as e:
            logger.debug("Failed to process %s: %s", ticker, e)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    if not rows:
        raise RuntimeError("yfinance returned no valid data for any ticker")

    df = pd.DataFrame(rows)

    numeric_cols = [
        "price", "change_pct", "amount", "total_mv", "circ_mv",
        "pe_ratio", "pb_ratio", "volume_ratio", "turnover_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    _enrich_info_fields(df)

    df.attrs["snapshot_source"] = "yfinance"
    logger.info("US snapshot: %d rows from yfinance", len(df))
    return df


def _enrich_info_fields(df: pd.DataFrame) -> None:
    """Best-effort enrichment of pe_ratio, pb_ratio, industry from yfinance info."""
    import yfinance as yf

    needs_pe = df["pe_ratio"].isna().sum() > len(df) * 0.5
    if not needs_pe:
        return

    for idx in df.index:
        ticker = df.at[idx, "code"]
        try:
            info = yf.Ticker(ticker).info
            if pd.isna(df.at[idx, "pe_ratio"]) or df.at[idx, "pe_ratio"] == 0:
                df.at[idx, "pe_ratio"] = info.get("trailingPE")
            if pd.isna(df.at[idx, "pb_ratio"]) or df.at[idx, "pb_ratio"] == 0:
                df.at[idx, "pb_ratio"] = info.get("priceToBook")
            if not df.at[idx, "industry"]:
                df.at[idx, "industry"] = info.get("industry", "")
            if not df.at[idx, "name"] or df.at[idx, "name"] == ticker:
                df.at[idx, "name"] = info.get("shortName", ticker)
        except Exception:
            pass


def fetch_daily_history_yfinance(
    ticker: str,
    *,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """Fetch daily OHLCV history for a US ticker via yfinance.

    Returns a DataFrame with columns: date, open, high, low, close, volume
    matching the schema expected by alphasift.daily's enrichment logic.
    """
    import yfinance as yf

    # Yahoo's end date is exclusive. Include today's completed session; never
    # silently lag one session merely because the machine is in US time.
    timezone = "Asia/Shanghai" if ticker.endswith((".SS", ".SZ", ".BJ")) else "America/New_York"
    local_now = pd.Timestamp.now(tz=timezone)
    end = local_now.normalize().tz_localize(None) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=max(lookback_days * 2, 180))
    history_range = {"period": "max"} if lookback_days == 0 else {
        "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")
    }
    hist = yf.download(ticker, **history_range, auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance daily history empty for {ticker}")

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.droplevel("Ticker")

    from alphasift.freshness import latest_completed_session

    completed_session = latest_completed_session(
        "cn" if timezone == "Asia/Shanghai" else "us", local_now
    )
    hist = hist[hist.index.strftime("%Y-%m-%d") <= completed_session]
    hist = hist.tail(max(lookback_days, 30)).copy() if lookback_days else hist.copy()
    hist = hist.rename(columns={
        "Open": "开盘", "High": "最高", "Low": "最低",
        "Close": "收盘", "Volume": "成交量",
    })
    hist.index.name = "日期"
    hist = hist.reset_index()
    hist.attrs["data_source"] = "Yahoo Finance/yfinance adjusted daily OHLCV"
    hist.attrs["retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return hist

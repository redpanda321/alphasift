"""Independent five-calendar-year / all-available-history lifecycle comparison.

Run with explicit symbols (sample) or --market cn/us (filtered market universe).
All prices in both windows share the adjustment basis of a single fresh download.
Agreement measures temporal robustness, not independent-provider verification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path

import pandas as pd

from alphasift.daily import _normalize_daily_history
from alphasift.lifecycle import compute_lifecycle_features, _pivots
from alphasift.lifecycle_contract import WINDOW_YEARS, STAGES, strategy_contract


def classify_window(history: pd.DataFrame) -> dict:
    df = _normalize_daily_history(history).reset_index(drop=True)
    f = compute_lifecycle_features(df, profile={"lookback_days": 0})
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
    d_date = pd.Timestamp(f["cycle_high_date"])
    d_age = int((dates > d_date).sum())
    dd = f["drawdown_from_cycle_high_pct"]
    scores = {"A": f["a_score"], "H": f["h_score"], "B": 0.0, "D": 0.0, "E": 0.0}
    eligible = {
        "A": f["a_eligible"],
        "H": f["h_eligible"],
        "B": False,
        "D": False,
        "E": False,
    }
    evidence = {
        "D_date": f["cycle_high_date"],
        "D_high": f["cycle_high"],
        "D_age_sessions": d_age,
        "drawdown_pct": dd,
    }
    # D is only a high-zone candidate; a final top cannot be confirmed in real time.
    eligible["D"] = f["prior_runup_pct"] >= 100 and dd <= 12 and d_age <= 63
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
    start = cutoff - pd.DateOffset(years=WINDOW_YEARS)
    full = classify_window(df)
    five = classify_window(df.loc[dates >= start])
    # Weekend/holiday allowance at the five-year boundary. Does not establish
    # completeness of every session, or availability back to the IPO.
    five_span = dates.min() <= start + pd.Timedelta(days=7)
    extra_history = bool((dates < start).any())
    comparable = bool(five_span and extra_history)
    same = five["stage"] == full["stage"]
    status = (
        "INSUFFICIENT_DISTINCT_HISTORY"
        if not comparable
        else "AGREEMENT"
        if same and five["stage"] in STAGES
        else "NO_MATCH"
        if same and five["stage"] == "UNCLASSIFIED"
        else "CONFLICT"
    )
    stage = five["stage"] if status == "AGREEMENT" else None
    return {
        "strategy": strategy_contract(),
        "as_of": as_of,
        "five_year_boundary": str(start.date()),
        "five_year_span_available": bool(five_span),
        "full_has_older_data": extra_history,
        "status": status,
        "consensus_stage": stage,
        "consensus_score": min(five["scores"][stage], full["scores"][stage])
        if stage
        else None,
        "five_year": five,
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
            else [
                c
                + (
                    ".BJ"
                    if c.startswith(("4", "8", "920"))
                    else ".SS"
                    if c.startswith("6")
                    else ".SZ"
                )
                for c in codes
            ]
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
    workers = max(1, min(int(os.getenv("ALPHASIFT_HISTORY_WORKERS", "24")), 48))
    print(f"{args.market}: workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for row in pool.map(one, symbols):
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

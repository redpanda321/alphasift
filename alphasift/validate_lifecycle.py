"""Reproducible live/local OHLCV validation, not a claim of a full-market scan.

python -m alphasift.validate_lifecycle --as-of YYYY-MM-DD --output data/lifecycle-validation.json
Use --input-dir with cn_CODE.csv/us_CODE.csv to replay saved histories offline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alphasift.daily import _normalize_daily_history, fetch_daily_history
from alphasift.lifecycle import compute_lifecycle_features
from alphasift.strategy import load_strategy

SAMPLES = {
    "cn": [
        "600519",
        "000858",
        "601318",
        "600036",
        "000063",
        "002415",
        "002230",
        "002475",
        "000725",
        "000100",
        "002371",
        "603501",
        "688008",
        "688981",
        "688012",
        "688041",
        "300308",
        "300502",
        "300394",
        "300124",
        "002049",
        "300418",
        "600570",
        "600588",
    ],
    "us": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "INTC",
        "MU",
        "AVGO",
        "QCOM",
        "TXN",
        "AMAT",
        "LRCX",
        "KLAC",
        "MRVL",
        "CSCO",
        "ANET",
        "COHR",
        "LITE",
        "ADBE",
        "CRM",
        "NOW",
        "SNOW",
        "PLTR",
        "U",
        "PATH",
        "PYPL",
        "ZM",
        "DOCU",
        "SHOP",
        "AMZN",
        "GOOGL",
        "META",
        "DELL",
        "SMCI",
        "WDC",
        "STX",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        required=True,
        help="Expected last completed session date; stale rows fail freshness",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--cn", default=",".join(SAMPLES["cn"]))
    parser.add_argument("--us", default=",".join(SAMPLES["us"]))
    args = parser.parse_args()
    expected = pd.Timestamp(args.as_of).date().isoformat()
    profile = load_strategy(
        Path(__file__).parent / "strategies/lifecycle_ah.yaml"
    ).screening.lifecycle_profile
    report = {
        "strategy": "lifecycle_ah",
        "version": "1.2",
        "expected_session": expected,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "explicit validation sample, not full market",
        "fundamentals": "unverified; no buy/add recommendations or fundamental top 10",
        "rows": [],
        "errors": [],
    }
    history_dir = args.output.parent / (args.output.stem + "-histories")
    history_dir.mkdir(parents=True, exist_ok=True)
    for market in ("cn", "us"):
        for code in filter(None, getattr(args, market).split(",")):
            code = code.strip()
            try:
                if args.input_dir:
                    hist = pd.read_csv(args.input_dir / f"{market}_{code}.csv")
                    source = "local CSV replay"
                else:
                    symbol = (
                        code
                        if market == "us"
                        else code + (".SS" if code.startswith("6") else ".SZ")
                    )
                    hist = fetch_daily_history(
                        symbol, lookback_days=int(profile["lookback_days"]), source="yfinance", retries=0,
                        cache_dir=None,
                    )
                    source = "Yahoo Finance/yfinance adjusted daily OHLCV"
                hist = _normalize_daily_history(hist)
                hist = hist[
                    pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d") <= expected
                ].copy()
                hist.to_csv(history_dir / f"{market}_{code}.csv", index=False)
                result = compute_lifecycle_features(hist, profile=profile)
                row = {
                    "market": market,
                    "code": code,
                    "source": source,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    **result,
                    "data_timestamp": None,
                    "timestamp_note": "daily session date only; exact quote time not supplied",
                    "fresh": result["data_as_of"] == expected,
                }
                report["rows"].append(row)
                print(
                    f"{market} {code}: {result['data_as_of']} A={result['a_score']}({result['a_eligible']}) H={result['h_score']}({result['h_eligible']})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - report per-symbol failures without hiding coverage gaps
                report["errors"].append(
                    {"market": market, "code": code, "error": str(exc)}
                )
                print(f"{market} {code}: ERROR {exc}", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        market: {
            "requested": len(list(filter(None, getattr(args, market).split(",")))),
            "success": sum(r["market"] == market for r in report["rows"]),
            "fresh": sum(r["market"] == market and r["fresh"] for r in report["rows"]),
        }
        for market in ("cn", "us")
    }
    for stage in ("a", "h"):
        report[f"top_{stage}"] = sorted(
            [
                r
                for r in report["rows"]
                if r["fresh"]
                and r[f"{stage}_eligible"]
                and r[f"{stage}_score"] >= profile["min_score"]
            ],
            key=lambda r: r[f"{stage}_score"],
            reverse=True,
        )[:20]
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return (
        0
        if all(
            v["fresh"] == v["requested"] and v["requested"] > 0
            for v in report["summary"].values()
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

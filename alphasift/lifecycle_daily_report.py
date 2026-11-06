"""Build a compact, production-safe financial overlay for lifecycle scans."""

from __future__ import annotations

import csv
import html
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd


def _value(frame: pd.DataFrame, key: str, date: Any) -> float | None:
    if key not in frame.index or date not in frame.columns:
        return None
    result = frame.at[key, date]
    return float(result) if pd.notna(result) else None


def _dates(frame: pd.DataFrame, cutoff: pd.Timestamp) -> list[Any]:
    if frame is None or frame.empty:
        return []
    return sorted((d for d in frame.columns if pd.Timestamp(d) <= cutoff), reverse=True)


def candidate_rows(scan: dict[str, Any], market: str) -> list[dict[str, Any]]:
    rows = []
    for item in scan["rows"]:
        if item.get("status") != "AGREEMENT":
            continue
        symbol = item["symbol"]
        code = symbol.rsplit(".", 1)[0] if market == "cn" else symbol
        stage = item["consensus_stage"]
        rows.append(
            {
                "market": market,
                "symbol": symbol,
                "code": code,
                "name": code,
                "stage": stage,
                "price": item["five_year"]["price"],
                "as_of": item["as_of"],
                "score": item["consensus_score"],
                "window_years": item.get("window_years"),
                "five_score": item["five_year"]["scores"][stage],
                "full_score": item["full_history"]["scores"][stage],
                "full_start": item["full_history"]["history_start"],
            }
        )
    return rows


def _fetch_ticker_snapshot(symbol: str, *, retries: int = 3):
    """Fetch a yfinance Ticker's statements/info, retrying transient failures.

    This is a network-heavy call (income statement, cash flow, balance sheet,
    and profile info each trigger their own request); a single retryable
    failure otherwise drops the whole candidate — including its market cap —
    from the daily report, not just this one field.
    """
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            ticker = yf.Ticker(symbol)
            income = ticker.income_stmt
            cashflow = ticker.cash_flow
            balance = ticker.quarterly_balance_sheet
            if balance.empty:
                balance = ticker.balance_sheet
            info = ticker.get_info()
            return ticker, income, cashflow, balance, info
        except Exception as exc:  # noqa: BLE001 -- retried below; final attempt re-raises
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(1.5 * (2**attempt), 10) + random.uniform(0, 0.5))
    raise last_error  # type: ignore[misc]


def fetch_financial(
    stock: dict[str, Any], *, now: pd.Timestamp | None = None
) -> dict[str, Any]:
    instant = now or pd.Timestamp.now(tz="UTC")
    cutoff = (
        instant.tz_localize(None).normalize() if instant.tzinfo else instant.normalize()
    )
    result = dict(stock)
    result["retrieved_at"] = instant.isoformat()
    result["source"] = f"https://finance.yahoo.com/quote/{stock['symbol']}/financials/"
    try:
        _ticker, income, cashflow, balance, info = _fetch_ticker_snapshot(stock["symbol"])
        result.update(
            name=info.get("shortName") or info.get("longName") or stock["code"],
            currency=info.get("financialCurrency"),
            sector=info.get("sector") or "Unclassified",
            industry=info.get("industry") or info.get("sector") or "Unclassified",
            market_cap=info.get("marketCap"),
        )
        financial = result["sector"] == "Financial Services"
        result["financial_company"] = financial
        common = [
            d
            for d in _dates(income, cutoff)
            if d in cashflow.columns and _value(income, "Net Income", d) is not None
        ]
        annual = []
        for date in common:
            profit = _value(income, "Net Income", date)
            cfo = _value(cashflow, "Operating Cash Flow", date)
            capex = _value(cashflow, "Capital Expenditure", date)
            annual.append(
                {
                    "period": pd.Timestamp(date).date().isoformat(),
                    "net_income": profit,
                    "revenue": _value(income, "Total Revenue", date),
                    "operating_cash_flow": cfo,
                    "free_cash_flow": cfo - abs(capex)
                    if cfo is not None and capex is not None
                    else None,
                    "stock_based_compensation": _value(
                        cashflow, "Stock Based Compensation", date
                    ),
                }
            )
        balance_dates = _dates(balance, cutoff)
        if not annual or not balance_dates:
            return {
                **result,
                "status": "incomplete",
                "error": "missing income/cashflow or balance sheet",
            }
        result.update(annual[0], annual_history=annual)
        balance_date = balance_dates[0]
        debt = _value(balance, "Total Debt", balance_date)
        cash = _value(balance, "Cash And Cash Equivalents", balance_date)
        result.update(
            balance_period=pd.Timestamp(balance_date).date().isoformat(),
            total_debt=debt,
            cash=cash,
            net_debt=debt - cash if debt is not None and cash is not None else None,
            equity=_value(balance, "Stockholders Equity", balance_date),
        )
        prior = next(
            (
                a
                for a in annual[1:]
                if 330
                <= (pd.Timestamp(annual[0]["period"]) - pd.Timestamp(a["period"])).days
                <= 400
            ),
            None,
        )
        result["profit_yoy_pct"] = (
            (annual[0]["net_income"] / prior["net_income"] - 1) * 100
            if prior and prior["net_income"] and prior["net_income"] > 0
            else None
        )
        result["net_margin_pct"] = (
            result["net_income"] / result["revenue"] * 100
            if result.get("revenue") and result["revenue"] > 0
            else None
        )
        result["status"] = "ok"
        return result
    except Exception as exc:  # noqa: BLE001 -- one unavailable ticker must not abort the market scan
        return {**result, "status": "unavailable", "error": str(exc)}


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def score_financial(
    row: dict[str, Any], *, now: pd.Timestamp | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    instant = (now or pd.Timestamp.now(tz="UTC")).tz_localize(None)
    if row.get("status") != "ok":
        return None, "financial data unavailable or incomplete"
    if row.get("financial_company"):
        return None, "financial company requires capital and asset-quality metrics"
    required = (
        "free_cash_flow",
        "operating_cash_flow",
        "net_income",
        "revenue",
        "total_debt",
        "cash",
        "equity",
        "profit_yoy_pct",
        "currency",
    )
    if any(row.get(key) is None for key in required):
        return None, "missing required metrics"
    if any(
        row[key] <= 0
        for key in (
            "free_cash_flow",
            "operating_cash_flow",
            "net_income",
            "revenue",
            "equity",
        )
    ):
        return None, "nonpositive latest FCF/CFO/profit/revenue/equity"
    if (instant - pd.Timestamp(row["period"])).days > 550 or (
        instant - pd.Timestamp(row["balance_period"])
    ).days > 200:
        return None, "statement age gate"
    history = row["annual_history"][:3]
    if len(history) < 3 or any(
        a.get("free_cash_flow") is None or a.get("net_income") is None for a in history
    ):
        return None, "missing three-year history"
    if any(
        not 330 <= (pd.Timestamp(a["period"]) - pd.Timestamp(b["period"])).days <= 400
        for a, b in pairwise(history)
    ):
        return None, "nonconsecutive annual periods"
    fcf, profit, revenue = row["free_cash_flow"], row["net_income"], row["revenue"]
    prior_fcf = history[1]["free_cash_flow"]
    fcf_growth = fcf / prior_fcf - 1 if prior_fcf > 0 else None
    net_debt = row["total_debt"] - row["cash"]
    cash_score = 15 * _clip(fcf / revenue / 0.20)
    cash_score += 10 * _clip(row["operating_cash_flow"] / profit)
    cash_score += 10 * sum(a["free_cash_flow"] > 0 for a in history) / 3
    cash_score += 5 * _clip((fcf_growth + 0.20) / 0.40) if fcf_growth is not None else 0
    debt_score = 30 * (1 - _clip(max(net_debt, 0) / fcf / 5))
    profit_score = 10 * _clip(profit / revenue / 0.20)
    profit_score += 10 * _clip((row["profit_yoy_pct"] + 20) / 40)
    profit_score += 10 * sum(a["net_income"] > 0 for a in history) / 3
    penalties = 5 * (row["profit_yoy_pct"] < -20)
    penalties += 5 * (fcf_growth is not None and fcf_growth < -0.20)
    penalties += 5 * (
        row.get("stock_based_compensation") is not None
        and row["stock_based_compensation"] / revenue > 0.10
    )
    result = dict(row)
    result.update(
        fundamental_score=round(cash_score + debt_score + profit_score - penalties, 2),
        fcf_component=round(cash_score, 2),
        debt_component=round(debt_score, 2),
        profit_component=round(profit_score, 2),
        penalties=penalties,
        fcf_margin_pct=round(fcf / revenue * 100, 2),
    )
    result["combined_score"] = round(
        0.8 * result["fundamental_score"] + 0.2 * float(row["score"]), 2
    )
    return result, None


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "market",
        "sector",
        "industry",
        "stage",
        "code",
        "symbol",
        "name",
        "price",
        "market_cap",
        "as_of",
        "score",
        "window_years",
        "five_score",
        "full_score",
        "full_start",
        "combined_score",
        "fundamental_score",
        "fcf_component",
        "debt_component",
        "profit_component",
        "penalties",
        "currency",
        "period",
        "balance_period",
        "free_cash_flow",
        "total_debt",
        "cash",
        "net_debt",
        "net_margin_pct",
        "profit_yoy_pct",
        "retrieved_at",
        "source",
    )
    return {key: row.get(key) for key in keys}


def build_report(
    scans: dict[str, dict[str, Any]],
    *,
    max_workers: int = 2,
    fetcher=fetch_financial,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    stocks = [
        row for market, scan in scans.items() for row in candidate_rows(scan, market)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        financials = list(pool.map(fetcher, stocks))
    scored, exclusions = [], Counter()
    for row in financials:
        result, reason = score_financial(row, now=now)
        if result:
            scored.append(result)
        else:
            exclusions[reason or "unknown"] += 1
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(row["market"], row["industry"], row["stage"])].append(row)
    top5, summaries = [], []
    for (market, industry, stage), group in sorted(groups.items()):
        chosen = sorted(group, key=lambda r: (-r["combined_score"], r["code"]))[:5]
        for rank, row in enumerate(chosen, 1):
            top5.append({**compact_row(row), "rank": rank})
        summaries.append(
            {
                "market": market,
                "industry": industry,
                "stage": stage,
                "eligible": len(group),
                "selected": len(chosen),
            }
        )
    agreements_by_market = Counter(row["market"] for row in stocks)
    eligible_by_market = Counter(row["market"] for row in scored)
    selected_by_market = Counter(row["market"] for row in top5)
    coverage = {
        market: {
            "snapshot": scan.get("snapshot_count"),
            "attempted": scan["attempted"],
            "as_of": scan["as_of"],
            "statuses": dict(Counter(r["status"] for r in scan["rows"])),
        }
        for market, scan in scans.items()
    }
    stage_counts = {
        market: dict(
            Counter(
                r["consensus_stage"] for r in scan["rows"] if r["status"] == "AGREEMENT"
            )
        )
        for market, scan in scans.items()
    }
    market_funnel = {
        market: {
            "snapshot": scan.get("snapshot_count"),
            "attempted": scan["attempted"],
            "agreement": agreements_by_market[market],
            "financially_eligible": eligible_by_market[market],
            "selected": selected_by_market[market],
        }
        for market, scan in scans.items()
    }
    return {
        "generated_at": (now or pd.Timestamp.now(tz="UTC")).isoformat(),
        "strategy": next(iter(scans.values())).get("strategy"),
        "scope": len(stocks),
        "eligible": len(scored),
        "selected": len(top5),
        "coverage": coverage,
        "market_funnel": market_funnel,
        "stage_counts": stage_counts,
        "exclusions": dict(exclusions),
        "summary": summaries,
        "top5": top5,
        "disclaimer": "Research candidates only. Not financial guidance or execution instructions.",
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lifecycle-industry-top5.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if report["top5"]:
        with (output_dir / "lifecycle-industry-top5.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["top5"][0]))
            writer.writeheader()
            writer.writerows(report["top5"])
    headings = (
        "market",
        "industry",
        "rank",
        "code",
        "name",
        "stage",
        "combined_score",
        "fundamental_score",
        "score",
        "price",
        "as_of",
        "period",
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in headings)
        + "</tr>"
        for row in report["top5"]
    )
    document = (
        "<!doctype html><meta charset='utf-8'><title>AlphaSift Daily</title>"
        "<style>body{font:14px system-ui;padding:24px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}</style>"
        f"<h1>AlphaSift Daily</h1><p>{html.escape(report['disclaimer'])}</p><table><thead><tr>"
        + "".join(f"<th>{key}</th>" for key in headings)
        + f"</tr></thead><tbody>{body}</tbody></table>"
    )
    (output_dir / "lifecycle-industry-top5.html").write_text(document, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    scans = {
        market: json.loads(
            (args.data_dir / f"lifecycle-crosscheck-{market}.json").read_text(
                encoding="utf-8"
            )
        )
        for market in ("cn", "us")
    }
    report = build_report(scans)
    write_report(report, args.data_dir)
    print(
        json.dumps({key: report[key] for key in ("scope", "eligible", "selected")}),
        flush=True,
    )


if __name__ == "__main__":
    main()

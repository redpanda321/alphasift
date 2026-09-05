import pandas as pd

from alphasift.lifecycle_daily_report import build_report, score_financial


def finance(stock):
    scale = 100
    return {
        **stock,
        "status": "ok",
        "name": stock["code"],
        "currency": "USD",
        "sector": "Technology",
        "industry": "Software",
        "financial_company": False,
        "period": "2025-12-31",
        "balance_period": "2026-06-30",
        "free_cash_flow": 20 * scale,
        "operating_cash_flow": 25 * scale,
        "net_income": 10 * scale,
        "revenue": 100 * scale,
        "total_debt": 10 * scale,
        "cash": 30 * scale,
        "net_debt": -20 * scale,
        "equity": 50 * scale,
        "profit_yoy_pct": 10,
        "net_margin_pct": 10,
        "stock_based_compensation": 1,
        "retrieved_at": "2026-09-05T00:00:00+00:00",
        "source": "test",
        "annual_history": [
            {
                "period": f"{year}-12-31",
                "free_cash_flow": 20 * scale,
                "net_income": 10 * scale,
            }
            for year in (2025, 2024, 2023)
        ],
    }


def scan(market, count=6, stage="A", start=0):
    rows = []
    for i in range(start, start + count):
        symbol = f"T{i}" if market == "us" else f"60000{i}.SS"
        rows.append(
            {
                "symbol": symbol,
                "status": "AGREEMENT",
                "consensus_stage": stage,
                "consensus_score": 60 + i,
                "as_of": "2026-09-04",
                "five_year": {"price": 10, "scores": {stage: 70 + i}},
                "full_history": {
                    "scores": {stage: 65 + i},
                    "history_start": "2000-01-01",
                },
            }
        )
    return {
        "attempted": count,
        "snapshot_count": count,
        "as_of": "2026-09-04",
        "rows": rows,
        "strategy": {"id": "x"},
    }


def test_report_takes_five_per_market_industry():
    report = build_report(
        {"cn": scan("cn"), "us": scan("us")},
        fetcher=finance,
        now=pd.Timestamp("2026-09-05", tz="UTC"),
    )
    assert report["scope"] == 12
    assert report["selected"] == 10
    assert report["stage_counts"] == {"cn": {"A": 6}, "us": {"A": 6}}
    assert all(row["industry"] == "Software" for row in report["top5"])


def test_report_takes_five_per_market_industry_and_stage():
    mixed = scan("us")
    mixed["rows"] += scan("us", stage="H", start=6)["rows"]
    mixed["attempted"] = mixed["snapshot_count"] = 12
    report = build_report(
        {"us": mixed}, fetcher=finance, now=pd.Timestamp("2026-09-05", tz="UTC")
    )
    assert report["selected"] == 10
    assert {row["stage"] for row in report["top5"]} == {"A", "H"}


def test_financial_company_not_scored_like_industrial_company():
    row = finance({"score": 80, "code": "BANK"})
    row["financial_company"] = True
    result, reason = score_financial(row, now=pd.Timestamp("2026-09-05", tz="UTC"))
    assert result is None and "capital" in reason

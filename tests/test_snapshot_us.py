from types import SimpleNamespace

import pandas as pd

from alphasift import snapshot_us


def _quote(symbol: str, market_cap: int) -> dict:
    return {
        "symbol": symbol,
        "shortName": f"{symbol} Corp",
        "regularMarketPrice": 10.0,
        "regularMarketChangePercent": 1.5,
        "regularMarketVolume": 1_000_000,
        "averageDailyVolume3Month": 800_000,
        "marketCap": market_cap,
        "sharesOutstanding": 100_000_000,
        "trailingPE": 12.0,
        "priceToBook": 2.0,
    }


def test_auto_snapshot_paginates_the_full_us_equity_screen(monkeypatch):
    calls = []
    pages = {
        0: {"total": 3, "quotes": [_quote("AAA", 3_000_000_000), _quote("BBB", 4_000_000_000)]},
        2: {"total": 3, "quotes": [_quote("ZZZZ", 5_000_000_000)]},
    }

    class FakeEquityQuery:
        def __init__(self, operator, operand):
            self.operator = operator
            self.operand = operand

    def fake_screen(query, *, offset, size, sortField, sortAsc):
        calls.append((offset, size, sortField, sortAsc))
        return pages[offset]

    fake_yfinance = SimpleNamespace(EquityQuery=FakeEquityQuery, screen=fake_screen)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yfinance)
    monkeypatch.setattr(snapshot_us, "_US_SCREEN_PAGE_SIZE", 2, raising=False)
    monkeypatch.setattr(snapshot_us, "_fetch_sp500_tickers", lambda: ["ONLY"])

    result = snapshot_us.fetch_us_snapshot()

    assert result["code"].tolist() == ["AAA", "BBB", "ZZZZ"]
    assert len(result) == 3
    assert calls == [(0, 2, "ticker", True), (2, 2, "ticker", True)]
    assert result.attrs["snapshot_source"] == "yahoo_screen"


def test_screen_quote_mapping_keeps_filter_columns():
    result = snapshot_us._screen_quotes_to_snapshot([_quote("TEST", 7_000_000_000)])

    row = result.iloc[0]
    assert row["code"] == "TEST"
    assert row["total_mv"] == 7_000_000_000
    assert row["amount"] == 10_000_000
    assert row["volume_ratio"] == 1.25
    assert row["turnover_rate"] == 1.0
    assert pd.notna(row["pe_ratio"])

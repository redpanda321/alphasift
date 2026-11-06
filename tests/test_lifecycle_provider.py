from types import SimpleNamespace

import pandas as pd
import pytest

from alphasift import snapshot_us
from alphasift.daily import _normalize_daily_history


@pytest.mark.parametrize(
    "symbol,zone,hour,last_day",
    [
        ("AAPL", "America/New_York", 17, "2026-09-04"),
        ("AAPL", "America/New_York", 12, "2026-09-03"),
        ("600519.SS", "Asia/Shanghai", 16, "2026-09-04"),
        ("600519.SS", "Asia/Shanghai", 12, "2026-09-03"),
    ],
)
def test_completed_session_and_exclusive_end(monkeypatch, symbol, zone, hour, last_day):
    instant = pd.Timestamp(f"2026-09-04 {hour}:00", tz=zone)
    calls = []

    class Clock:
        @staticmethod
        def now(tz=None):
            return instant.tz_convert(tz)

    class PandasProxy:
        Timestamp = Clock

        def __getattr__(self, name):
            return getattr(pd, name)

    def download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return pd.DataFrame(
            {
                "Open": [10, 11],
                "High": [11, 12],
                "Low": [9, 10],
                "Close": [10, 11],
                "Volume": [100, 200],
            },
            index=pd.to_datetime(["2026-09-03", "2026-09-04"]),
        )

    monkeypatch.setattr(snapshot_us, "pd", PandasProxy())
    monkeypatch.setitem(
        __import__("sys").modules, "yfinance", SimpleNamespace(download=download)
    )
    result = snapshot_us.fetch_daily_history_yfinance(symbol)
    normalized = _normalize_daily_history(result)
    assert pd.Timestamp(normalized["date"].iloc[-1]).date().isoformat() == last_day
    assert calls[0][1]["end"] == "2026-09-05"
    assert calls[0][1]["auto_adjust"] is True
    assert "data_timestamp" not in result.attrs


def test_unlimited_download_uses_max_period_without_tail(monkeypatch):
    calls = []

    def download(symbol, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {"Close": 10.0}, index=pd.bdate_range("2000-01-01", periods=3000)
        )

    monkeypatch.setitem(
        __import__("sys").modules, "yfinance", SimpleNamespace(download=download)
    )
    result = snapshot_us.fetch_daily_history_yfinance("AAPL", lookback_days=0)
    assert len(result) == 3000
    assert calls[0]["period"] == "max"
    assert "start" not in calls[0]

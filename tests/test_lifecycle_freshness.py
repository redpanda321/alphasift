import pandas as pd
import pytest

from alphasift.config import Config
from alphasift.freshness import latest_completed_session
from alphasift.lifecycle import compute_lifecycle_features, enrich_lifecycle_features
from alphasift.pipeline import screen


@pytest.mark.parametrize(
    "market,now,expected",
    [
        ("cn", "2026-09-04T06:00:00Z", "2026-09-03"),
        ("cn", "2026-09-04T08:00:00Z", "2026-09-04"),
        ("us", "2026-09-04T19:00:00Z", "2026-09-03"),
        ("us", "2026-09-04T21:00:00Z", "2026-09-04"),
        ("us", "2026-09-07T21:00:00Z", "2026-09-04"),
        ("cn", "2026-10-01T08:00:00Z", "2026-09-30"),
        ("us", "2026-11-27T18:01:00Z", "2026-11-27"),
    ],
)
def test_exchange_session_holidays_and_early_close(market, now, expected):
    assert latest_completed_session(market, now) == expected


def history():
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end="2026-09-04", periods=3000),
            "close": 10.0,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "volume": 1000.0,
        }
    )


def test_all_years_are_retained_and_window_is_explicit():
    hist = history()
    assert compute_lifecycle_features(hist)["history_sessions"] == 3000
    assert (
        compute_lifecycle_features(hist, profile={"lookback_days": 1260})[
            "history_sessions"
        ]
        == 1260
    )


def test_every_scan_refetches_and_rejects_old_bars(tmp_path):
    calls = []
    hist = history()

    def fetcher(code, **kwargs):
        calls.append(kwargs)
        return hist.copy()

    rows = pd.DataFrame([{"code": "AAPL", "price": 10.0, "total_mv": 2e9}])
    for _ in range(2):
        result = enrich_lifecycle_features(
            rows,
            market="us",
            fetcher=fetcher,
            cache_dir=tmp_path,
            expected_session="2026-09-04",
        )
        assert result.attrs["lifecycle_success_count"] == 1
    assert len(calls) == 2
    assert all(c["cache_dir"] is None and c["lookback_days"] == 0 for c in calls)
    stale = enrich_lifecycle_features(
        rows, market="us", fetcher=fetcher, expected_session="2026-09-07"
    )
    assert stale.empty
    assert "outdated history" in stale.attrs["lifecycle_errors"][0]


@pytest.mark.parametrize("market", ["cn", "us"])
def test_pipeline_disables_snapshot_fallback_on_every_run(
    monkeypatch, tmp_path, market
):
    calls = []

    def snapshot(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame([{"code": "A", "name": "A", "price": 10.0, "amount": 0.0}])

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", snapshot)
    config = Config(fallback_snapshot_path=tmp_path / "old.json")
    for _ in range(2):
        result = screen("lifecycle_ah", market=market, config=config, use_llm=False)
        assert result.snapshot_retrieved_at
        assert result.expected_daily_session == latest_completed_session(
            market, result.scan_started_at
        )
        assert result.freshness_policy == "network_refresh_completed_daily"
    assert len(calls) == 2
    assert all(c["fallback_snapshot_path"] is None for c in calls)

import numpy as np
import pandas as pd
import pytest

from alphasift.lifecycle import compute_lifecycle_features, enrich_lifecycle_features
from alphasift.strategy import load_strategy


def _history_from_controls(controls, *, periods=800):
    x = np.arange(periods)
    points_x = np.array([item[0] for item in controls])
    points_y = np.array([item[1] for item in controls], dtype=float)
    close = np.interp(x, points_x, points_y)
    volume = np.full(periods, 1_000_000.0)
    volume[np.argmin(close)] = 2_200_000
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=periods),
            "open": close * 1.002,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": volume,
        }
    )


def test_a_model_does_not_require_prior_boom():
    hist = _history_from_controls(
        [
            (0, 10),
            (580, 9.5),
            (680, 9.2),
            (745, 6.0),
            (770, 5.8),
            (799, 6.35),
        ]
    )
    features = compute_lifecycle_features(hist)

    assert features["a_score"] > features["h_score"]
    assert features["h_score"] <= 49
    assert features["distance_52w_low_pct"] < 12  # intraday low, not closing low
    assert features["a_eligible"]
    assert not features["h_eligible"]


def test_h_model_requires_boom_and_repeated_lower_highs():
    hist = _history_from_controls(
        [
            (0, 8),
            (180, 7),
            (360, 12),
            (480, 30),
            (530, 17),
            (575, 24),
            (625, 12),
            (670, 19),
            (715, 8.5),
            (755, 13),
            (790, 5.0),
            (799, 5.15),
        ]
    )
    features = compute_lifecycle_features(hist)

    assert features["prior_runup_pct"] >= 100
    assert features["drawdown_from_cycle_high_pct"] >= 75
    assert features["lower_high_count"] >= 2
    assert features["h_score"] > features["a_score"]


def test_enrichment_preserves_separate_a_and_h_scores():
    a_hist = _history_from_controls([(0, 10), (650, 9), (760, 5.8), (799, 6.2)])
    h_hist = _history_from_controls(
        [
            (0, 8),
            (350, 12),
            (470, 30),
            (530, 16),
            (575, 24),
            (625, 11),
            (670, 18),
            (720, 8),
            (760, 12),
            (799, 5),
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "code": "A",
                "name": "A Corp",
                "price": 6.2,
                "amount": 1e8,
                "total_mv": 2e9,
            },
            {"code": "H", "name": "H Corp", "price": 5, "amount": 1e8, "total_mv": 2e9},
        ]
    )

    result = enrich_lifecycle_features(
        candidates,
        market="us",
        profile={"min_score": 0, "top_per_stage": 20},
        fetcher=lambda code, **kwargs: a_hist if code == "A" else h_hist,
    )

    assert {"a_score", "h_score", "lifecycle_stage", "data_as_of"} <= set(
        result.columns
    )
    assert len(result) == 2


def test_lifecycle_strategy_yaml_loads():
    strategy = load_strategy(
        __import__("pathlib").Path(__file__).parents[1]
        / "strategies"
        / "lifecycle_ah.yaml"
    )
    assert strategy.screening.market_scope == ["cn", "us"]
    assert strategy.screening.factor_weights == {"lifecycle": 1.0}
    assert strategy.screening.lifecycle_profile["mode"] == "both"
    assert strategy.screening.lifecycle_profile["monthly_min_bars"] == 24


def test_slow_cycle_prefers_all_history_monthly_then_weekly_fallback():
    long = _history_from_controls([(0, 10), (799, 6.2)])
    assert compute_lifecycle_features(long)["slow_cycle_timeframe"] == "all_available_history_monthly"

    weekly_only = long.tail(260).copy()
    features = compute_lifecycle_features(weekly_only, profile={"min_history_days": 252})
    assert features["slow_cycle_timeframe"] == "weekly_fallback"
    assert features["slow_cycle_bars"] >= 52


def test_new_low_without_boom_cannot_be_h_even_with_zero_score_threshold():
    hist = _history_from_controls([(0, 30), (500, 15), (750, 8), (799, 5)])
    rows = pd.DataFrame([{"code": "X", "total_mv": 2e9}])
    result = enrich_lifecycle_features(
        rows,
        market="us",
        profile={"mode": "h", "min_score": 0},
        fetcher=lambda *a, **kw: hist,
    )
    assert result.empty
    assert result.attrs["lifecycle_success_count"] == 1


def test_completed_h_is_not_also_a():
    hist = _history_from_controls(
        [
            (0, 8),
            (350, 12),
            (470, 30),
            (530, 16),
            (575, 24),
            (625, 11),
            (670, 18),
            (720, 8),
            (760, 12),
            (799, 5),
        ]
    )
    f = compute_lifecycle_features(hist)
    assert f["h_eligible"] and not f["a_eligible"]
    assert f["a_stage"] == "非A结构"


def test_rsi_handles_uninterrupted_rise_fall_and_flat():
    from alphasift.lifecycle import _rsi_series

    assert _rsi_series(pd.Series(np.arange(1.0, 101.0))).iloc[-1] == 100
    assert _rsi_series(pd.Series(np.arange(100.0, 0.0, -1))).iloc[-1] == 0
    assert _rsi_series(pd.Series(np.ones(100))).iloc[-1] == 50


def test_lower_highs_must_be_consecutive_at_end():
    from alphasift.lifecycle import _descending_count

    assert _descending_count([30, 20, 25, 15, 18]) == 0
    assert _descending_count([30, 20, 25, 18, 10]) == 2


def test_undated_or_duplicate_bars_fail_instead_of_fabricating_dates():
    hist = _history_from_controls([(0, 10), (799, 5)])
    with pytest.raises(ValueError, match="dated history"):
        compute_lifecycle_features(hist.drop(columns="date"))
    hist.loc[799, "date"] = hist.loc[798, "date"]
    with pytest.raises(ValueError, match="duplicate"):
        compute_lifecycle_features(hist)


def test_stale_and_fatal_risk_never_rank_and_unknown_risk_never_means_buy():
    hist = _history_from_controls([(0, 10), (650, 9), (760, 5.8), (799, 6.2)])
    rows = pd.DataFrame([{"code": "X", "total_mv": 2e9, "fatal_risk": True}])
    options = {
        "market": "us",
        "profile": {"min_score": 0},
        "fetcher": lambda *a, **kw: hist,
    }
    assert enrich_lifecycle_features(rows, **options).empty
    rows["fatal_risk"] = False
    hist.attrs["daily_stale"] = True
    assert enrich_lifecycle_features(rows, **options).empty
    hist.attrs.clear()
    result = enrich_lifecycle_features(rows, **options)
    assert len(result) == 1
    assert result.iloc[0]["data_timestamp"] == ""
    assert result.iloc[0]["fundamental_risk_status"] == "unverified"
    assert result.iloc[0]["lifecycle_action"].startswith("①")


def test_price_metrics_use_intraday_lows_and_post_d_low():
    hist = _history_from_controls(
        [
            (0, 1),
            (470, 30),
            (530, 16),
            (575, 24),
            (625, 11),
            (670, 18),
            (720, 8),
            (760, 12),
            (799, 5),
        ]
    )
    f = compute_lifecycle_features(hist)
    assert f["low_52w"] == round(hist["low"].tail(252).min(), 4)
    assert f["distance_post_d_low_pct"] < f["distance_cycle_low_pct"]
    assert np.isfinite(f["a_score"]) and 0 <= f["a_score"] <= 100


def test_future_extension_does_not_leak_into_historical_cutoff():
    hist = _history_from_controls([(0, 10), (650, 9), (760, 5.8), (799, 6.2)])
    extended = pd.concat(
        [
            hist,
            _history_from_controls([(0, 100), (799, 200)]).assign(
                date=pd.bdate_range(
                    hist.date.iloc[-1] + pd.Timedelta(days=1), periods=800
                )
            ),
        ]
    )
    cutoff = extended[extended.date <= hist.date.iloc[-1]]
    assert compute_lifecycle_features(cutoff) == compute_lifecycle_features(hist)

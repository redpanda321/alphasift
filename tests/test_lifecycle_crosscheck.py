import numpy as np
import pandas as pd
import pytest

from alphasift.lifecycle_crosscheck import crosscheck, classify_window


def history(n=2000):
    c = np.linspace(10, 50, n)
    return pd.DataFrame({"date": pd.bdate_range(end="2026-09-04", periods=n),
                         "open": c, "high": c*1.01, "low": c*.99, "close": c, "volume": 1000})


def test_full_and_five_are_separate_calendar_windows():
    result = crosscheck(history(), as_of="2026-09-04")
    assert result["full_history"]["history_sessions"] == 2000
    assert result["five_year"]["history_start"] == "2021-09-06"
    assert result["five_year"]["history_sessions"] < 2000
    assert result["five_year_span_available"]
    assert result["full_has_older_data"]
    assert result["window_years"] == 5
    assert result['strategy']['id'] == 'lifecycle_5y_full_wm'
    assert result['strategy']['require_weekly_monthly_agreement'] is True
    assert 'monthly' in result['five_year']['evidence']
    stage = result['consensus_stage']
    assert result['consensus_score'] == min(result['five_year']['scores'][stage],
                                            result['full_history']['scores'][stage])


def test_listing_under_five_years_does_not_claim_short_window_agreement():
    # A one-year slice is not a valid substitute for a five-year slow cycle.
    # The individual classification may still expose its slow-cycle timeframe,
    # but it cannot claim two-window agreement.
    result = crosscheck(history(700), as_of="2026-09-04")
    assert result["window_years"] is None
    assert not result["five_year_span_available"]
    assert result["status"] == "INSUFFICIENT_DISTINCT_HISTORY"
    assert result["five_year"] is not None
    assert result["full_history"] is not None


def test_listing_under_one_year_cannot_claim_cross_validation():
    result = crosscheck(history(200), as_of="2026-09-04")
    assert result["status"] == "INSUFFICIENT_DISTINCT_HISTORY"
    assert result["consensus_stage"] is None


def test_old_top_changes_full_history_stage():
    df = history()
    df.loc[0, "high"] = 300
    result = crosscheck(df, as_of="2026-09-04")
    assert result["five_year"]["stage"] == "D"
    assert result["full_history"]["stage"] != "D"
    assert result["status"] == "CONFLICT"


def test_future_bars_are_excluded_and_stale_rejected():
    df = history()
    result = crosscheck(df, as_of="2026-09-03")
    assert result["full_history"]["history_sessions"] == 1999
    with pytest.raises(ValueError, match="latest daily bar"):
        crosscheck(df, as_of="2026-09-07")


def test_monotonic_rally_is_d_candidate_not_b_or_e():
    result = classify_window(history())
    assert result["matches"] == ["D"]


@pytest.mark.parametrize("stage,controls", [
    ("B", [(0,10),(600,9),(650,6),(720,10),(860,18),(899,15)]),
    ("E", [(0,10),(500,20),(700,50),(800,25),(850,28),(899,33)]),
    ("F", [(0,10),(500,9),(600,60),(650,30),(700,45),(750,20),(899,23)]),
    ("G", [(0,10),(500,9),(600,60),(650,30),(700,45),(750,20),(800,35),(850,12),(899,15)]),
])
def test_first_pullback_and_first_bear_rebound(stage, controls):
    x, y = zip(*controls)
    close = np.interp(np.arange(900), x, y)
    df = pd.DataFrame({"date": pd.bdate_range(end="2026-09-04", periods=900), "close": close})
    assert classify_window(df)["stage"] == stage


def test_second_higher_low_and_second_rally_high_is_c_not_b():
    # A second confirmed rally high above the first, with a higher-low
    # pullback in between, should read as C (one leg further up than B) and
    # must not also satisfy B's single-rally-high gate.
    controls = [(0,10),(600,9),(650,6),(720,10),(860,18),(900,12),(1000,24),(1050,20)]
    x, y = zip(*controls)
    close = np.interp(np.arange(1051), x, y)
    df = pd.DataFrame({"date": pd.bdate_range(end="2026-09-04", periods=1051), "close": close})
    result = classify_window(df)
    assert result["stage"] == "C"
    assert "B" not in result["matches"]


def test_e_f_g_h_are_mutually_exclusive_by_confirmed_leg_count():
    # Same boom-and-bust shape at increasing numbers of confirmed post-D
    # lower-high/lower-low legs; each additional leg should move the
    # classification one stage further down the cycle without ambiguity.
    legs = {
        "E": [(0,10),(500,20),(700,50),(800,25),(850,28),(899,33)],
        "F": [(0,10),(500,9),(600,60),(650,30),(700,45),(750,20),(899,23)],
        "G": [(0,10),(500,9),(600,60),(650,30),(700,45),(750,20),(800,35),(850,12),(899,15)],
    }
    for stage, controls in legs.items():
        x, y = zip(*controls)
        close = np.interp(np.arange(900), x, y)
        df = pd.DataFrame({"date": pd.bdate_range(end="2026-09-04", periods=900), "close": close})
        result = classify_window(df)
        assert result["matches"] == [stage]

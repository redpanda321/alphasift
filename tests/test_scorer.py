import pandas as pd

from alphasift.models import ScreeningConfig
from alphasift.scorer import compute_screen_scores


def test_value_factor_favors_lower_positive_pe_and_pb():
    df = pd.DataFrame(
        [
            {
                "code": "low",
                "pe_ratio": 5.0,
                "pb_ratio": 0.6,
                "amount": 100_000_000,
                "turnover_rate": 3.0,
                "volume_ratio": 1.2,
                "change_pct": 0.0,
            },
            {
                "code": "high",
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
                "amount": 100_000_000,
                "turnover_rate": 3.0,
                "volume_ratio": 1.2,
                "change_pct": 0.0,
            },
        ]
    )

    scored = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"value": 1.0}),
    ).set_index("code")

    assert scored.loc["low", "screen_score"] > scored.loc["high", "screen_score"]
    assert scored.loc["low", "factor_value_score"] > scored.loc["high", "factor_value_score"]


def test_dynamic_value_weights_narrow_static_deep_value_advantage():
    df = pd.DataFrame([
        {
            "code": "static_deep_value",
            "pe_ratio": 5.0,
            "pb_ratio": 0.5,
            "amount": 80_000_000,
            "turnover_rate": 0.2,
            "volume_ratio": 0.4,
            "change_pct": -4.0,
        },
        {
            "code": "active_value",
            "pe_ratio": 7.0,
            "pb_ratio": 0.7,
            "amount": 200_000_000,
            "turnover_rate": 2.0,
            "volume_ratio": 1.2,
            "change_pct": 1.0,
        },
    ])

    static_scored = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"value": 0.55, "liquidity": 0.15, "stability": 0.20, "size": 0.10}),
    ).set_index("code")
    dynamic_scored = compute_screen_scores(
        df,
        ScreeningConfig(
            factor_weights={
                "value": 0.34,
                "stability": 0.20,
                "liquidity": 0.14,
                "momentum": 0.10,
                "activity": 0.10,
                "reversal": 0.06,
                "size": 0.06,
            },
            scoring_profile={
                "momentum_chase_start_pct": 2.5,
                "momentum_chase_penalty_slope": 18.0,
                "activity_ideal_volume_ratio": 1.2,
                "activity_ideal_turnover_rate": 2.0,
            },
        ),
    ).set_index("code")

    static_gap = float(static_scored.loc["static_deep_value", "screen_score"]) - float(
        static_scored.loc["active_value", "screen_score"]
    )
    dynamic_gap = float(dynamic_scored.loc["static_deep_value", "screen_score"]) - float(
        dynamic_scored.loc["active_value", "screen_score"]
    )

    assert float(dynamic_scored.loc["active_value", "factor_momentum_score"]) > float(
        dynamic_scored.loc["static_deep_value", "factor_momentum_score"]
    )
    assert float(dynamic_scored.loc["active_value", "factor_activity_score"]) > float(
        dynamic_scored.loc["static_deep_value", "factor_activity_score"]
    )
    assert dynamic_gap < static_gap


def test_scoring_profile_can_tighten_intraday_chase_penalty():
    df = pd.DataFrame([{"code": "hot", "change_pct": 6.0}])

    default_score = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"momentum": 1.0}),
    ).loc[0, "factor_momentum_score"]
    stricter_score = compute_screen_scores(
        df,
        ScreeningConfig(
            factor_weights={"momentum": 1.0},
            scoring_profile={
                "momentum_chase_start_pct": 1.0,
                "momentum_chase_penalty_slope": 40.0,
            },
        ),
    ).loc[0, "factor_momentum_score"]

    assert stricter_score < default_score


def test_theme_heat_factor_uses_board_heat_score_when_available():
    df = pd.DataFrame([
        {"code": "hot", "board_heat_score": 82},
        {"code": "cold", "board_heat_score": 35},
    ])

    scored = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"theme_heat": 1.0}),
    ).set_index("code")

    assert scored.loc["hot", "factor_theme_heat_score"] > scored.loc["cold", "factor_theme_heat_score"]
    assert scored.loc["hot", "screen_score"] > scored.loc["cold", "screen_score"]


def test_theme_heat_factor_uses_reliable_trend_and_cooling_signal():
    df = pd.DataFrame([
        {
            "code": "warming",
            "board_heat_score": 60,
            "board_heat_trend_score": 10,
            "board_heat_persistence_score": 100,
            "board_heat_cooling_score": 0,
            "board_heat_observations": 2,
        },
        {
            "code": "cooling",
            "board_heat_score": 60,
            "board_heat_trend_score": -10,
            "board_heat_persistence_score": 40,
            "board_heat_cooling_score": 8,
            "board_heat_observations": 2,
        },
        {
            "code": "thin",
            "board_heat_score": 60,
            "board_heat_trend_score": 10,
            "board_heat_persistence_score": 100,
            "board_heat_cooling_score": 0,
            "board_heat_observations": 1,
        },
    ])

    scored = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"theme_heat": 1.0}),
    ).set_index("code")

    assert scored.loc["warming", "factor_theme_heat_score"] > scored.loc["thin", "factor_theme_heat_score"]
    assert scored.loc["thin", "factor_theme_heat_score"] > scored.loc["cooling", "factor_theme_heat_score"]


def test_topic_alignment_factor_matches_candidate_topics_to_hotspot_summary():
    df = pd.DataFrame([
        {
            "code": "aligned",
            "industry": "银行",
            "concepts": "低估值,中特估",
            "board_heat_summary": "银行:+1.20%:rank=3|地产:+0.20%:rank=8",
            "board_heat_score": 82,
        },
        {
            "code": "mismatch",
            "industry": "医药",
            "concepts": "创新药",
            "board_heat_summary": "银行:+1.20%:rank=3|地产:+0.20%:rank=8",
            "board_heat_score": 82,
        },
        {
            "code": "unknown",
            "industry": "",
            "concepts": "",
            "board_heat_summary": "银行:+1.20%:rank=3",
            "board_heat_score": 82,
        },
    ])

    scored = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"topic_alignment": 1.0}),
    ).set_index("code")

    assert scored.loc["aligned", "factor_topic_alignment_score"] > scored.loc["unknown", "factor_topic_alignment_score"]
    assert scored.loc["unknown", "factor_topic_alignment_score"] > scored.loc["mismatch", "factor_topic_alignment_score"]
    assert scored.loc["aligned", "screen_score"] > scored.loc["mismatch", "screen_score"]


def test_stability_factor_penalizes_high_daily_volatility_and_deep_drawdown():
    df = pd.DataFrame([
        {"code": "controlled", "change_pct": 0.5, "volatility_20d_pct": 22.0, "max_drawdown_20d_pct": -4.0, "atr_20_pct": 2.5, "daily_quality_score": 100, "daily_quality_flags": ""},
        {"code": "wild", "change_pct": 0.5, "volatility_20d_pct": 75.0, "max_drawdown_20d_pct": -24.0, "atr_20_pct": 9.0, "daily_quality_score": 60, "daily_quality_flags": "invalid_ohlc"},
    ])

    scored = compute_screen_scores(
        df,
        ScreeningConfig(factor_weights={"stability": 1.0}),
    ).set_index("code")

    assert scored.loc["controlled", "factor_stability_score"] > scored.loc["wild", "factor_stability_score"]
    assert scored.loc["controlled", "screen_score"] > scored.loc["wild", "screen_score"]

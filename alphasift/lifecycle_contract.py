"""Authoritative contract for the scheduled five-year/full-history strategy."""

STRATEGY_ID = 'lifecycle_5y_full_wm'
STRATEGY_VERSION = '2.2.0'
WINDOW_YEARS = 5
# Preference order for the "short" window compared against full history: try a
# full trailing 5-calendar-year span first, and fall back to a 1-calendar-year
# span for stocks whose full history is shorter than 5 years (e.g. recent
# IPOs) so they still get a two-window cross-check instead of being excluded.
FALLBACK_WINDOW_YEARS = (5, 1)
STAGES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')


def strategy_contract():
    return dict(id=STRATEGY_ID, version=STRATEGY_VERSION,
                windows=['trailing_5_calendar_years_or_1_calendar_year_fallback',
                         'all_provider_available_history'],
                window_fallback_years=list(FALLBACK_WINDOW_YEARS),
                stages=list(STAGES), decision='independent_window_classification_then_agreement',
                consensus_score='minimum_of_two_window_stage_scores',
                require_weekly_monthly_agreement=True,
                auxiliary_features='daily indicators plus confirmed weekly pivots and monthly regime confirmation',
                missing_history='exclude_from_agreement', conflict='retain_separately',
                full_history_is_verified_since_ipo=False)

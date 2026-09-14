"""Authoritative contract for the scheduled five-year/full-history strategy."""

STRATEGY_ID = 'lifecycle_5y_full_wm'
STRATEGY_VERSION = '2.3.0'
WINDOW_YEARS = 5
# Preference order for the "short" window compared against full history: try a
# A/H slow-cycle decisions use full available monthly history first, then a
# distinct trailing five-calendar-year comparison.  Shorter one-year windows
# are deliberately not treated as a cycle substitute; callers can use the
# weekly fallback recorded by the classifier instead.
FALLBACK_WINDOW_YEARS = (5,)
STAGES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')


def strategy_contract():
    return dict(id=STRATEGY_ID, version=STRATEGY_VERSION,
                windows=['all_provider_available_history_monthly',
                         'trailing_5_calendar_years_monthly',
                         'weekly_fallback_when_monthly_unavailable'],
                window_fallback_years=list(FALLBACK_WINDOW_YEARS),
                stages=list(STAGES), decision='independent_window_classification_then_agreement',
                consensus_score='minimum_of_two_window_stage_scores',
                require_weekly_monthly_agreement=True,
                auxiliary_features='daily indicators plus confirmed weekly pivots; all-history monthly, then 5-year monthly, then weekly slow-cycle regime',
                missing_history='exclude_from_agreement', conflict='retain_separately',
                full_history_is_verified_since_ipo=False)

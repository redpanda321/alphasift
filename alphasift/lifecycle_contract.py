"""Authoritative contract for the scheduled five-year/full-history strategy."""

STRATEGY_ID = 'lifecycle_5y_full'
STRATEGY_VERSION = '1.0.0'
WINDOW_YEARS = 5
STAGES = ('A', 'B', 'D', 'E', 'H')


def strategy_contract():
    return dict(id=STRATEGY_ID, version=STRATEGY_VERSION,
                windows=['trailing_5_calendar_years', 'all_provider_available_history'],
                stages=list(STAGES), decision='independent_window_classification_then_agreement',
                consensus_score='minimum_of_two_window_stage_scores',
                require_weekly_monthly_agreement=False,
                auxiliary_features='daily indicators and confirmed weekly structural pivots',
                missing_history='exclude_from_agreement', conflict='retain_separately',
                full_history_is_verified_since_ipo=False)

from pathlib import Path

import pandas as pd

from alphasift.lifecycle_fundamentals import legacy_functions, score_candidate


def ranker():
    return legacy_functions(Path('data/rank_fundamental_top50.py'), ('clip', 'rank'),
                            dict(pd=pd, asof=pd.Timestamp('2026-09-05'), translation={}))['rank']


def fixture(scale=1):
    f = dict(status='ok', financial_company=False, currency='USD', period='2025-12-31',
             balance_period='2026-06-30', profit_yoy_pct=10,
             annual_history=[dict(period=f'{year}-12-31', free_cash_flow=20*scale, net_income=10*scale)
                             for year in (2025, 2024, 2023)])
    f.update({k: v*scale for k,v in dict(free_cash_flow=20, operating_cash_flow=25, net_income=10,
              revenue=100, total_debt=10, cash=30, equity=50).items()})
    return f


def test_currency_scale_invariance_and_weights():
    one = score_candidate(dict(score='60', stage='D'), fixture(), ranker())
    two = score_candidate(dict(score='60', stage='D'), fixture(10000), ranker())
    assert one['combined_score'] == two['combined_score']
    assert one['combined_score'] == round(one['fundamental_score']*.8+12, 2)
    assert one['stage'] == 'D'


def test_missing_cashflow_is_not_zero_or_eligible():
    f = fixture(); f['free_cash_flow'] = None
    assert not score_candidate(dict(score=100), f, ranker())['eligible']


def test_financial_company_needs_specialized_review():
    f = fixture(); f['financial_company'] = True
    r = score_candidate(dict(score=100), f, ranker())
    assert not r['eligible'] and '资本充足率' in r['exclusion']


def test_stale_and_negative_cashflow_excluded():
    for key, value in [('balance_period', '2024-12-31'), ('free_cash_flow', -1)]:
        f = fixture(); f[key] = value
        assert not score_candidate(dict(score=100), f, ranker())['eligible']


def test_legacy_loader_does_not_execute_top_level(tmp_path):
    path = tmp_path/'legacy.py'
    path.write_text('raise RuntimeError("must not run")\ndef clip(x):\n    return x\n')
    assert legacy_functions(path, ('clip',), {})['clip'](2) == 2

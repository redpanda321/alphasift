import pytest

from alphasift.lifecycle_daily import validate_scan


def test_empty_scan_rejected():
    with pytest.raises(ValueError):
        validate_scan(dict(rows=[], attempted=0))


def test_provider_outage_rejected():
    with pytest.raises(ValueError):
        validate_scan(dict(rows=[dict(status='AGREEMENT'), dict(status='FAILED')], attempted=2))


def test_success_records_actual_session_and_coverage():
    r = validate_scan(dict(rows=[dict(status='AGREEMENT')], attempted=1, as_of='2026-09-04'))
    assert r == dict(attempted=1, failed=0, as_of='2026-09-04')

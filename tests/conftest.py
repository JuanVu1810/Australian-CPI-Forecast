import pytest

from api import main as api_main


def _clear_api_caches():
    api_main._rba_action_response.cache_clear()
    api_main._scenario_response.cache_clear()
    api_main._load_interval_calibration_factors.cache_clear()
    api_main._load_interval_coverage_report.cache_clear()


@pytest.fixture(autouse=True)
def _clear_api_response_caches():
    """The slow endpoints and report loaders cache their answers, so one test's patched data must not leak into the next."""
    _clear_api_caches()
    yield
    _clear_api_caches()

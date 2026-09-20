"""Smoke tests for the Streamlit demo's hosted-mode switch.

Hosted mode (CPI_DEMO_HOSTED=1) is what stops a public visitor from pointing the server
at an arbitrary URL, so it is worth pinning. These run the real script headlessly with an
unreachable API address, so they need no running API and fail fast on connection refused.
"""

from pathlib import Path

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest


APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
UNREACHABLE_API = "http://127.0.0.1:9"


def _run_app(monkeypatch, hosted: bool) -> AppTest:
    monkeypatch.setenv("API_BASE_URL", UNREACHABLE_API)
    if hosted:
        monkeypatch.setenv("CPI_DEMO_HOSTED", "1")
    else:
        monkeypatch.delenv("CPI_DEMO_HOSTED", raising=False)
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def _click(app: AppTest, label: str) -> AppTest:
    return [button for button in app.button if button.label == label][0].click().run()


def _api_box(app: AppTest) -> list:
    return [box for box in app.text_input if box.key == "api_base_url"]


def test_local_mode_lets_the_user_edit_the_api_address(monkeypatch):
    app = _run_app(monkeypatch, hosted=False)

    assert not app.exception
    assert len(_api_box(app)) == 1
    assert _api_box(app)[0].value == UNREACHABLE_API


def test_hosted_mode_locks_the_api_address(monkeypatch):
    app = _run_app(monkeypatch, hosted=True)

    assert not app.exception
    assert _api_box(app) == []


def test_hosted_mode_errors_hide_the_service_url_and_local_instructions(monkeypatch):
    app = _click(_run_app(monkeypatch, hosted=True), "Load live forecasts")

    banners = [info.value for info in app.info]
    assert any("not responding" in banner for banner in banners)
    assert not any("127.0.0.1" in banner or "uvicorn" in banner for banner in banners)


def test_local_mode_errors_keep_the_startup_hint(monkeypatch):
    app = _click(_run_app(monkeypatch, hosted=False), "Load live forecasts")

    assert any("uvicorn" in info.value for info in app.info)


# --- caching of answers that only change on a redeploy --------------------------------
# The scenario and RBA calls each take ~45s on Cloud Run, so the demo remembers their
# answers (shared by every visitor) instead of recomputing for each click.
SCENARIO_PAYLOAD = {
    "forecast_origin": "2025Q4",
    "target": "headline",
    "shock_value": 0.0,
    "shock_size": -1.0,
    "horizons": [1, 2],
    "quarters": ["2026Q1", "2026Q2"],
    "forecast": [3.1, 3.2],
    "interval_lower": [2.0, 2.1],
    "interval_upper": [4.2, 4.3],
    "caveat": "test caveat",
}
RBA_PAYLOAD = {
    "reportable_action": "hold",
    "reportable_model": "threshold",
    "target_quarter": "2026Q1",
    "forecast_origin": "2025Q4",
    "headline_forecast": 2.6,
    "trimmed_mean_forecast": 2.7,
    "caveat": "test caveat",
    "models": [],
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def api_calls(monkeypatch):
    """Replace the network with a fake that records every (method, path) it is asked for."""
    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        path = url.replace(UNREACHABLE_API, "")
        calls.append((method, path))
        payloads = {"/forecast/scenario": SCENARIO_PAYLOAD, "/rba-action": RBA_PAYLOAD}
        return _FakeResponse(payloads.get(path, {"models": []}))

    monkeypatch.setattr(requests, "request", fake_request)
    st.cache_data.clear()
    yield calls
    st.cache_data.clear()


def _count(calls, path):
    return sum(1 for _, called_path in calls if called_path == path)


def test_repeating_the_same_scenario_is_served_from_the_cache(monkeypatch, api_calls):
    app = _run_app(monkeypatch, hosted=False)

    app = _click(app, "Run scenario")
    app = _click(app, "Run scenario")

    assert not app.exception
    assert _count(api_calls, "/forecast/scenario") == 1


def test_a_different_scenario_is_not_served_the_cached_answer(monkeypatch, api_calls):
    app = _run_app(monkeypatch, hosted=False)
    app = _click(app, "Run scenario")

    [box for box in app.number_input if box.key == "scenario_shock_value"][0].set_value(1.0).run()
    app = _click(app, "Run scenario")

    assert _count(api_calls, "/forecast/scenario") == 2


def test_repeating_the_rba_refresh_is_served_from_the_cache(monkeypatch, api_calls):
    app = _run_app(monkeypatch, hosted=False)

    app = _click(app, "Refresh RBA action")
    app = _click(app, "Refresh RBA action")

    assert not app.exception
    assert _count(api_calls, "/rba-action") == 1

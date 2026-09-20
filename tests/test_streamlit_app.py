"""Smoke tests for the Streamlit demo's hosted-mode switch.

Hosted mode (CPI_DEMO_HOSTED=1) is what stops a public visitor from pointing the server
at an arbitrary URL, so it is worth pinning. These run the real script headlessly with an
unreachable API address, so they need no running API and fail fast on connection refused.
"""

from pathlib import Path

import pytest
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

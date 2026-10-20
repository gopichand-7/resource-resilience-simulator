"""Step 23: the app shell. Pure logic first, then the real app run headlessly with Streamlit's test runner."""
import copy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import app_state
from src.app_state import (
    DISCLOSURE,
    PAGES,
    SESSION_DEFAULTS,
    clear_response,
    init_session,
    load_baseline,
    reset_scenario,
)

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


# ---------- load_baseline ----------
def test_load_baseline_on_the_real_dataset():
    b = load_baseline()
    assert b["errors"] == []
    assert b["snapshot"]["metrics"]["Unmet hours (staffing gap)"] == 288
    assert set(b["data"]) >= {"Resources", "Projects", "Allocations"}


def test_missing_file_is_reported_not_raised(tmp_path):
    b = load_baseline(tmp_path / "nope.xlsx")
    assert b["data"] is None and b["snapshot"] is None
    assert len(b["errors"]) == 1 and b["errors"][0]


def test_corrupt_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_text("this is not an excel file")
    b = load_baseline(bad)
    assert b["data"] is None and b["errors"]


def test_invalid_data_returns_every_error_and_no_snapshot(tmp_path):
    from src.data_loader import load_data
    d = load_data()
    d["Resources"].loc[0, "WeeklyCapacityHrs"] = -5
    d["Allocations"].loc[0, "ResourceID"] = "R99"
    path = tmp_path / "broken.xlsx"
    with pd.ExcelWriter(path) as xl:
        for name, frame in d.items():
            frame.to_excel(xl, sheet_name=name, index=False)
    b = load_baseline(path)
    assert b["snapshot"] is None
    assert len(b["errors"]) >= 2


# ---------- session state ----------
def test_init_session_fills_only_missing_keys():
    state = {"scenario_label": "kept"}
    init_session(state)
    assert state["scenario_label"] == "kept"
    assert set(state) == set(SESSION_DEFAULTS)


def test_reset_scenario_clears_shock_and_response():
    state = init_session({})
    state.update(scenario_data="x", scenario_label="R15 out", response_data="y", response_label="sub")
    reset_scenario(state)
    assert state == SESSION_DEFAULTS


def test_clear_response_keeps_the_shock():
    state = init_session({})
    state.update(scenario_data="x", scenario_label="R15 out", response_data="y", response_label="sub",
                 response_kind="k", response_result={"a": 1})
    clear_response(state)
    assert state["scenario_data"] == "x" and state["scenario_label"] == "R15 out"
    assert state["response_data"] is None and state["response_label"] == ""
    assert state["response_kind"] is None and state["response_result"] is None


def test_the_defaults_are_not_shared_between_sessions():
    a, b = init_session({}), init_session({})
    a["scenario_label"] = "changed"
    assert b["scenario_label"] == ""


# ---------- the framing ----------
def test_disclosure_states_what_the_tool_is_and_is_not():
    text = DISCLOSURE.lower()
    for word in ["student-built", "synthetic", "prototype"]:
        assert word in text
    for word in ["forecast", "optimiz", "machine learning", "recommend"]:
        assert word in text            # named only to say it does NOT do them
    assert "does not" in text


def test_there_are_four_pages_and_each_has_a_file():
    assert [k for k, _ in PAGES] == ["portfolio_map", "allocation_planner", "shock_simulator", "tradeoffs"]
    for key, _ in PAGES:
        assert (ROOT / "app_pages" / f"{key}.py").exists(), key


# ---------- the real app, run headlessly ----------
@pytest.fixture(autouse=True)
def fresh_cache():
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_app_starts_without_errors_and_shows_title_and_disclosure():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == app_state.APP_TITLE
    assert any(DISCLOSURE in info.value for info in at.info)


def test_app_opens_on_the_first_page():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.header[0].value == "Portfolio Map"


def test_sidebar_shows_baseline_and_a_disabled_reset_button():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("Baseline plan" in m.value for m in at.sidebar.markdown)
    reset = [b for b in at.sidebar.button if b.label == "Reset to baseline"][0]
    assert reset.disabled


def test_baseline_is_stored_for_the_pages_to_use():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.session_state["baseline"]["snapshot"]["metrics"]["Required hours"] == 5430


def test_sidebar_shows_the_shock_and_reset_clears_it():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["scenario_data"] = {"anything": 1}
    at.session_state["scenario_label"] = "R15 unavailable, weeks 4-5"
    at.run()
    assert any("R15 unavailable, weeks 4-5" in m.value for m in at.sidebar.markdown)
    reset = [b for b in at.sidebar.button if b.label == "Reset to baseline"][0]
    assert not reset.disabled
    reset.click().run()
    assert at.session_state["scenario_data"] is None
    assert at.session_state["scenario_label"] == ""
    assert any("Baseline plan" in m.value for m in at.sidebar.markdown)


def test_app_explains_a_broken_dataset_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(app_state, "load_baseline",
                        lambda path=None: {"data": None, "errors": ["Resources: capacity is negative", "x"], "snapshot": None})
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("cannot start" in e.value for e in at.error)
    assert any("capacity is negative" in m.value for m in at.markdown)
    assert at.title[0].value == app_state.APP_TITLE      # header still shown
    assert "baseline" not in at.session_state            # pages never run on bad data
    assert not at.header                                 # no page was rendered


# ---------- Step 24: errors are explained, never a red traceback ----------
def test_an_unexpected_failure_inside_a_page_is_explained(monkeypatch):
    from src import shock_view
    monkeypatch.setattr(shock_view, "run_shock", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page("app_pages/shock_simulator.py").run()
    [b for b in at.main.button if b.label == "Apply shock"][0].click().run()
    assert not at.exception
    assert any("Something went wrong" in e.value for e in at.error)
    assert any("RuntimeError: boom" in c.value for c in at.code)
    assert at.title[0].value == app_state.APP_TITLE                 # the app shell is still there
    assert any(b.label == "Reset to baseline" for b in at.sidebar.button)


def test_a_broken_dataset_screen_offers_a_reload_button(monkeypatch):
    monkeypatch.setattr(app_state, "load_baseline",
                        lambda path=None: {"data": None, "errors": ["x is wrong"], "snapshot": None})
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any(b.label == "Reload the data" for b in at.button)


def test_a_check_that_itself_fails_becomes_a_message(monkeypatch):
    monkeypatch.setattr(app_state, "validate_data", lambda data: (_ for _ in ()).throw(KeyError("Week")))
    b = load_baseline()
    assert b["snapshot"] is None and "could not be checked" in b["errors"][0]


def test_a_missing_file_message_does_not_show_a_server_path(tmp_path):
    b = load_baseline(tmp_path / "nope.xlsx")
    assert "nope.xlsx" in b["errors"][0] and str(tmp_path) not in b["errors"][0]


# ---------- Step 25: every page says what to do first and what comes next ----------
def test_every_page_has_a_start_here_line_and_it_is_shown():
    for key, _ in app_state.PAGES:
        start, _next = app_state.GUIDE[key]
        assert start.startswith("Start here:") and len(start) <= 170
        at = AppTest.from_file(str(Path(APP).parent / "app_pages" / f"{key}.py"), default_timeout=90)
        at.session_state["baseline"] = load_baseline()
        for k, v in app_state.SESSION_DEFAULTS.items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, key
        assert any(c.value == start for c in at.caption), key


def test_the_guide_points_forward_and_never_recommends():
    names = {title for _, title in app_state.PAGES}
    for key, (start, nxt) in app_state.GUIDE.items():
        for text in (start, nxt):
            assert not any(w in text.lower() for w in ["recommend", "should", "best option", "optimal"])
        if nxt:
            assert nxt.startswith("Next:") and any(n in nxt for n in names)
    assert app_state.GUIDE["portfolio_map"][1] and app_state.GUIDE["allocation_planner"][1]

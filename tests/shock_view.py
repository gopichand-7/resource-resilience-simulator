"""Shock Simulator. Expected numbers come from the hand calculations in test_scenarios.py (raw Excel), not the engine."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app_state import load_baseline
from src.data_loader import load_data
from src.shock_view import (
    SHOCK_DEADLINE,
    SHOCK_EFFORT,
    SHOCK_KINDS,
    SHOCK_RESOURCE,
    apply_shock,
    bottom_line,
    owner_questions,
    project_status_table,
    before_after_figure,
    calendar_weeks,
    conflict_change_display,
    deadline_range,
    default_resource,
    metric_tiles,
    project_options,
    resource_options,
    resource_preview,
    run_shock,
    shock_story,
)

PAGE = str(Path(__file__).resolve().parent.parent / "app_pages" / "shock_simulator.py")


@pytest.fixture(scope="module")
def baseline():
    return load_baseline()


@pytest.fixture(scope="module")
def data(baseline):
    return baseline["data"]


@pytest.fixture(scope="module")
def r15(baseline):
    return run_shock(baseline, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])


@pytest.fixture(scope="module")
def p01(baseline):
    return run_shock(baseline, SHOCK_DEADLINE, project_id="P01", new_deadline=6)


@pytest.fixture(scope="module")
def p04(baseline):
    return run_shock(baseline, SHOCK_EFFORT, project_id="P04", percent=20)


# ---------- choices ----------
def test_choice_lists(data):
    r = resource_options(data)
    assert len(r) == 28 and r.iloc[0]["Label"] == "R01 Aarav Reddy (Software Engineer)"
    assert len(project_options(data)) == 9
    assert calendar_weeks(data) == (1, 12)


def test_default_person_is_the_one_who_is_the_only_qualified_for_the_most_requirements(data):
    assert default_resource(data) == "R15"        # R15 is the only qualified person for Q001 and Q020


def test_deadline_range(data):
    assert deadline_range(data, "P01") == (1, 9)  # starts week 1, deadline week 10
    assert deadline_range(data, "P04") == (4, 9)


def test_deadline_cannot_move_earlier_when_it_equals_the_start_week(data):
    import copy
    d = copy.deepcopy(data)
    d["Projects"].loc[d["Projects"]["ProjectID"] == "P01", "DeadlineWeek"] = 1
    assert deadline_range(d, "P01") is None


def test_resource_preview_hand_calculated(data):
    projects, hours = resource_preview(data, "R15", [4, 5])
    assert projects == ["P01", "P02", "P06"] and hours == 76          # 2 weeks x (12 + 14 + 12)
    assert resource_preview(data, "R25", [12]) == ([], 0.0)


# ---------- applying ----------
def test_labels_are_plain_words(data):
    _, l1 = apply_shock(data, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])
    _, l2 = apply_shock(data, SHOCK_DEADLINE, project_id="P01", new_deadline=6)
    _, l3 = apply_shock(data, SHOCK_EFFORT, project_id="P04", percent=20)
    _, l4 = apply_shock(data, SHOCK_RESOURCE, resource_id="R15", weeks=[7])
    assert l1 == "Rohit Sharma (R15) unavailable, weeks 4-5"
    assert l2 == "Cloud Migration (P01) deadline moved from week 10 to week 6"
    assert l3 == "Security Upgrade (P04) needs 20% more work"
    assert l4 == "Rohit Sharma (R15) unavailable, week 7"


def test_unknown_shock_type_and_bad_inputs_raise_clear_errors(data):
    with pytest.raises(ValueError, match="Unknown shock type"):
        apply_shock(data, "Meteor strike")
    with pytest.raises(ValueError, match="Unknown resource"):
        apply_shock(data, SHOCK_RESOURCE, resource_id="R99", weeks=[4])
    with pytest.raises(ValueError, match="earlier"):
        apply_shock(data, SHOCK_DEADLINE, project_id="P01", new_deadline=10)
    with pytest.raises(ValueError, match="0%"):
        apply_shock(data, SHOCK_EFFORT, project_id="P04", percent=-1)


def test_run_shock_returns_everything_the_page_stores(r15):
    assert set(r15) == {"scenario_data", "scenario_label", "scenario_snapshot", "comparison"}


def test_the_baseline_is_untouched_by_running_a_shock(baseline):
    before = baseline["data"]["Allocations"].copy(deep=True)
    run_shock(baseline, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])
    assert baseline["data"]["Allocations"].equals(before)
    assert baseline["snapshot"]["metrics"]["Unmet hours (staffing gap)"] == 288


# ---------- tiles ----------
def tiles_of(result):
    return {label: (value, delta) for label, value, delta, _ in metric_tiles(result["comparison"])}


def test_tiles_for_each_shock(r15, p01, p04):
    assert tiles_of(r15)["Unmet hours"] == (364, 76)                      # 288 + 76
    assert tiles_of(p01)["Unmet hours"] == (476, 188)                     # 288 + 188
    assert tiles_of(p04)["Unmet hours"] == (324, 36)                      # 288 + 36
    assert tiles_of(p04)["Shortage hours"] == (132, 10)          # Q020 40h -> 50h
    assert tiles_of(p04)["Exposed"] == (1, 0)


# ---------- the story ----------
def test_story_for_the_resource_shock(r15):
    lines = shock_story(r15["comparison"])
    assert lines[0] == "Unmet hours go from 288h to 364h (+76h)."
    assert "Q001 (P01, Cloud Infrastructure) +24h" in lines[1] and "Q011 (P02, Cloud Infrastructure) +28h" in lines[1]
    assert "Q035 (P06, Cloud Infrastructure) +24h" in lines[1]
    assert "P01 0h to 24h" in lines[2] and "P02 0h to 28h" in lines[2] and "P06 0h to 24h" in lines[2]
    assert "Cloud Infrastructure (Q001, Q020) +28h" in lines[3]           # Q001 new 24h + Q020 40h -> 44h
    assert "P05 via P01 -> P05" in lines[4] and "P09 via P02 -> P09" in lines[4]
    assert "not a predicted delay" in lines[4]


def test_story_for_the_deadline_shock(p01):
    lines = shock_story(p01["comparison"])
    assert lines[0] == "Unmet hours go from 288h to 476h (+188h)."
    assert "P01 0h to 188h" in lines[2]
    assert "P05 via P01 -> P05" in lines[-1]


def test_story_when_nothing_changes(baseline):
    result = run_shock(baseline, SHOCK_RESOURCE, resource_id="R25", weeks=[12])   # R25 has no work in week 12
    assert shock_story(result["comparison"]) == ["This shock does not create any new unmet hours in the plan."]


def test_the_story_never_recommends_or_predicts(r15, p01, p04):
    for result in (r15, p01, p04):
        text = " ".join(shock_story(result["comparison"])).lower()
        for word in ["recommend", "should", "best", "will slip", "will be late", "risk score"]:
            assert word not in text


# ---------- chart and conflict table ----------
def test_before_after_chart(baseline, p04):
    fig = before_after_figure(baseline["snapshot"]["demand"], p04["scenario_snapshot"]["demand"])
    assert [t.name for t in fig.data] == ["Baseline", "After the shock"]
    assert list(fig.data[0].x) == ["P04", "P08"]                           # only projects with unmet hours
    assert list(fig.data[0].y) == [70, 218] and list(fig.data[1].y) == [106, 218]


def test_conflict_change_table_is_whole_hours_with_friendly_headers(p04):
    t = conflict_change_display(p04["comparison"])
    assert list(t.columns) == ["Status", "Type", "Requirement", "Project", "Before h", "After h", "Change h"]
    row = t[(t["Type"] == "Capacity shortage") & (t["Requirement"] == "Q020")].iloc[0]
    assert (row["Status"], row["Before h"], row["After h"], row["Change h"]) == ("Worsened", 40, 50, 10)


def test_conflict_change_table_is_empty_when_nothing_changed(baseline):
    result = run_shock(baseline, SHOCK_EFFORT, project_id="P04", percent=0)
    assert conflict_change_display(result["comparison"]).empty


# ---------- the real page, run headlessly ----------
@pytest.fixture
def page(baseline):
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.session_state["baseline"] = baseline
    for key, default in {"scenario_data": None, "scenario_label": "", "scenario_snapshot": None, "comparison": None,
                         "response_data": None, "response_label": ""}.items():
        at.session_state[key] = default
    return at.run()


def test_page_opens_with_the_defaults_and_no_results(page):
    assert not page.exception
    assert page.header[0].value == "Shock Simulator"
    assert page.radio[0].value == SHOCK_RESOURCE
    assert "R15" in page.selectbox[0].value
    assert page.slider[0].value == (4, 5)
    assert any("76h on P01, P02, P06" in c.value for c in page.caption)
    assert any("press Apply shock" in i.value for i in page.info)
    assert not page.metric


def test_applying_the_default_shock_shows_the_story(page):
    page.button[0].click().run()
    assert not page.exception
    assert page.session_state["scenario_label"] == "Rohit Sharma (R15) unavailable, weeks 4-5"
    tiles = {m.label: (m.value, m.delta) for m in page.metric}
    assert tiles["Unmet hours"] == ("364", "+76")
    text = " ".join(m.value for m in page.markdown)
    assert "Unmet hours go from 288h to 364h (+76h)." in text
    assert page.subheader[0].value == "What the shock did"


def test_choosing_a_deadline_shock(page):
    page.radio[0].set_value(SHOCK_DEADLINE).run()
    pick = [o for o in page.selectbox[0].options if o.startswith("P01")][0]
    page.selectbox[0].select(pick).run()
    page.slider[0].set_value(6).run()
    page.button[0].click().run()
    assert not page.exception
    assert page.session_state["scenario_label"] == "Cloud Migration (P01) deadline moved from week 10 to week 6"
    assert {m.label: m.value for m in page.metric}["Unmet hours"] == "476"


def test_choosing_an_effort_shock(page):
    page.radio[0].set_value(SHOCK_EFFORT).run()
    pick = [o for o in page.selectbox[0].options if o.startswith("P04")][0]
    page.selectbox[0].select(pick).run()
    page.button[0].click().run()                       # default 20%
    assert page.session_state["scenario_label"] == "Security Upgrade (P04) needs 20% more work"
    assert {m.label: m.value for m in page.metric}["Unmet hours"] == "324"


def test_shocks_do_not_stack(page):
    page.button[0].click().run()                       # R15 out
    page.radio[0].set_value(SHOCK_EFFORT).run()
    pick = [o for o in page.selectbox[0].options if o.startswith("P04")][0]
    page.selectbox[0].select(pick).run()
    page.button[0].click().run()
    assert {m.label: m.value for m in page.metric}["Unmet hours"] == "324"      # 288 + 36, not 288 + 76 + 36


def test_applying_a_shock_clears_any_earlier_response(page):
    page.session_state["response_data"] = {"x": 1}
    page.session_state["response_label"] = "something"
    page.button[0].click().run()
    assert page.session_state["response_data"] is None and page.session_state["response_label"] == ""


def test_a_person_with_no_work_in_those_weeks_is_told_so(page):
    person = [o for o in page.selectbox[0].options if o.startswith("R25")][0]
    page.selectbox[0].select(person).run()
    page.slider[0].set_value((12, 12)).run()
    assert any("no planned hours in those weeks" in c.value for c in page.caption)


def test_the_page_keeps_tile_values_short(page):
    page.button[0].click().run()
    for m in page.metric:
        assert len(m.value) <= 16
        assert len(m.label) <= 14, m.label           # longer labels are cut off with "..." on a laptop screen


def test_the_page_explains_how_shocks_are_modelled(page):
    page.button[0].click().run()
    assert any("How is each shock modelled?" in e.label for e in page.expander)


# ---------- through the whole app (sidebar + navigation) ----------
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app():
    import streamlit as st
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=90).run()
    return at.switch_page("app_pages/shock_simulator.py").run()


def test_the_sidebar_shows_the_shock_and_reset_clears_it(app):
    [b for b in app.main.button if b.label == "Apply shock"][0].click().run()
    assert any("Shock: Rohit Sharma (R15) unavailable, weeks 4-5" in m.value for m in app.sidebar.markdown)
    reset = [b for b in app.sidebar.button if b.label == "Reset to baseline"][0]
    assert not reset.disabled
    reset.click().run()
    assert app.session_state["comparison"] is None and app.session_state["scenario_data"] is None
    assert not app.metric                                # the results are gone
    assert any("press Apply shock" in i.value for i in app.info)


def test_a_shock_survives_moving_to_another_page(app):
    [b for b in app.main.button if b.label == "Apply shock"][0].click().run()
    app.switch_page("app_pages/portfolio_map.py").run()
    assert not app.exception
    assert any("Shock:" in m.value for m in app.sidebar.markdown)
    app.switch_page("app_pages/shock_simulator.py").run()
    assert {m.label: m.value for m in app.metric}["Unmet hours"] == "364"


# ---------- bottom line, project status, questions for the owner ----------
def test_bottom_line_for_the_resource_shock(baseline, r15):
    text = bottom_line(r15["comparison"], r15["scenario_label"])
    assert text == ("Rohit Sharma (R15) unavailable, weeks 4-5. Unmet hours rise from 288h to 364h (+76h). "
                    "P01, P02 and P06 lose cover. P05 and P09 become exposed through their dependencies.")


def test_bottom_line_for_a_single_project(p04):
    text = bottom_line(p04["comparison"], p04["scenario_label"])
    assert text.endswith("Unmet hours rise from 288h to 324h (+36h). P04 loses cover. P03 becomes exposed through its dependencies.")


def test_bottom_line_when_nothing_changes(baseline):
    shock = run_shock(baseline, SHOCK_RESOURCE, resource_id="R25", weeks=[12])
    assert bottom_line(shock["comparison"], "X") == "X. This shock does not create any new unmet hours in the plan."


def test_the_bottom_line_never_recommends_or_predicts(r15, p01, p04):
    for shock in (r15, p01, p04):
        text = bottom_line(shock["comparison"], shock["scenario_label"]).lower()
        for banned in ["recommend", "should", "best", "will slip", "delay"]:
            assert banned not in text


def test_project_status_table_for_the_r15_shock(baseline, r15):
    t = project_status_table(r15["scenario_data"], baseline["snapshot"], r15["scenario_snapshot"]).set_index("Project")
    assert list(t.columns) == ["Window", "Required h", "Unmet h", "Change h", "Status", "Dependencies"]
    expected = {                       # unmet, change vs baseline, status: hand-checked against the Excel
        "P01 Cloud Migration": ("24", "+24", "▲ Gap", "feeds P05"),
        "P02 Customer Portal Redesign": ("28", "+28", "▲ Gap", "feeds P09"),
        "P03 Mobile App v2": ("0", "no change", "◇ Exposed", "after P04"),
        "P04 Security Upgrade": ("70", "no change", "■ Large gap", "feeds P03"),
        "P05 Analytics Platform": ("0", "no change", "◇ Exposed", "after P01, P06"),
        "P06 Data Migration": ("24", "+24", "▲ Gap", "feeds P05"),
        "P07 Internal Automation": ("0", "no change", "● Staffed", "independent"),
        "P08 Reporting System Modernization": ("218", "no change", "■ Large gap", "independent"),
        "P09 Partner API Integration": ("0", "no change", "◇ Exposed", "after P02"),
    }
    assert len(t) == 9
    for name, (unmet, change, status, dep) in expected.items():
        row = t.loc[name]
        assert (row["Unmet h"], row["Change h"], row["Status"], row["Dependencies"]) == (unmet, change, status, dep), name
    assert t.loc["P01 Cloud Migration", "Window"] == "Wk 1-10" and t.loc["P01 Cloud Migration", "Required h"] == "456"


def test_project_status_table_follows_a_moved_deadline(baseline, p01):
    t = project_status_table(p01["scenario_data"], baseline["snapshot"], p01["scenario_snapshot"]).set_index("Project")
    assert t.loc["P01 Cloud Migration", "Window"] == "Wk 1-6"
    assert t.loc["P01 Cloud Migration", "Change h"] == "+188"


def test_owner_questions_for_the_r15_shock(baseline, r15):
    q = owner_questions(baseline["snapshot"], r15["scenario_snapshot"], r15["comparison"])
    assert [title for title, _ in q] == ["Who covers Q011 (28h more unmet)?",
                                         "Is a slip on P01, P02 and P06 acceptable?",
                                         "What about the 288h already unmet?"]
    assert q[1][1].startswith("P05 and P09 depend on them.")
    assert q[2][1] == "The largest is Q043 (162h on P08). This shock did not cause it."


def test_owner_question_names_a_scarce_skill_correctly(baseline, p04):
    q = owner_questions(baseline["snapshot"], p04["scenario_snapshot"], p04["comparison"])
    assert q[0][0] == "Who covers Q020 (14h more unmet)?"
    assert q[0][1] == "Only one person meets the minimum level for Cloud Infrastructure, so anyone else would be below it."
    assert q[1][0] == "Is a slip on P04 acceptable?" and q[1][1].startswith("P03 depends on it.")


def test_no_questions_when_nothing_changes(baseline):
    shock = run_shock(baseline, SHOCK_RESOURCE, resource_id="R25", weeks=[12])
    assert owner_questions(baseline["snapshot"], shock["scenario_snapshot"], shock["comparison"]) == []


def test_the_questions_never_recommend(baseline, r15, p01, p04):
    for shock in (r15, p01, p04):
        for title, detail in owner_questions(baseline["snapshot"], shock["scenario_snapshot"], shock["comparison"]):
            assert not any(w in (title + detail).lower() for w in ["recommend", "should", "best option"])


def test_the_page_shows_the_bottom_line_table_and_questions(page):
    page.button[0].click().run()
    assert not page.exception
    html = " ".join(m.value for m in page.markdown)
    assert "Unmet hours rise from 288h to 364h (+76h)." in html and "Bottom line" in html
    assert "Decisions for the program owner" in html and "Who covers Q011" in html
    status = page.table[0].value.set_index("Project")["Status"]
    assert status["P04 Security Upgrade"] == "■ Large gap" and status["P09 Partner API Integration"] == "◇ Exposed"
    assert any("Large gap" in c.value and "dependency" in c.value for c in page.caption)
    assert any("Start here" in c.value for c in page.caption)

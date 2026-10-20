"""Allocation Planner. Expected numbers were hand-calculated from the raw Excel file and the fit formula
(50 x skill match + 30 x availability + 20 x headroom), not copied from the engine."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app_state import load_baseline
from src.data_loader import load_data
from src.planner_view import (
    candidate_summary,
    candidate_view,
    requirement_conflicts,
    requirement_facts,
    requirement_options,
    weekly_figure,
)
from src.scenarios import take_snapshot

PAGE = str(Path(__file__).resolve().parent.parent / "app_pages" / "allocation_planner.py")


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def snap(data):
    return take_snapshot(data)


# ---------- choosing a requirement ----------
def test_unmet_requirements_come_first_largest_gap_first(data, snap):
    o = requirement_options(data, snap)
    assert list(o["RequirementID"]) == ["Q043", "Q020", "Q045"]          # 162h, 70h, 56h
    assert list(o["GapHrs"]) == [162, 70, 56]


def test_all_requirements_can_be_listed(data, snap):
    o = requirement_options(data, snap, only_unmet=False)
    assert len(o) == 50
    assert list(o["RequirementID"][:3]) == ["Q043", "Q020", "Q045"]      # unmet ones still first


def test_labels_say_what_the_requirement_is(data, snap):
    label = requirement_options(data, snap).iloc[0]["Label"]
    assert label == "Q043 | P08 Reporting System Modernization | Data Visualization (min 5) | 162h unmet"


def test_fully_staffed_labels_carry_no_unmet_text(data, snap):
    o = requirement_options(data, snap, only_unmet=False).set_index("RequirementID")
    assert "unmet" not in o.loc["Q001", "Label"]


# ---------- requirement facts ----------
def test_facts_for_q043(data, snap):
    f = requirement_facts(data, snap, "Q043")
    assert (f["ProjectID"], f["Skill"], f["MinProficiency"], f["HoursPerWeek"]) == ("P08", "Data Visualization", 5, 18)
    assert (f["StartWeek"], f["EndWeek"]) == (3, 11)
    assert (f["RequiredHrs"], f["AllocatedHrs"], f["GapHrs"]) == (162, 0, 162)     # 18h x 9 weeks
    assert (f["QualifiedCount"], f["Scarcity"]) == (1, "Single point of failure")


def test_facts_for_q020_and_a_staffed_requirement(data, snap):
    f = requirement_facts(data, snap, "Q020")
    assert (f["RequiredHrs"], f["GapHrs"], f["HoursPerWeek"], f["StartWeek"], f["EndWeek"]) == (70, 70, 10, 4, 10)
    q001 = requirement_facts(data, snap, "Q001")
    assert (q001["RequiredHrs"], q001["AllocatedHrs"], q001["GapHrs"]) == (120, 120, 0)


def test_unknown_requirement_is_a_clear_error(data, snap):
    with pytest.raises(ValueError, match="Unknown requirement"):
        requirement_facts(data, snap, "Q999")
    with pytest.raises(ValueError, match="Unknown requirement"):
        candidate_view(data, snap, "Q999")


# ---------- candidates ----------
def test_q043_candidates_and_the_misleading_top_scores(data, snap):
    c = candidate_view(data, snap, "Q043").set_index("ResourceID")
    assert len(c) == 6
    assert list(c.index[:1]) == ["R12"] and c.index[-1] == "R11"          # sorted by fit score
    assert list(c.index[c["MeetsMinimum"]]) == ["R11"]                     # only R11 has level 5
    # R11: skill 5/5 = 1.0; free 4h of 18h needed = 0.2222; peak load 90% -> headroom 0.1
    assert c.loc["R11", "FitScore"] == pytest.approx(50 * 1 + 30 * (4 / 18) + 20 * 0.1)
    # R12: level 4 of 5 = 0.8; 24h free >= 18h so availability 1; peak load 40% -> headroom 0.6
    assert c.loc["R12", "FitScore"] == pytest.approx(50 * 0.8 + 30 * 1 + 20 * 0.6)
    assert c.loc["R12", "FitScore"] > c.loc["R11", "FitScore"]


def test_q045_has_five_people_who_meet_the_minimum(data, snap):
    c = candidate_view(data, snap, "Q045")
    assert len(c) == 6 and int(c["MeetsMinimum"].sum()) == 5               # R05 (level 2) is below the minimum 3
    r20 = c.set_index("ResourceID").loc["R20"]
    assert r20["FitScore"] == pytest.approx(50 * 1 + 30 * 1 + 20 * 0.75)  # peak load 25% -> headroom 0.75


# ---------- the plain-words summary ----------
def test_summary_for_q043_warns_that_the_top_score_is_not_qualified(data, snap):
    f = requirement_facts(data, snap, "Q043")
    lines = candidate_summary(f, candidate_view(data, snap, "Q043"))
    assert lines[0] == "6 people have the Data Visualization skill; 1 of them meets the minimum level (5)."
    assert "None of the people who meet the minimum has 18h free every week" in lines[1]
    assert "most free hours in their tightest week is 4h" in lines[1]
    assert "Lakshmi Devi" in lines[2] and "below the minimum proficiency" in lines[2]


def test_summary_for_q020(data, snap):
    f = requirement_facts(data, snap, "Q020")
    lines = candidate_summary(f, candidate_view(data, snap, "Q020"))
    assert lines[0] == "5 people have the Cloud Infrastructure skill; 1 of them meets the minimum level (4)."
    assert "is 2h" in lines[1]                                              # R15 has 2h free in the tightest week
    assert "Neha Gupta" in lines[2]                                         # highest score, but level 3 < 4


def test_summary_for_q045_has_no_warning_because_the_top_candidate_qualifies(data, snap):
    f = requirement_facts(data, snap, "Q045")
    lines = candidate_summary(f, candidate_view(data, snap, "Q045"))
    assert lines[1] == "5 of them have the full 8h free in every week of the window."
    assert len(lines) == 2


def test_summary_when_nobody_has_the_skill(data, snap):
    f = dict(requirement_facts(data, snap, "Q043"))
    empty = candidate_view(data, snap, "Q043").iloc[0:0]
    assert candidate_summary(f, empty) == ["Nobody in the dataset has the Data Visualization skill."]


def test_summary_when_nobody_meets_the_minimum(data, snap):
    f = requirement_facts(data, snap, "Q043")
    c = candidate_view(data, snap, "Q043")
    c = c[~c["MeetsMinimum"]]
    assert "Nobody meets the minimum proficiency" in candidate_summary(f, c)[1]


def test_the_planner_never_recommends(data, snap):
    for rid in ["Q020", "Q043", "Q045"]:
        f = requirement_facts(data, snap, rid)
        c = candidate_view(data, snap, rid)
        text = " ".join(candidate_summary(f, c) + list(c.columns)).lower()
        for word in ["recommend", "best", "should", "choose", "pick"]:
            assert word not in text, (rid, word)


# ---------- conflicts and chart ----------
def test_conflicts_for_one_requirement(data, snap):
    c = requirement_conflicts(snap, "Q020")
    assert sorted(c["ConflictType"]) == ["Capacity shortage", "Competing requirements", "Staffing gap"]
    assert requirement_conflicts(snap, "Q001").empty


def test_weekly_chart_for_an_unstaffed_requirement(snap):
    fig = weekly_figure(snap, "Q043")
    assert list(fig.data[0].x) == [f"Wk {w}" for w in range(3, 12)]
    assert sum(fig.data[0].y) == 0 and list(fig.data[1].y) == [18] * 9


def test_weekly_chart_for_a_staffed_requirement(snap):
    fig = weekly_figure(snap, "Q001")
    assert sum(fig.data[0].y) == 120 and sum(fig.data[1].y) == 0 and len(fig.data[0].x) == 10


# ---------- the real page, run headlessly ----------
@pytest.fixture
def page():
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["baseline"] = load_baseline()
    return at.run()


def test_page_opens_on_the_biggest_gap(page):
    assert not page.exception
    assert page.header[0].value == "Allocation Planner"
    assert page.subheader[0].value == "Q043: Data Visualization on Reporting System Modernization"
    assert {m.label: m.value for m in page.metric}["Unmet hours"] == "162 of 162h"
    assert len(page.selectbox[0].options) == 3


def test_page_shows_the_misleading_top_score_warning(page):
    text = " ".join(m.value for m in page.markdown)
    assert "Lakshmi Devi" in text and "A high score is not the same as being qualified." in text


def test_choosing_another_requirement_changes_the_page(page):
    q020 = [o for o in page.selectbox[0].options if o.startswith("Q020")][0]
    page.selectbox[0].select(q020).run()
    assert page.subheader[0].value.startswith("Q020: Cloud Infrastructure on Security Upgrade")
    assert {m.label: m.value for m in page.metric}["Unmet hours"] == "70 of 70h"


def test_unticking_the_filter_lists_every_requirement(page):
    page.checkbox[0].uncheck().run()
    assert not page.exception
    assert len(page.selectbox[0].options) == 50
    assert page.subheader[0].value.startswith("Q043")


def test_a_fully_staffed_requirement_shows_no_problems_table(page):
    page.checkbox[0].uncheck().run()
    q001 = [o for o in page.selectbox[0].options if o.startswith("Q001")][0]
    page.selectbox[0].select(q001).run()
    assert not page.exception
    assert {m.label: m.value for m in page.metric}["Unmet hours"] == "0 of 120h"
    assert not any("Problems the engine found" in m.value for m in page.markdown)


# ---------- readability: nothing important may be cut off ----------
def test_no_tile_value_is_long_enough_to_be_cut_off(page):
    # long text in a big tile shows as "Software ..." on a laptop screen; keep tile values short
    for m in page.metric:
        assert len(m.value) <= 16, (m.label, m.value)
        assert len(m.label) <= 14, m.label


def test_role_is_shown_in_full_as_normal_text(page):
    text = " ".join(m.value for m in page.markdown)
    assert "**Role needed:** Data Analyst" in text
    assert "Single point of failure (1 qualified)" in text


def _candidate_table(page):
    return [x for x in page.table if "Fit score" in x.value.columns][0].value


def test_candidate_table_puts_the_answer_columns_first(page):
    table = _candidate_table(page).reset_index()
    assert list(table.columns[:6]) == ["ID", "Name", "Fit score", "Fit", "Flags", "Meets min"]


def test_candidate_table_has_no_empty_cells_and_no_checkboxes(page):
    table = _candidate_table(page)
    assert (table["Flags"] != "").all()
    assert set(table["Meets min"]) <= {"Yes", "No"}


def test_the_chart_has_a_heading(page):
    assert any("Hours needed each week" in m.value for m in page.markdown)


def test_the_formula_uses_a_multiplication_sign(page):
    text = " ".join(m.value for m in page.markdown)
    assert "× skill match" in text and " x skill match" not in text


def test_a_staffed_requirement_names_who_is_already_planned():
    from src.data_loader import load_data
    from src.planner_view import assignee_line, candidate_table, candidate_view, current_assignees, requirement_facts
    from src.scenarios import take_snapshot
    data = load_data()
    snapshot = take_snapshot(data)
    who = current_assignees(data, "Q001")
    line = assignee_line(who)
    assert "Rohit Sharma (R15), 120h" in line
    table = candidate_table(candidate_view(data, snapshot, "Q001"), who["ResourceID"])
    flags = table.set_index("ID").loc["R15", "Flags"]
    assert flags.startswith("Currently planned")
    from src.planner_view import candidate_summary
    text = " ".join(candidate_summary(requirement_facts(data, snapshot, "Q001"), candidate_view(data, snapshot, "Q001")))
    assert "fully staffed" in text and "None of the people who meet" not in text


def test_an_unstaffed_requirement_keeps_the_availability_sentence(page):
    text = " ".join(x.value for x in page.markdown)
    assert "fully staffed" not in text


def test_level_is_a_whole_number(page):
    table = _candidate_table(page)
    assert str(table["Level"].dtype).startswith("int")

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_the_choice_survives_leaving_the_page_and_coming_back():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.switch_page("app_pages/allocation_planner.py").run()
    at.checkbox[0].set_value(False).run()
    label = [o for o in at.selectbox[0].options if o.startswith("Q001")][0]
    at.selectbox[0].set_value(label).run()
    at.switch_page("app_pages/portfolio_map.py").run()
    at.switch_page("app_pages/allocation_planner.py").run()
    assert at.checkbox[0].value is False
    assert at.selectbox[0].value == label
    assert not at.exception

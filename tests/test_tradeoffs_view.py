"""Trade-offs. Expected numbers are hand-calculated from the raw Excel, not taken from the engine.

Setup used throughout: Rohit Sharma (R15) unavailable in weeks 4-5.
  Baseline unmet 288h. R15 holds 12h/week on Q001 in those weeks, so Q001 loses 24h (and the rest of
  his 76h is on P02 and P06): unmet becomes 364h.

Responses checked by hand
  * Substitute R16 on Q001, weeks 4-5: 2 weeks x 12h = 24h covered      -> unmet 364 - 24 = 340.
  * Reallocate R26 from Q028 to Q043, weeks 3-4, 8h/week: 16h moved      -> unmet unchanged (364);
        Q043 162 -> 146 (18h/week gap in weeks 3-4 less 8) and Q028 0 -> 16 (the problem moved).
  * Extend P01 from week 10 to 12: total effort is constant, so weekly demand falls and the new weeks
        start unstaffed. Q001 (12h x 10 = 120h -> 10h x 12): weeks 4,5,11,12 unmet = 4 x 10 = 40 (was 24).
        Q004 (8h x 6 = 48h -> 6h x 8): weeks 11,12 = 12h.   Q005 (10h x 10 -> 8.33h x 12): weeks 11,12 = 16.67h.
"""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.app_state import SESSION_DEFAULTS, load_baseline
from src.shock_view import SHOCK_RESOURCE, run_shock
from src.tradeoffs_view import (
    RESPONSE_KINDS,
    RESPONSE_REALLOCATE,
    RESPONSE_SUBSTITUTE,
    RESPONSE_TIMELINE,
    apply_response,
    qualification_note,
    changes_display,
    person_names,
    style_changes,
    compress_weeks,
    comparison_table,
    cost_lines,
    extendable_projects,
    fixed_lines,
    gap_weeks,
    reallocate_limit,
    reallocate_sources,
    run_response,
    still_open_line,
    substitute_people,
    three_way_figure,
    timeline_note,
    unmet_options,
    unavailable_people,
    warnings_display,
)

PAGE = str(Path(__file__).resolve().parent.parent / "app_pages" / "tradeoffs.py")


@pytest.fixture(scope="module")
def baseline():
    return load_baseline()


@pytest.fixture(scope="module")
def shock(baseline):
    return run_shock(baseline, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])


@pytest.fixture(scope="module")
def sub(baseline, shock):
    return run_response(baseline, shock["scenario_data"], RESPONSE_SUBSTITUTE,
                        resource_id="R16", requirement_id="Q001", weeks=[4, 5])


@pytest.fixture(scope="module")
def realloc(baseline, shock):
    return run_response(baseline, shock["scenario_data"], RESPONSE_REALLOCATE, resource_id="R26",
                        requirement_id="Q043", from_requirement="Q028", weeks=[3, 4], hours_per_week=8.0)


@pytest.fixture(scope="module")
def timeline(baseline, shock):
    return run_response(baseline, shock["scenario_data"], RESPONSE_TIMELINE, project_id="P01", new_deadline=12)


def unmet(result_or_snapshot):
    snap = result_or_snapshot["response_result"]["snapshot"] if "response_result" in result_or_snapshot else result_or_snapshot
    return snap["metrics"]["Unmet hours (staffing gap)"]


def gap(result, requirement_id):
    d = result["response_result"]["snapshot"]["demand"].set_index("RequirementID")
    return float(d.loc[requirement_id, "GapHrs"])


# ---------- choices ----------
def test_the_three_responses_are_the_engine_responses():
    assert RESPONSE_KINDS == [RESPONSE_SUBSTITUTE, RESPONSE_REALLOCATE, RESPONSE_TIMELINE]


def test_unmet_requirements_after_the_shock_largest_first(shock):
    options = unmet_options(shock["scenario_data"], shock["scenario_snapshot"])
    assert list(options["RequirementID"]) == ["Q043", "Q020", "Q045", "Q011", "Q001", "Q035"]
    assert list(options["GapHrs"]) == [162, 70, 56, 28, 24, 24]


def test_weeks_with_unmet_hours(shock):
    assert gap_weeks(shock["scenario_snapshot"], "Q001") == [4, 5]
    assert gap_weeks(shock["scenario_snapshot"], "Q043") == list(range(3, 12))


def test_a_person_who_is_unavailable_is_not_offered_as_cover(shock):
    data, snap = shock["scenario_data"], shock["scenario_snapshot"]
    assert unavailable_people(data, [4, 5]) == {"R15"}
    assert unavailable_people(data, [6]) == set()
    assert unavailable_people(data, []) == set()
    with_weeks = substitute_people(data, snap, "Q001", [4, 5])
    assert "R15" not in set(with_weeks["ResourceID"])
    assert "R15" in set(substitute_people(data, snap, "Q001")["ResourceID"])      # without weeks nobody is filtered


def test_the_cover_list_is_sorted_by_fit_and_says_when_someone_is_below_the_minimum(shock):
    people = substitute_people(shock["scenario_data"], shock["scenario_snapshot"], "Q001", [4, 5])
    assert people.iloc[0]["ResourceID"] == "R16"
    assert people.iloc[0]["Label"] == "R16 Neha Gupta | fit 78 (Partial) | below minimum level"


def test_no_baseline_unavailability_means_nobody_is_filtered(baseline):
    assert unavailable_people(baseline["data"], [4, 5]) == set()


def test_sources_are_the_other_requirements_the_person_holds(shock):
    s = reallocate_sources(shock["scenario_data"], "R26", "Q043")
    assert "Q028" in set(s["RequirementID"]) and "Q043" not in set(s["RequirementID"])
    row = s[s["RequirementID"] == "Q028"].iloc[0]
    assert row["Label"] == "Q028 | P05 Requirements Analysis | 56h planned"      # 8h x weeks 3-9 = 7 weeks = 56h


def test_the_most_hours_that_can_move_is_the_smaller_of_held_and_unmet(shock):
    data, snap = shock["scenario_data"], shock["scenario_snapshot"]
    # R26 holds 8h/week on Q028; Q043 has 18h/week unmet (162h over weeks 3-11)
    assert reallocate_limit(data, snap, "R26", "Q028", "Q043", [3, 4]) == 8
    # R12 holds 16h/week on Q042: still 16 (below Q043's 18)
    assert reallocate_limit(data, snap, "R12", "Q042", "Q043", [3, 4]) == 16
    # a week where the person holds nothing on the source: impossible
    assert reallocate_limit(data, snap, "R26", "Q028", "Q043", [2, 3]) == 0
    assert reallocate_limit(data, snap, "R26", "Q028", "Q043", []) == 0


def test_only_projects_that_can_still_move_later_are_offered(shock):
    p = extendable_projects(shock["scenario_data"])
    assert list(p["ProjectID"]) == ["P01", "P02", "P04", "P06", "P08", "P09"]
    assert list(p["Deadline"]) == [10, 9, 10, 8, 11, 11]


# ---------- applying ----------
def test_labels_say_what_was_done(shock):
    data = shock["scenario_data"]
    assert apply_response(data, RESPONSE_SUBSTITUTE, resource_id="R16", requirement_id="Q001", weeks=[4, 5])[1] == \
        "R16 covers the unmet hours of Q001, weeks 4-5"
    assert apply_response(data, RESPONSE_REALLOCATE, resource_id="R26", requirement_id="Q043", from_requirement="Q028",
                          weeks=[3, 4], hours_per_week=8.0)[1] == "R26 moves 8h a week from Q028 to Q043, weeks 3-4"
    assert apply_response(data, RESPONSE_TIMELINE, project_id="P01", new_deadline=12)[1] == "P01 deadline moves to week 12"
    assert apply_response(data, RESPONSE_SUBSTITUTE, resource_id="R16", requirement_id="Q001", weeks=[4])[1].endswith("week 4")


def test_a_move_the_rules_forbid_raises_a_plain_error(shock):
    with pytest.raises(ValueError):                         # R15 is unavailable in those weeks
        apply_response(shock["scenario_data"], RESPONSE_SUBSTITUTE, resource_id="R15", requirement_id="Q001", weeks=[4, 5])
    with pytest.raises(ValueError):
        apply_response(shock["scenario_data"], "Wave a wand")


def test_a_response_changes_neither_the_shocked_plan_nor_the_baseline(baseline, shock):
    before_shock = {k: v.copy() for k, v in shock["scenario_data"].items()}
    before_base = {k: v.copy() for k, v in baseline["data"].items()}
    run_response(baseline, shock["scenario_data"], RESPONSE_SUBSTITUTE, resource_id="R16", requirement_id="Q001", weeks=[4, 5])
    for name, frame in before_shock.items():
        pd.testing.assert_frame_equal(shock["scenario_data"][name], frame)
    for name, frame in before_base.items():
        pd.testing.assert_frame_equal(baseline["data"][name], frame)


def test_the_response_is_built_on_the_shocked_plan_not_the_baseline(sub):
    assert unmet(sub) == 340                                # 364 - 24. Built on the baseline it would be 288 - 0.


def test_run_response_returns_what_the_page_keeps(sub):
    assert set(sub) == {"response_data", "response_label", "response_kind", "response_result"}
    assert sub["response_kind"] == RESPONSE_SUBSTITUTE
    assert set(sub["response_result"]) == {"snapshot", "evaluation"}
    assert set(SESSION_DEFAULTS) >= set(sub)


# ---------- the three responses, by hand ----------
def test_substitute_covers_exactly_the_unmet_hours(sub):
    assert unmet(sub) == 340
    assert gap(sub, "Q001") == 0
    table = changes_display(sub["response_result"]["evaluation"])
    assert len(table) == 1                       # weeks 4 and 5 are one run
    row = table.iloc[0]
    assert (row["Person"], row["Weeks"], row["Change h/wk"], row["Total h"]) == ("R16", "4-5", 12, 24)


def test_reallocate_moves_the_problem_without_solving_it(realloc):
    assert unmet(realloc) == 364                            # nothing was created
    assert gap(realloc, "Q043") == 146                      # 162 - 16
    assert gap(realloc, "Q028") == 16                       # the donor now has the gap
    t = changes_display(realloc["response_result"]["evaluation"])
    assert list(t["Change h/wk"]) == [-8, 8] and list(t["Total h"]) == [-16, 16] and set(t["Weeks"]) == {"3-4"}


def test_extending_a_deadline_does_not_add_anyone_so_unmet_hours_can_rise(timeline):
    assert gap(timeline, "Q001") == pytest.approx(40)
    assert gap(timeline, "Q004") == pytest.approx(12)
    assert gap(timeline, "Q005") == pytest.approx(100 / 12 * 2)
    assert unmet(timeline) == pytest.approx(364 + 16 + 12 + 100 / 12 * 2)      # 364 + 44.67 = 408.67
    assert changes_display(timeline["response_result"]["evaluation"]).empty


# ---------- reading the result ----------
def test_comparison_table_has_three_plans_and_the_change(baseline, shock, sub):
    t = comparison_table(baseline["snapshot"], shock["scenario_snapshot"], sub["response_result"]["snapshot"])
    assert list(t.columns) == ["Measure", "Baseline", "After shock", "After response", "Change from the response"]
    row = t[t["Measure"] == "Unmet hours"].iloc[0]
    assert (row["Baseline"], row["After shock"], row["After response"], row["Change from the response"]) == \
        ("288", "364", "340", "-24")
    assert t[t["Measure"] == "Over-allocated hours"].iloc[0]["Change from the response"] == "no change"
    assert len(t) == 5


def test_what_it_fixed_for_the_substitute(sub):
    lines = fixed_lines(sub["response_result"]["evaluation"])
    assert lines[0] == "Unmet hours fall by 24h (364h to 340h)."
    assert "Staffing gap on Q001 (24h to 0h)" in lines[1]


def test_what_it_fixed_says_so_when_nothing_was_recovered(realloc, timeline):
    lines = fixed_lines(realloc["response_result"]["evaluation"])
    assert lines[0] == "Unmet hours do not change (364h)."
    assert "Staffing gap on Q043 (162h to 146h)" in lines[1]            # reduced, not resolved: still counts as fixed
    assert "Capacity shortage on Q043 (82h to 74h)" in lines[1]
    assert fixed_lines(timeline["response_result"]["evaluation"])[0] == "Unmet hours do not fall; they rise by 45h."


def test_what_it_cost_the_substitute_a_proficiency_warning(sub):
    lines = cost_lines(sub["response_result"]["evaluation"], RESPONSE_SUBSTITUTE)
    assert lines == ["Warning, R16: 24h on Q001 (weeks 4-5) are staffed by someone below the required proficiency level. "
                     "They count as staffed, but qualification risk remains."]


def test_qualification_note_is_calculated_and_empty_when_nobody_is_below_level(sub, realloc):
    note = qualification_note(sub["response_result"]["evaluation"])
    assert note.startswith("Qualification: 24h of the hours this response staffs are with people below the required proficiency.")
    assert "not fully recovered" in note
    ev = sub["response_result"]["evaluation"]
    empty = {**ev, "warnings": ev["warnings"].iloc[0:0]}
    assert qualification_note(empty) == ""


def test_what_it_cost_the_reallocation_a_new_gap_and_two_warnings(realloc):
    lines = cost_lines(realloc["response_result"]["evaluation"], RESPONSE_REALLOCATE)
    assert any("Staffing gap on Q028 (0h to 16h)" in line for line in lines)
    assert any("below the required proficiency level" in line for line in lines)
    assert any("role mismatch" in line for line in lines)
    assert lines[-1] == "The hours were moved, not created: the problem moved to another requirement."


def test_what_it_cost_the_extension_the_unstaffed_new_weeks(timeline):
    lines = cost_lines(timeline["response_result"]["evaluation"], RESPONSE_TIMELINE)
    text = " ".join(lines)
    for expected in ["Staffing gap on Q004 (0h to 12h)", "Staffing gap on Q001 (24h to 40h)"]:
        assert expected in text


def test_a_response_with_no_cost_says_so():
    empty = pd.DataFrame(columns=["Status", "ConflictType", "RequirementID", "ResourceID", "BaselineHours", "ScenarioHours"])
    metrics = pd.DataFrame({"Metric": ["Downstream projects exposed"], "Baseline": [1.0], "Scenario": [1.0], "Delta": [0.0]})
    evaluation = {"vs_shock": {"metrics": metrics, "conflict_changes": empty},
                  "warnings": pd.DataFrame(columns=["ResourceID", "RequirementID", "Weeks", "Hours", "Warning"])}
    assert cost_lines(evaluation, RESPONSE_SUBSTITUTE) == ["No new conflicts or warnings were created."]


def test_still_open_compared_with_the_original_plan(sub, realloc):
    assert still_open_line(sub["response_result"]["evaluation"]) == \
        "Compared with the original plan, 52h more is still unmet (340h against 288h)."
    assert "76h more" in still_open_line(realloc["response_result"]["evaluation"])


def test_still_open_when_back_at_the_original_level():
    m = pd.DataFrame({"Metric": ["Unmet hours (staffing gap)"], "Baseline": [288.0], "Scenario": [288.0], "Delta": [0.0]})
    assert still_open_line({"vs_baseline": {"metrics": m}}) == "Unmet hours are back to the original plan's level."


def test_week_lists_are_compressed_into_ranges():
    assert compress_weeks("3, 4, 5, 6, 7, 8, 9, 10, 11") == "3-11"
    assert compress_weeks("4, 5") == "4-5"
    assert compress_weeks("2, 4, 5, 9") == "2, 4-5, 9"
    assert compress_weeks("7") == "7"


def test_hours_in_the_moved_table_are_plain_numbers(realloc):
    t = changes_display(realloc["response_result"]["evaluation"])
    assert list(t["Before h/wk"]) == [8, 0] and list(t["After h/wk"]) == [0, 8]
    shown = style_changes(t).to_html()
    assert "+8" in shown and "-8" in shown and "8.0" not in shown


def test_the_timeline_note_only_appears_for_the_timeline_response():
    assert "adds nobody" in timeline_note(RESPONSE_TIMELINE)
    assert timeline_note(RESPONSE_SUBSTITUTE) == "" and timeline_note(RESPONSE_REALLOCATE) == ""


def test_warning_table_is_whole_hours_with_friendly_headers(sub):
    w = warnings_display(sub["response_result"]["evaluation"])
    assert list(w.columns) == ["Person", "Requirement", "Weeks", "Hours", "Warning"]
    assert w.iloc[0]["Hours"] == 24 and w.iloc[0]["Warning"] == "Below required proficiency"


def test_three_way_chart_has_baseline_shock_and_response(baseline, shock, sub):
    fig = three_way_figure(baseline["snapshot"], shock["scenario_snapshot"], sub["response_result"]["snapshot"])
    assert [t.name for t in fig.data] == ["Baseline", "After shock", "After response"]
    assert list(fig.data[0].x) == sorted(fig.data[0].x)
    assert sum(fig.data[0].y) == 288 and sum(fig.data[1].y) == 364 and sum(fig.data[2].y) == 340


def test_the_page_text_never_ranks_or_recommends(sub, realloc, timeline):
    for result, kind in [(sub, RESPONSE_SUBSTITUTE), (realloc, RESPONSE_REALLOCATE), (timeline, RESPONSE_TIMELINE)]:
        ev = result["response_result"]["evaluation"]
        text = " ".join(fixed_lines(ev) + cost_lines(ev, kind) + [still_open_line(ev), timeline_note(kind)]).lower()
        for banned in ["recommend", "best option", "should ", "optimal", "predict"]:
            assert banned not in text


# ---------- the real page, run headlessly ----------
@pytest.fixture
def page(baseline):
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.session_state["baseline"] = baseline
    for key, default in SESSION_DEFAULTS.items():
        at.session_state[key] = default
    return at.run()


def shocked_page(page, baseline):
    shock = run_shock(baseline, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])
    for key, value in shock.items():
        page.session_state[key] = value
    return page.run()


def button(page, label):
    return [b for b in page.button if b.label == label][0]


def test_without_a_shock_the_page_asks_for_one(page):
    assert not page.exception
    assert page.header[0].value == "Trade-offs"
    assert any("Apply a shock first" in i.value for i in page.info)
    assert not page.radio


def test_with_a_shock_the_page_shows_the_choices_and_no_result_yet(page, baseline):
    page = shocked_page(page, baseline)
    assert not page.exception
    assert any("Rohit Sharma (R15) unavailable, weeks 4-5" in m.value for m in page.markdown)
    assert page.radio[0].options == RESPONSE_KINDS
    assert page.selectbox[0].value.startswith("Q043")
    assert any("press Try this response" in i.value for i in page.info)


def test_trying_a_substitute_on_the_page(page, baseline):
    page = shocked_page(page, baseline)
    page.selectbox[0].select([o for o in page.selectbox[0].options if o.startswith("Q001")][0]).run()
    assert page.slider[0].value == (4, 5)
    assert page.selectbox[1].value.startswith("R16")
    button(page, "Try this response").click().run()
    assert not page.exception
    assert page.session_state["response_label"] == "R16 covers the unmet hours of Q001, weeks 4-5"
    assert page.subheader[-1].value == "What this response did"
    assert page.table[0].value.iloc[0].tolist() == ["Unmet hours", "288", "364", "340", "-24"]
    assert any("Unmet hours fall by 24h (364h to 340h)." in m.value for m in page.markdown)
    assert any("Warning, R16: 24h on Q001" in m.value for m in page.markdown)
    assert any(m.value.startswith("Qualification: 24h") for m in page.markdown)


def test_trying_a_reallocation_on_the_page(page, baseline):
    page = shocked_page(page, baseline)
    page.radio[0].set_value(RESPONSE_REALLOCATE).run()
    assert not page.exception
    person = [o for o in page.selectbox[1].options if o.startswith("R26")][0]
    page.selectbox[1].select(person).run()
    source = [o for o in page.selectbox[2].options if o.startswith("Q028")][0]
    page.selectbox[2].select(source).run()
    page.slider[0].set_value((3, 4)).run()
    page.slider[1].set_value(8).run()
    button(page, "Try this response").click().run()
    assert not page.exception
    assert page.session_state["response_label"] == "R26 moves 8h a week from Q028 to Q043, weeks 3-4"
    assert page.table[0].value.iloc[0].tolist() == ["Unmet hours", "288", "364", "364", "no change"]


def test_trying_an_extension_on_the_page(page, baseline):
    page = shocked_page(page, baseline)
    page.radio[0].set_value(RESPONSE_TIMELINE).run()
    assert not page.exception
    page.slider[0].set_value(12).run()
    button(page, "Try this response").click().run()
    assert page.session_state["response_label"] == "P01 deadline moves to week 12"
    assert page.table[0].value.iloc[0].tolist() == ["Unmet hours", "288", "364", "409", "+45"]
    assert any("adds nobody" in c.value for c in page.caption)


def test_a_new_response_replaces_the_last_one(page, baseline):
    page = shocked_page(page, baseline)
    button(page, "Try this response").click().run()          # default: first requirement, first person
    first = page.session_state["response_label"]
    page.radio[0].set_value(RESPONSE_TIMELINE).run()
    button(page, "Try this response").click().run()
    assert page.session_state["response_label"] != first and page.session_state["response_kind"] == RESPONSE_TIMELINE


def test_undo_clears_the_response_and_keeps_the_shock(page, baseline):
    page = shocked_page(page, baseline)
    button(page, "Try this response").click().run()
    assert page.session_state["response_data"] is not None
    button(page, "Undo this response").click().run()
    assert page.session_state["response_data"] is None and page.session_state["response_result"] is None
    assert page.session_state["scenario_data"] is not None
    assert any("press Try this response" in i.value for i in page.info)


def test_a_move_with_no_hours_to_move_is_explained_not_attempted(page, baseline):
    page = shocked_page(page, baseline)
    page.radio[0].set_value(RESPONSE_REALLOCATE).run()
    person = [o for o in page.selectbox[1].options if o.startswith("R26")][0]
    page.selectbox[1].select(person).run()
    source = [o for o in page.selectbox[2].options if o.startswith("Q028")][0]
    page.selectbox[2].select(source).run()
    page.slider[0].set_value((2, 3)).run()                   # R26 holds nothing on Q028 in week 2
    assert any("no hours to move" in w.value for w in page.warning)
    assert button(page, "Try this response").disabled


def test_no_cell_is_long_enough_to_be_cut_off(page, baseline):
    page = shocked_page(page, baseline)
    button(page, "Try this response").click().run()
    for table in page.table:
        for column in table.value.columns:
            assert len(str(column)) <= 26
    for m in page.metric:
        assert len(m.label) <= 16 and len(m.value) <= 16


# ---------- grouped rows, names, remembered form ----------
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_a_long_run_of_weeks_is_one_row_not_nine(baseline):
    shock = run_shock(baseline, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])
    result = run_response(baseline, shock["scenario_data"], RESPONSE_SUBSTITUTE,
                          resource_id="R12", requirement_id="Q043", weeks=list(range(3, 12)))
    t = changes_display(result["response_result"]["evaluation"])
    assert len(t) == 1
    row = t.iloc[0]
    assert (row["Requirement"], row["Weeks"], row["Change h/wk"], row["Total h"]) == ("Q043", "3-11", 18, 162)


def test_names_are_shown_next_to_the_ids(sub, shock):
    names = person_names(shock["scenario_data"])
    assert names["R16"] == "R16 Neha Gupta"
    t = changes_display(sub["response_result"]["evaluation"], names)
    assert t.iloc[0]["Person"] == "R16 Neha Gupta"
    assert warnings_display(sub["response_result"]["evaluation"], names).iloc[0]["Person"] == "R16 Neha Gupta"


def test_a_run_is_split_when_the_weeks_are_not_next_to_each_other(baseline):
    shock = run_shock(baseline, SHOCK_RESOURCE, resource_id="R15", weeks=[4, 5])
    result = run_response(baseline, shock["scenario_data"], RESPONSE_SUBSTITUTE,
                          resource_id="R16", requirement_id="Q001", weeks=[4, 5])
    ch = result["response_result"]["evaluation"]["allocation_changes"].copy()
    ch.loc[ch["Week"] == 5, "Week"] = 8
    t = changes_display({"allocation_changes": ch})
    assert list(t["Weeks"]) == ["4", "8"] and list(t["Total h"]) == [12, 12]


@pytest.fixture
def app():
    import streamlit as st
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.switch_page("app_pages/shock_simulator.py").run()
    [b for b in at.main.button if b.label == "Apply shock"][0].click().run()
    return at.switch_page("app_pages/tradeoffs.py").run()


def test_the_form_survives_moving_to_another_page(app):
    app.radio[0].set_value(RESPONSE_REALLOCATE).run()
    person = [o for o in app.selectbox[1].options if o.startswith("R26")][0]
    app.selectbox[1].select(person).run()
    source = [o for o in app.selectbox[2].options if o.startswith("Q028")][0]
    app.selectbox[2].select(source).run()
    app.slider[0].set_value((3, 4)).run()
    app.slider[1].set_value(5).run()
    app.switch_page("app_pages/portfolio_map.py").run()
    app.switch_page("app_pages/tradeoffs.py").run()
    assert not app.exception
    assert app.radio[0].value == RESPONSE_REALLOCATE
    assert app.selectbox[1].value == person and app.selectbox[2].value == source
    assert app.slider[0].value == (3, 4) and app.slider[1].value == 5


def test_changing_the_requirement_puts_the_weeks_back_to_the_full_range(app):
    q001 = [o for o in app.selectbox[0].options if o.startswith("Q001")][0]
    q043 = [o for o in app.selectbox[0].options if o.startswith("Q043")][0]
    app.selectbox[0].select(q001).run()
    assert app.slider[0].value == (4, 5)
    app.selectbox[0].select(q043).run()
    assert not app.exception
    assert app.slider[0].value == (3, 11)


def test_the_deadline_choice_survives_moving_to_another_page(app):
    app.radio[0].set_value(RESPONSE_TIMELINE).run()
    app.slider[0].set_value(12).run()
    app.switch_page("app_pages/portfolio_map.py").run()
    app.switch_page("app_pages/tradeoffs.py").run()
    assert not app.exception
    assert app.radio[0].value == RESPONSE_TIMELINE and app.slider[0].value == 12


def test_no_session_state_warning_for_value_and_key_together(app, caplog):
    app.radio[0].set_value(RESPONSE_REALLOCATE).run()
    assert not app.exception
    assert "created with a default value but also had its value set" not in caplog.text


def test_covering_fractional_hours_for_someone_who_already_works_on_the_requirement(baseline):
    """P04 +20% makes Q022 need 9.6h in a week; R19 already holds whole hours on Q022. This used to crash."""
    shock = run_shock(baseline, "A project needs more work", project_id="P04", percent=20.0)
    result = run_response(baseline, shock["scenario_data"], RESPONSE_SUBSTITUTE,
                          resource_id="R19", requirement_id="Q022", weeks=gap_weeks(shock["scenario_snapshot"], "Q022"))
    assert gap(result, "Q022") == pytest.approx(0)
    assert unmet(result) == pytest.approx(324 - 8)

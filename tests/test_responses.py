"""Response Engine: Substitute, Reallocate, Adjust timeline.

Expected numbers were calculated by hand from the raw Excel file (R15, R16, R18 and R20 loads and the
requirement windows), not from the engine. IDs appear only where a test checks a planted situation.
"""
import copy

import pandas as pd
import pytest

from src.data_loader import load_data
from src.responses import (
    CHANGE_COLUMNS,
    WARNING_COLUMNS,
    allocation_changes,
    apply_reallocate,
    apply_substitute,
    apply_timeline_extension,
    assignment_warnings,
    evaluate_response,
)
from src.scenarios import (
    apply_deadline_acceleration,
    apply_effort_increase,
    apply_resource_unavailable,
    compare_scenario,
)
from src.validator import validate_data


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def shocked(data):
    """R15 (the only strong cloud engineer) is out in weeks 4-5: Q001 24h, Q011 28h, Q035 24h become unmet."""
    return apply_resource_unavailable(data, "R15", [4, 5])


def metric(result, name, column="Delta"):
    return result["metrics"].set_index("Metric").loc[name, column]


def gap_of(data, requirement):
    from src.capacity import build_requirement_demand
    t = build_requirement_demand(data).set_index("RequirementID")
    return t.loc[requirement, "GapHrs"]


def load(data, resource, week):
    from src.capacity import build_capacity_table
    t = build_capacity_table(data)
    return t[(t["ResourceID"] == resource) & (t["Week"] == week)].iloc[0]["PlannedHrs"]


# =========================================================================
# Substitute
# =========================================================================
def test_substitute_covers_the_unmet_hours_hand_calculated(data, shocked):
    # R16 is a qualified cloud engineer (3 >= 3) with no work in week 4 and 10h in week 5
    after = apply_substitute(shocked, "Q011", "R16", [4, 5])
    assert gap_of(shocked, "Q011") == 28
    assert gap_of(after, "Q011") == 0
    assert load(after, "R16", 4) == 14       # 0 + 14
    assert load(after, "R16", 5) == 24       # 10 + 14, still under 40h
    result = compare_scenario(shocked, after)
    assert metric(result, "Unmet hours (staffing gap)") == -28
    assert metric(result, "Over-allocated resources") == 0
    assert validate_data(after) == []


def test_substitute_reports_exactly_which_hours_moved(shocked):
    after = apply_substitute(shocked, "Q011", "R16", [4, 5])
    changes = allocation_changes(shocked, after)
    assert list(changes.columns) == CHANGE_COLUMNS
    assert len(changes) == 2
    assert set(changes["ResourceID"]) == {"R16"} and set(changes["RequirementID"]) == {"Q011"}
    assert list(changes["Week"]) == [4, 5]
    assert list(changes["DeltaHrs"]) == [14, 14]
    assert (changes["BeforeHrs"] == 0).all()


def test_substitute_with_a_qualified_person_raises_no_warning(shocked):
    after = apply_substitute(shocked, "Q011", "R16", [4, 5])
    assert assignment_warnings(shocked, after).empty


def test_substitute_partial_hours(shocked):
    after = apply_substitute(shocked, "Q011", "R16", [4, 5], hours_per_week=7)
    assert gap_of(after, "Q011") == 14


def test_substitute_twice_into_the_same_week_adds_hours_instead_of_duplicating(shocked):
    once = apply_substitute(shocked, "Q011", "R16", [4], hours_per_week=7)
    twice = apply_substitute(once, "Q011", "R16", [4], hours_per_week=7)
    a = twice["Allocations"]
    mine = a[(a["ResourceID"] == "R16") & (a["RequirementID"] == "Q011") & (a["Week"] == 4)]
    assert len(mine) == 1 and mine.iloc[0]["PlannedHrs"] == 14
    assert validate_data(twice) == []          # no duplicate key, unique AllocationIDs


def test_substitute_default_skips_weeks_with_nothing_unmet(shocked):
    after = apply_substitute(shocked, "Q011", "R16", [3, 4])  # week 3 is fully covered already
    changes = allocation_changes(shocked, after)
    assert list(changes["Week"]) == [4]


def test_three_substitutions_leave_r16_over_capacity_and_say_so(data, shocked):
    # Week 5 for R16: 10h own work + 14 (Q011) + 12 (Q001) + 12 (Q035) = 48h against 40h -> 8h over.
    # Week 4: 14 + 12 + 12 = 38h, fine.
    after = apply_substitute(shocked, "Q011", "R16", [4, 5])
    after = apply_substitute(after, "Q035", "R16", [4, 5])
    after = apply_substitute(after, "Q001", "R16", [4, 5])   # R16 has cloud 3, Q001 asks for 4
    out = evaluate_response(data, shocked, after)
    assert metric(out["vs_shock"], "Unmet hours (staffing gap)") == -76
    assert metric(out["vs_shock"], "Over-allocated resources") == 1
    assert metric(out["vs_shock"], "Over-allocated hours") == 8
    assert metric(out["vs_baseline"], "Unmet hours (staffing gap)") == 0   # back to the original 288h
    w = out["warnings"]
    assert list(w.columns) == WARNING_COLUMNS
    below = w[w["Warning"] == "Below required proficiency"]
    assert list(below["RequirementID"]) == ["Q001"] and below.iloc[0]["Hours"] == 24
    over = w[w["Warning"] == "Over capacity"]
    assert len(over) == 1                                  # reported once for the person, not per requirement
    assert over.iloc[0]["ResourceID"] == "R16" and over.iloc[0]["Weeks"] == "5" and over.iloc[0]["Hours"] == 8
    assert validate_data(after) == []


def test_substitute_role_mismatch_is_a_warning_not_an_error(shocked):
    # R01 is a Software Engineer with cloud 2 (Q011 needs a Cloud/DevOps Engineer at level 3) and already
    # carries 34h in weeks 4-5, so +14h = 48h against 40h: 8h over in each of the two weeks = 16h
    after = apply_substitute(shocked, "Q011", "R01", [4, 5])
    w = assignment_warnings(shocked, after).set_index("Warning")
    assert set(w.index) == {"Below required proficiency", "Role mismatch", "Over capacity"}
    assert w.loc["Over capacity", "Hours"] == 16 and w.loc["Over capacity", "Weeks"] == "4, 5"
    assert gap_of(after, "Q011") == 0                       # still allowed: the planner may accept the trade-off


def test_substitute_errors(shocked, data):
    with pytest.raises(ValueError, match="Unknown requirement"):
        apply_substitute(shocked, "Q999", "R16", [4])
    with pytest.raises(ValueError, match="Unknown resource"):
        apply_substitute(shocked, "Q011", "R99", [4])
    with pytest.raises(ValueError, match="no Cloud Infrastructure skill"):
        apply_substitute(shocked, "Q011", "R21", [4])          # a UX designer
    with pytest.raises(ValueError, match="at least one week"):
        apply_substitute(shocked, "Q011", "R16", [])
    with pytest.raises(ValueError, match="outside"):
        apply_substitute(shocked, "Q011", "R16", [12])          # Q011 runs weeks 1-9
    with pytest.raises(ValueError, match="more than the 14h unmet"):
        apply_substitute(shocked, "Q011", "R16", [4], hours_per_week=15)
    with pytest.raises(ValueError, match="more than 0"):
        apply_substitute(shocked, "Q011", "R16", [4], hours_per_week=0)
    with pytest.raises(ValueError, match="no unmet hours"):
        apply_substitute(data, "Q001", "R16", [4])               # fully staffed in the original plan


def test_substitute_cannot_use_a_person_who_is_unavailable(shocked):
    # R15 is out in weeks 4-5, so cannot be the substitute there
    with pytest.raises(ValueError, match="unavailable in week"):
        apply_substitute(shocked, "Q011", "R15", [4, 5])


# =========================================================================
# Reallocate
# =========================================================================
def test_reallocate_moves_the_problem_it_does_not_solve_it(data):
    # R18 (Test Automation 5) holds 18h/week on Q016 (P03) in weeks 6-12. Move 8h/week to Q045 (P08) in weeks 6-11.
    after = apply_reallocate(data, "R18", "Q016", "Q045", range(6, 12), 8)
    assert gap_of(data, "Q045") == 56 and gap_of(after, "Q045") == 8      # 56 - 6 x 8
    assert gap_of(data, "Q016") == 0 and gap_of(after, "Q016") == 48      # 6 x 8 now unmet on P03
    result = compare_scenario(data, after)
    assert metric(result, "Unmet hours (staffing gap)") == 0               # total unchanged
    assert metric(result, "Allocated hours") == 0
    assert metric(result, "Requirements with gaps") == 1                   # Q016 joins Q045, Q020, Q043
    affected = result["affected_requirements"].set_index("RequirementID")
    assert list(affected.index) == ["Q016"] and affected.loc["Q016", "ProjectID"] == "P03"
    assert validate_data(after) == []


def test_reallocate_does_not_change_the_persons_total_load(data):
    after = apply_reallocate(data, "R18", "Q016", "Q045", range(6, 12), 8)
    for week in range(6, 12):
        assert load(after, "R18", week) == load(data, "R18", week)


def test_reallocate_changes_table_shows_donor_and_recipient(data):
    after = apply_reallocate(data, "R18", "Q016", "Q045", [6, 7], 8)
    ch = allocation_changes(data, after)
    donor = ch[ch["RequirementID"] == "Q016"]
    recipient = ch[ch["RequirementID"] == "Q045"]
    assert list(donor["DeltaHrs"]) == [-8, -8] and list(donor["AfterHrs"]) == [10, 10]
    assert list(recipient["DeltaHrs"]) == [8, 8] and recipient["ProjectID"].eq("P08").all()


def test_reallocate_removes_a_donor_row_that_reaches_zero(data):
    # double P08's effort so Q045 needs 16h/week; R20 (Test Automation 3 = the minimum) gives all 10h of Q033 in weeks 5-8
    big = apply_effort_increase(data, "P08", 100)
    after = apply_reallocate(big, "R20", "Q033", "Q045", [5, 6, 7, 8], 10)
    a = after["Allocations"]
    q033 = a[(a["ResourceID"] == "R20") & (a["RequirementID"] == "Q033")]
    assert list(q033["Week"]) == [4]                 # only week 4 is left
    assert (pd.to_numeric(a["PlannedHrs"]) > 0).all()
    assert validate_data(after) == []


def test_reallocate_into_a_below_minimum_person_is_a_warning(data):
    # R16 has Test Automation 3 (Q045 asks for 3, fine) -> use Q043 (data viz, min 5): R12 has 4, below the minimum
    after = apply_reallocate(data, "R12", next_req(data, "R12"), "Q043", [4], 8)
    w = assignment_warnings(data, after)
    assert "Below required proficiency" in set(w["Warning"])


def next_req(data, resource):
    a = data["Allocations"]
    return a[(a["ResourceID"] == resource) & (a["Week"] == 4)].iloc[0]["RequirementID"]


def test_reallocate_errors(data):
    with pytest.raises(ValueError, match="two different"):
        apply_reallocate(data, "R18", "Q016", "Q016", [6], 4)
    with pytest.raises(ValueError, match="Unknown requirement"):
        apply_reallocate(data, "R18", "Q999", "Q045", [6], 4)
    with pytest.raises(ValueError, match="Unknown resource"):
        apply_reallocate(data, "R99", "Q016", "Q045", [6], 4)
    with pytest.raises(ValueError, match="more than 0"):
        apply_reallocate(data, "R18", "Q016", "Q045", [6], 0)
    with pytest.raises(ValueError, match="outside"):
        apply_reallocate(data, "R18", "Q016", "Q045", [3], 4)          # Q045 starts in week 5
    with pytest.raises(ValueError, match="more than the 8h unmet"):
        apply_reallocate(data, "R18", "Q016", "Q045", [6], 9)          # only 8h/week is unmet on Q045
    with pytest.raises(ValueError, match="only 0h|has only"):
        apply_reallocate(data, "R19", "Q016", "Q045", [6], 4)          # R19 holds nothing on Q016
    with pytest.raises(ValueError, match="no Test Automation skill"):
        apply_reallocate(data, "R21", next_req(data, "R21"), "Q045", [5], 4)


def test_failed_reallocate_changes_nothing(data):
    before = data["Allocations"].copy(deep=True)
    with pytest.raises(ValueError):
        apply_reallocate(data, "R18", "Q016", "Q045", [6, 7], 9)
    pd.testing.assert_frame_equal(data["Allocations"], before)


def test_reallocate_cannot_use_an_unavailable_person(data):
    out = apply_resource_unavailable(data, "R18", [7])
    with pytest.raises(ValueError, match="unavailable in week"):
        apply_reallocate(out, "R18", "Q016", "Q045", [6, 7], 4)


# =========================================================================
# Adjust timeline
# =========================================================================
@pytest.fixture(scope="module")
def p01_squeezed(data):
    return apply_deadline_acceleration(data, "P01", 6)       # every P01 requirement now ends in week 6


def test_extension_stretches_windows_with_effort_constant(p01_squeezed):
    after = apply_timeline_extension(p01_squeezed, "P01", 8)
    req = after["Project_Requirements"].set_index("RequirementID")
    assert (req.loc["Q001", "EndWeek"], req.loc["Q001", "HoursPerWeek"]) == (8, pytest.approx(15))        # 120h / 8 wks
    assert req.loc["Q002", "HoursPerWeek"] == pytest.approx(128 / 7)                                       # weeks 2-8
    assert req.loc["Q003", "HoursPerWeek"] == pytest.approx(10)                                            # 60h / weeks 3-8
    assert req.loc["Q004", "HoursPerWeek"] == pytest.approx(12)                                            # 48h / weeks 5-8
    assert req.loc["Q005", "HoursPerWeek"] == pytest.approx(12.5)                                          # 100h / weeks 1-8
    assert after["Projects"].set_index("ProjectID").loc["P01", "DeadlineWeek"] == 8


def test_extension_alone_does_not_reduce_total_unmet_hours(data, p01_squeezed):
    # unmet = total effort - existing allocation, and neither changed: the extension buys ROOM, not coverage
    after = apply_timeline_extension(p01_squeezed, "P01", 8)
    result = compare_scenario(p01_squeezed, after)
    assert metric(result, "Unmet hours (staffing gap)") == 0
    assert metric(result, "Required hours") == 0
    assert metric(result, "Allocated hours") == 0
    assert gap_of(after, "Q001") == 48
    pd.testing.assert_frame_equal(after["Allocations"], p01_squeezed["Allocations"])


def test_extension_eases_the_cloud_capacity_shortage_hand_calculated(p01_squeezed):
    # Q001 unmet 48h. Only R15 qualifies; R15 is 38/40 in weeks 1-6 (2h free) and 26/40 in weeks 7-8 (14h free).
    # window 1-6: 48 - 6x2 = 36h short;  window 1-8: 48 - (6x2 + 2x14) = 8h short.
    after = apply_timeline_extension(p01_squeezed, "P01", 8)
    changes = compare_scenario(p01_squeezed, after)["conflict_changes"]
    row = changes[(changes["ConflictType"] == "Capacity shortage") & (changes["RequirementID"] == "Q001")].iloc[0]
    assert (row["BaselineHours"], row["ScenarioHours"], row["Status"]) == (36, 8, "Improved")
    # a longer window also holds more of R15's other commitments: 156h -> 156 + 2 x (14 + 12) = 208h
    comp = changes[(changes["ConflictType"] == "Competing requirements") & (changes["RequirementID"] == "Q001")].iloc[0]
    assert (comp["BaselineHours"], comp["ScenarioHours"], comp["Status"]) == (156, 208, "Worsened")


def test_extension_on_the_original_plan(data):
    after = apply_timeline_extension(data, "P04", 12)         # Q020-Q022 end in week 10 = P04's deadline
    req = after["Project_Requirements"].set_index("RequirementID")
    assert req.loc["Q020", "HoursPerWeek"] == pytest.approx(70 / 9)       # 10h x 7 weeks over weeks 4-12
    assert req.loc["Q022", "HoursPerWeek"] == pytest.approx(40 / 7)       # 8h x 5 weeks over weeks 6-12
    assert validate_data(after) == []


def test_extension_leaves_requirements_that_end_earlier_untouched(p01_squeezed):
    d = copy.deepcopy(p01_squeezed)
    d["Project_Requirements"].loc[d["Project_Requirements"]["RequirementID"] == "Q004", "EndWeek"] = 5
    after = apply_timeline_extension(d, "P01", 8)
    req = after["Project_Requirements"].set_index("RequirementID")
    assert req.loc["Q004", "EndWeek"] == 5
    assert req.loc["Q003", "EndWeek"] == 8


def test_extension_errors(data, p01_squeezed):
    with pytest.raises(ValueError, match="later"):
        apply_timeline_extension(data, "P01", 10)
    with pytest.raises(ValueError, match="later"):
        apply_timeline_extension(data, "P01", 8)
    with pytest.raises(ValueError, match="calendar"):
        apply_timeline_extension(data, "P03", 13)             # calendar ends at week 12
    with pytest.raises(ValueError, match="Unknown project"):
        apply_timeline_extension(data, "P99", 12)
    d = copy.deepcopy(data)
    p01 = d["Project_Requirements"]["ProjectID"] == "P01"
    d["Project_Requirements"].loc[p01, "EndWeek"] = 9         # nothing ends at the deadline (week 10)
    with pytest.raises(ValueError, match="No requirement"):
        apply_timeline_extension(d, "P01", 12)


# =========================================================================
# Cross-cutting
# =========================================================================
@pytest.mark.parametrize("respond", [
    lambda d: apply_substitute(d, "Q011", "R16", [4, 5]),
    lambda d: apply_reallocate(d, "R18", "Q016", "Q045", [6, 7], 8),
    lambda d: apply_timeline_extension(d, "P04", 12),
], ids=["substitute", "reallocate", "timeline"])
def test_responses_never_modify_their_input(shocked, respond):
    before = {k: v.copy(deep=True) for k, v in shocked.items()}
    respond(shocked)
    assert set(shocked) == set(before)
    for name, frame in before.items():
        pd.testing.assert_frame_equal(shocked[name], frame)


def test_response_keeps_the_unavailability_overlay(shocked):
    after = apply_substitute(shocked, "Q011", "R16", [4, 5])
    assert sorted(after["Unavailability"]["Week"]) == [4, 5]


def test_evaluate_response_returns_the_four_parts(data, shocked):
    out = evaluate_response(data, shocked, apply_substitute(shocked, "Q011", "R16", [4, 5]))
    assert set(out) == {"vs_shock", "vs_baseline", "allocation_changes", "warnings"}
    assert metric(out["vs_baseline"], "Unmet hours (staffing gap)") == 48     # 288 -> 336: Q001 + Q035 still open


def test_a_response_that_changes_nothing_changes_nothing(data):
    out = evaluate_response(data, data, copy.deepcopy(data))
    assert (out["vs_shock"]["metrics"]["Delta"] == 0).all()
    assert out["allocation_changes"].empty and out["warnings"].empty


def test_engine_never_recommends_or_scores(data, shocked):
    out = evaluate_response(data, shocked, apply_substitute(shocked, "Q011", "R16", [4, 5]))
    text = " ".join(
        " ".join(map(str, f.columns)) for f in [out["allocation_changes"], out["warnings"],
                                                out["vs_shock"]["metrics"], out["vs_shock"]["conflict_changes"]]
    ).lower()
    assert "recommend" not in text and "best" not in text and "score" not in text and "resilience" not in text

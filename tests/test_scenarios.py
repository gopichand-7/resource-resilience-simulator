"""Steps 19-22: snapshot, the three shocks, and baseline-vs-scenario comparison.

Expected numbers were calculated by hand from the raw Excel file, not from the engine.
IDs appear only where a test checks that a planted situation is still in the data.
"""
import copy

import pandas as pd
import pytest

from src.data_loader import load_data
from src.scenarios import (
    METRIC_COLUMNS,
    METRIC_NAMES,
    apply_deadline_acceleration,
    apply_effort_increase,
    apply_resource_unavailable,
    compare_scenario,
    take_snapshot,
)
from src.validator import validate_data


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def base(data):
    return take_snapshot(data)


def metric(result, name, column="Delta"):
    m = result["metrics"].set_index("Metric")
    return m.loc[name, column]


def gap(result, req):
    r = result["affected_requirements"].set_index("RequirementID")
    return r.loc[req, "ScenarioGapHrs"]


# ---------- Step 19: baseline snapshot ----------
def test_snapshot_baseline_numbers(base):
    m = base["metrics"]
    assert m["Required hours"] == 5430
    assert m["Allocated hours"] == 5142
    assert m["Unmet hours (staffing gap)"] == 288
    assert m["Requirements with gaps"] == 3
    assert m["Projects with gaps"] == 2
    assert m["Over-allocated resources"] == 0
    assert m["Capacity-shortage requirements"] == 2
    assert m["Capacity-shortage hours"] == 122  # Q020 40h + Q043 82h
    assert m["Downstream projects exposed"] == 1  # P04 -> P03


def test_snapshot_has_every_part(base):
    assert set(base) == {"capacity", "weekly_demand", "demand", "conflicts", "scarcity", "metrics"}
    assert list(base["metrics"]) == METRIC_NAMES


# ---------- a zero-change scenario equals the baseline ----------
def test_no_change_scenario_changes_nothing(data, base):
    same = copy.deepcopy(data)
    result = compare_scenario(data, same, base)
    assert (result["metrics"]["Delta"] == 0).all()
    assert result["conflict_changes"].empty
    assert result["affected_requirements"].empty
    assert result["affected_projects"].empty
    assert result["skill_shortages"].empty
    assert result["exposed_projects"].empty


def test_zero_percent_effort_increase_changes_nothing(data, base):
    result = compare_scenario(data, apply_effort_increase(data, "P04", 0), base)
    assert (result["metrics"]["Delta"] == 0).all()


def test_comparison_table_shape(data, base):
    result = compare_scenario(data, apply_effort_increase(data, "P04", 20), base)
    assert list(result["metrics"].columns) == METRIC_COLUMNS
    assert set(result) == {"metrics", "conflict_changes", "affected_requirements", "affected_projects",
                           "skill_shortages", "exposed_projects"}


# ---------- baseline is never mutated; scenario data stays valid ----------
@pytest.mark.parametrize("shock", [
    lambda d: apply_resource_unavailable(d, "R15", [4, 5]),
    lambda d: apply_deadline_acceleration(d, "P01", 6),
    lambda d: apply_effort_increase(d, "P04", 20),
])
def test_shocks_never_touch_the_baseline_and_stay_valid(data, shock):
    before = {k: v.copy(deep=True) for k, v in data.items()}
    scenario = shock(data)
    assert set(data) == set(before)  # no table added to the baseline
    for name, frame in before.items():
        pd.testing.assert_frame_equal(data[name], frame)
    assert validate_data(scenario) == []


# ---------- Shock 3: effort +20% on P04 ----------
def test_effort_increase_p04_hand_calculated(data, base):
    r = compare_scenario(data, apply_effort_increase(data, "P04", 20), base)
    assert gap(r, "Q020") == pytest.approx(84)  # 70 -> 84
    assert gap(r, "Q021") == pytest.approx(14)  # 0 -> 14
    assert gap(r, "Q022") == pytest.approx(8)   # 0 -> 8
    assert metric(r, "Required hours") == pytest.approx(36)    # 180 -> 216
    assert metric(r, "Allocated hours") == 0                   # allocations unchanged
    assert metric(r, "Unmet hours (staffing gap)") == pytest.approx(36)
    p = r["affected_projects"].set_index("ProjectID")
    assert list(p.index) == ["P04"]
    assert p.loc["P04", "BaselineGapHrs"] == pytest.approx(70)
    assert p.loc["P04", "ScenarioGapHrs"] == pytest.approx(106)


def test_effort_increase_exposes_p03_downstream(data, base):
    r = compare_scenario(data, apply_effort_increase(data, "P04", 20), base)
    assert list(r["exposed_projects"]["ProjectID"]) == ["P03"]
    assert r["exposed_projects"].iloc[0]["Path"] == "P04 -> P03"


def test_effort_increase_only_touches_the_chosen_project(data):
    s = apply_effort_increase(data, "P04", 20)
    other = data["Project_Requirements"]["ProjectID"] != "P04"
    pd.testing.assert_series_equal(
        s["Project_Requirements"].loc[other, "HoursPerWeek"].astype(float),
        data["Project_Requirements"].loc[other, "HoursPerWeek"].astype(float))
    pd.testing.assert_frame_equal(s["Allocations"], data["Allocations"])


def test_effort_increase_worsens_the_cloud_shortage(data, base):
    r = compare_scenario(data, apply_effort_increase(data, "P04", 20), base)
    row = r["conflict_changes"].query("ConflictType == 'Capacity shortage'").iloc[0]
    assert (row["RequirementID"], row["Status"]) == ("Q020", "Worsened")
    assert row["BaselineHours"] == pytest.approx(40)
    assert row["ScenarioHours"] == pytest.approx(50)
    assert r["skill_shortages"].iloc[0]["Skill"] == "Cloud Infrastructure"


# ---------- Shock 2: P01 deadline 10 -> 6 ----------
def test_deadline_acceleration_p01_hand_calculated(data, base):
    r = compare_scenario(data, apply_deadline_acceleration(data, "P01", 6), base)
    expected = {"Q001": 48, "Q002": 48, "Q003": 20, "Q004": 32, "Q005": 40}
    for req, hours in expected.items():
        assert gap(r, req) == pytest.approx(hours), req
    p = r["affected_projects"].set_index("ProjectID")
    assert p.loc["P01", "ScenarioGapHrs"] == pytest.approx(188)
    assert metric(r, "Required hours") == 0              # total effort is constant
    assert metric(r, "Allocated hours") == -188          # only post-deadline allocations are removed
    assert metric(r, "Unmet hours (staffing gap)") == pytest.approx(188)  # 288 -> 476


def test_deadline_acceleration_new_weekly_demand(data):
    s = apply_deadline_acceleration(data, "P01", 6)
    req = s["Project_Requirements"].set_index("RequirementID")
    # total effort stays constant: weekly demand x number of weeks
    old = data["Project_Requirements"].set_index("RequirementID")
    for rid in ["Q001", "Q002", "Q003", "Q004", "Q005"]:
        old_total = old.loc[rid, "HoursPerWeek"] * (old.loc[rid, "EndWeek"] - old.loc[rid, "StartWeek"] + 1)
        new_total = req.loc[rid, "HoursPerWeek"] * (req.loc[rid, "EndWeek"] - req.loc[rid, "StartWeek"] + 1)
        assert new_total == pytest.approx(old_total), rid
        assert req.loc[rid, "EndWeek"] <= 6
    assert req.loc["Q001", "HoursPerWeek"] == pytest.approx(20)
    assert req.loc["Q002", "HoursPerWeek"] == pytest.approx(25.6)
    assert req.loc["Q003", "HoursPerWeek"] == pytest.approx(15)
    assert req.loc["Q004", "HoursPerWeek"] == pytest.approx(24)
    assert req.loc["Q005", "HoursPerWeek"] == pytest.approx(100 / 6)


def test_deadline_acceleration_removes_late_allocations_and_does_not_redistribute(data):
    s = apply_deadline_acceleration(data, "P01", 6)
    a_old, a_new = data["Allocations"], s["Allocations"]
    p01_new = a_new[a_new["ProjectID"] == "P01"]
    assert (p01_new["Week"] <= 6).all()
    # allocations in the kept weeks are exactly the old ones (nothing moved or added)
    kept = a_old[(a_old["ProjectID"] == "P01") & (a_old["Week"] <= 6)]
    assert p01_new["PlannedHrs"].sum() == kept["PlannedHrs"].sum()
    pd.testing.assert_frame_equal(
        a_new[a_new["ProjectID"] != "P01"].reset_index(drop=True),
        a_old[a_old["ProjectID"] != "P01"].reset_index(drop=True))
    assert s["Projects"].set_index("ProjectID").loc["P01", "DeadlineWeek"] == 6


def test_deadline_acceleration_exposes_p05(data, base):
    r = compare_scenario(data, apply_deadline_acceleration(data, "P01", 6), base)
    assert "P05" in set(r["exposed_projects"]["ProjectID"])
    assert metric(r, "Downstream projects exposed") == 1  # P03 (already) + P05


def test_deadline_must_be_earlier_and_not_before_start(data):
    with pytest.raises(ValueError, match="earlier"):
        apply_deadline_acceleration(data, "P01", 10)
    with pytest.raises(ValueError, match="earlier"):
        apply_deadline_acceleration(data, "P01", 12)
    start = int(data["Projects"].set_index("ProjectID").loc["P01", "StartWeek"])
    with pytest.raises(ValueError, match="before"):
        apply_deadline_acceleration(data, "P01", start - 1)
    with pytest.raises(ValueError, match="Unknown project"):
        apply_deadline_acceleration(data, "P99", 5)


def test_requirement_starting_after_the_new_deadline_collapses_into_the_deadline_week():
    data = _mini()
    s = apply_deadline_acceleration(data, "PX", 3)
    req = s["Project_Requirements"].set_index("RequirementID")
    assert (req.loc["LATE", "StartWeek"], req.loc["LATE", "EndWeek"]) == (3, 3)
    assert req.loc["LATE", "HoursPerWeek"] == pytest.approx(40)  # 10h x 4 weeks into one week
    assert (req.loc["EARLY", "StartWeek"], req.loc["EARLY", "EndWeek"]) == (1, 3)
    assert req.loc["EARLY", "HoursPerWeek"] == pytest.approx(20)  # 10h x 6 weeks / 3


def _mini():
    return {
        "Resources": pd.DataFrame({"ResourceID": ["A"], "WeeklyCapacityHrs": [40]}),
        "Calendar": pd.DataFrame({"Week": list(range(1, 9))}),
        "Projects": pd.DataFrame({"ProjectID": ["PX"], "StartWeek": [1], "DeadlineWeek": [8]}),
        "Project_Requirements": pd.DataFrame({
            "RequirementID": ["EARLY", "LATE"], "ProjectID": ["PX", "PX"],
            "StartWeek": [1, 5], "EndWeek": [6, 8], "HoursPerWeek": [10, 10]}),
        "Allocations": pd.DataFrame({"ResourceID": ["A"], "ProjectID": ["PX"], "RequirementID": ["EARLY"],
                                     "Week": [5], "PlannedHrs": [10]}),
        "Dependencies": pd.DataFrame({"PredecessorID": [], "SuccessorID": []}),
    }


# ---------- Shock 1: R15 unavailable weeks 4-5 ----------
def test_resource_unavailable_hand_calculated(data, base):
    r = compare_scenario(data, apply_resource_unavailable(data, "R15", [4, 5]), base)
    assert metric(r, "Allocated hours") == -76  # 2 weeks x (12 + 14 + 12)
    assert metric(r, "Unmet hours (staffing gap)") == 76  # 288 -> 364
    assert metric(r, "Required hours") == 0
    gaps = r["affected_requirements"].set_index("RequirementID")["ScenarioGapHrs"]
    assert gaps.to_dict() == {"Q001": pytest.approx(24), "Q011": pytest.approx(28), "Q035": pytest.approx(24)}
    assert set(r["affected_projects"]["ProjectID"]) == {"P01", "P02", "P06"}


def test_resource_unavailable_exposes_p05_and_p09(data, base):
    r = compare_scenario(data, apply_resource_unavailable(data, "R15", [4, 5]), base)
    assert set(r["exposed_projects"]["ProjectID"]) == {"P05", "P09"}
    assert not set(r["exposed_projects"]["ProjectID"]) & set(r["affected_projects"]["ProjectID"])


def test_resource_unavailable_capacity_is_zero_in_those_weeks_only(data):
    from src.capacity import build_capacity_table
    t = build_capacity_table(apply_resource_unavailable(data, "R15", [4, 5]))
    r15 = t[t["ResourceID"] == "R15"].set_index("Week")
    assert r15.loc[4, "CapacityHrs"] == 0 and r15.loc[5, "CapacityHrs"] == 0
    assert r15.loc[6, "CapacityHrs"] == 40
    assert r15.loc[4, "PlannedHrs"] == 0  # planned hours removed with it


def test_resource_unavailable_conflict_changes(data, base):
    r = compare_scenario(data, apply_resource_unavailable(data, "R15", [4, 5]), base)
    ch = r["conflict_changes"]
    q020 = ch[(ch["ConflictType"] == "Capacity shortage") & (ch["RequirementID"] == "Q020")].iloc[0]
    assert (q020["BaselineHours"], q020["ScenarioHours"], q020["Status"]) == (40, 44, "Worsened")
    q001 = ch[(ch["ConflictType"] == "Capacity shortage") & (ch["RequirementID"] == "Q001")].iloc[0]
    assert (q001["ScenarioHours"], q001["Status"]) == (24, "New")


def test_resource_unavailable_stacks_when_applied_twice(data):
    once = apply_resource_unavailable(data, "R15", [4])
    twice = apply_resource_unavailable(once, "R15", [5, 4])
    assert sorted(twice["Unavailability"]["Week"]) == [4, 5]  # no duplicate for week 4


def test_resource_unavailable_errors(data):
    with pytest.raises(ValueError, match="Unknown resource"):
        apply_resource_unavailable(data, "R99", [4])
    with pytest.raises(ValueError, match="at least one week"):
        apply_resource_unavailable(data, "R15", [])
    with pytest.raises(ValueError, match="calendar"):
        apply_resource_unavailable(data, "R15", [4, 99])


def test_unavailable_resource_with_no_work_in_those_weeks_changes_no_allocation(data, base):
    # R25 has no allocations in week 12 (its requirement ends in week 8)
    r = compare_scenario(data, apply_resource_unavailable(data, "R25", [12]), base)
    assert metric(r, "Allocated hours") == 0
    assert r["affected_requirements"].empty


# ---------- error cases for effort ----------
def test_effort_increase_errors(data):
    with pytest.raises(ValueError, match="0%"):
        apply_effort_increase(data, "P04", -5)
    with pytest.raises(ValueError, match="Unknown project"):
        apply_effort_increase(data, "P99", 10)


# ---------- the comparison does not invent a recommendation ----------
def test_comparison_contains_no_score_or_recommendation(data, base):
    r = compare_scenario(data, apply_effort_increase(data, "P04", 20), base)
    for frame in r.values():
        text = " ".join(map(str, frame.columns)).lower()
        assert "recommend" not in text and "resilience" not in text and "score" not in text


# ---------- an affected project is not also reported as merely "exposed" ----------
def test_project_that_is_itself_affected_is_not_listed_as_exposed(data, base):
    # P04 blocks P03. Give BOTH more work: P03 is affected in its own right, so it is not "exposed".
    scenario = apply_effort_increase(apply_effort_increase(data, "P04", 20), "P03", 20)
    result = compare_scenario(data, scenario, base)
    assert {"P03", "P04"} <= set(result["affected_projects"]["ProjectID"])
    assert "P03" not in set(result["exposed_projects"]["ProjectID"])

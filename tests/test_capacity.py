import copy
import math

import pandas as pd
import pytest

from src.capacity import (
    CAPACITY_COLUMNS,
    DEMAND_COLUMNS,
    DEMAND_WEEK_COLUMNS,
    build_capacity_table,
    build_requirement_demand,
    build_requirement_week_table,
)
from src.data_loader import load_data


@pytest.fixture(scope="module")
def table():
    return build_capacity_table(load_data())


def row(table, resource, week):
    return table[(table["ResourceID"] == resource) & (table["Week"] == week)].iloc[0]


def tiny(capacity, allocations, weeks=(1, 2)):
    """Minimal hand-made dataset: one resource 'X' with the given weekly capacity."""
    return {
        "Resources": pd.DataFrame({"ResourceID": ["X"], "WeeklyCapacityHrs": [capacity]}),
        "Calendar": pd.DataFrame({"Week": list(weeks)}),
        "Allocations": pd.DataFrame(allocations, columns=["ResourceID", "Week", "PlannedHrs"]),
    }


# ---------- shape ----------
def test_one_row_per_resource_per_week(table):
    assert len(table) == 28 * 12
    assert not table.duplicated(["ResourceID", "Week"]).any()
    assert list(table.columns) == CAPACITY_COLUMNS


def test_total_planned_hours_are_conserved(table):
    assert table["PlannedHrs"].sum() == load_data()["Allocations"]["PlannedHrs"].sum()


# ---------- hand-calculated values from the V2 dataset ----------
def test_critical_resource_week_matches_the_spec_example(table):
    # R15 in week 4: 12h (P01) + 14h (P02) + 12h (P06) = 38h of 40h
    r = row(table, "R15", 4)
    assert r["CapacityHrs"] == 40
    assert r["PlannedHrs"] == 38
    assert r["AvailableHrs"] == 2
    assert r["Utilization"] == pytest.approx(0.95)
    assert r["LoadHeadroom"] == pytest.approx(0.05)


def test_week_with_no_allocation_is_fully_available(table):
    # R25 only works on a requirement that ends in week 8
    r = row(table, "R25", 9)
    assert r["PlannedHrs"] == 0
    assert r["AvailableHrs"] == 40
    assert r["Utilization"] == 0
    assert r["LoadHeadroom"] == 1


def test_part_time_capacity_is_respected(table):
    # R26 has 20h capacity and 8h planned in week 4
    r = row(table, "R26", 4)
    assert r["CapacityHrs"] == 20
    assert r["AvailableHrs"] == 12
    assert r["Utilization"] == pytest.approx(0.40)


def test_planned_hours_vary_by_week_for_the_same_resource(table):
    # R15: three requirements in week 4, two in week 9 (P06 requirement ended in week 8)
    assert row(table, "R15", 9)["PlannedHrs"] == 26
    assert row(table, "R15", 10)["PlannedHrs"] == 12


def test_baseline_plan_has_no_negative_available_hours(table):
    assert (table["AvailableHrs"] >= 0).all()


# ---------- edge cases (hand-made data) ----------
def test_over_allocation_gives_negative_available_and_zero_headroom():
    t = build_capacity_table(tiny(40, [("X", 1, 30), ("X", 1, 20)]))
    r = t[t["Week"] == 1].iloc[0]
    assert r["PlannedHrs"] == 50
    assert r["AvailableHrs"] == -10
    assert r["Utilization"] == pytest.approx(1.25)
    assert r["LoadHeadroom"] == 0


def test_exactly_full_week():
    r = build_capacity_table(tiny(40, [("X", 1, 40)])).iloc[0]
    assert r["AvailableHrs"] == 0
    assert r["Utilization"] == 1
    assert r["LoadHeadroom"] == 0


def test_zero_capacity_resource_has_no_utilization_and_zero_headroom():
    t = build_capacity_table(tiny(0, [("X", 1, 5)]))
    r = t[t["Week"] == 1].iloc[0]
    assert math.isnan(r["Utilization"])
    assert r["LoadHeadroom"] == 0
    assert r["AvailableHrs"] == -5


def test_resource_with_no_allocations_at_all():
    t = build_capacity_table(tiny(40, []))
    assert len(t) == 2
    assert (t["PlannedHrs"] == 0).all()
    assert (t["LoadHeadroom"] == 1).all()


def test_numeric_values_stored_as_text_are_converted():
    t = build_capacity_table(tiny("40", [("X", "1", "10")]))
    r = t[t["Week"] == 1].iloc[0]
    assert r["CapacityHrs"] == 40
    assert r["PlannedHrs"] == 10


# ---------- purity ----------
def test_input_data_is_not_modified():
    data = load_data()
    before = copy.deepcopy(data)
    build_capacity_table(data)
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])


# =====================================================================
# Step 8: requirement demand and staffing gap
# =====================================================================
@pytest.fixture(scope="module")
def demand():
    return build_requirement_demand(load_data())


@pytest.fixture(scope="module")
def weekly_demand():
    return build_requirement_week_table(load_data())


def req_row(demand, rid):
    return demand[demand["RequirementID"] == rid].iloc[0]


def tiny_demand(requirements, allocations):
    return {
        "Project_Requirements": pd.DataFrame(
            requirements, columns=["RequirementID", "ProjectID", "HoursPerWeek", "StartWeek", "EndWeek"]
        ),
        "Allocations": pd.DataFrame(allocations, columns=["RequirementID", "Week", "PlannedHrs"]),
    }


def test_demand_has_one_row_per_requirement(demand):
    assert len(demand) == 50
    assert demand["RequirementID"].is_unique
    assert list(demand.columns) == DEMAND_COLUMNS


def test_demand_totals_match_hand_calculation(demand):
    assert demand["RequiredHrs"].sum() == 5430
    assert demand["AllocatedHrs"].sum() == 5142
    assert demand["GapHrs"].sum() == 288


def test_fully_open_requirement_p04_matches_spec_example(demand):
    # Q020: 10h/week for weeks 4-10 (7 weeks), nobody assigned
    r = req_row(demand, "Q020")
    assert r["ProjectID"] == "P04"
    assert r["RequiredHrs"] == 70
    assert r["AllocatedHrs"] == 0
    assert r["GapHrs"] == 70
    assert r["Coverage"] == 0


def test_fully_staffed_requirement_has_no_gap(demand):
    r = req_row(demand, "Q001")  # 12h x 10 weeks, fully allocated
    assert r["RequiredHrs"] == 120
    assert r["GapHrs"] == 0
    assert r["Coverage"] == 1


def test_split_staffing_is_summed_across_resources(demand):
    r = req_row(demand, "Q007")  # 36h/week split between two people, weeks 2-9
    assert r["RequiredHrs"] == 288
    assert r["AllocatedHrs"] == 288
    assert r["GapHrs"] == 0


def test_only_the_three_open_requirements_have_gaps(demand):
    gaps = demand[demand["GapHrs"] > 0].set_index("RequirementID")["GapHrs"].to_dict()
    assert gaps == {"Q020": 70, "Q043": 162, "Q045": 56}


def test_project_level_coverage_from_requirement_hours(demand):
    by_project = demand.groupby("ProjectID")[["RequiredHrs", "AllocatedHrs"]].sum()
    coverage = by_project["AllocatedHrs"] / by_project["RequiredHrs"]
    assert coverage["P08"] == pytest.approx(264 / 482)  # 264h of 482h, from the raw rows
    assert coverage["P04"] == pytest.approx(110 / 180)  # 110h of 180h, from the raw rows
    assert coverage.drop(["P04", "P08"]).eq(1).all()


def test_weekly_table_rows_equal_requirement_window_weeks(weekly_demand):
    assert len(weekly_demand) == load_data()["Project_Requirements"].eval(
        "EndWeek - StartWeek + 1"
    ).sum()
    assert list(weekly_demand.columns) == DEMAND_WEEK_COLUMNS
    assert not weekly_demand.duplicated(["RequirementID", "Week"]).any()


def test_weekly_table_totals_match_summary(weekly_demand, demand):
    assert weekly_demand["RequiredHrs"].sum() == demand["RequiredHrs"].sum()
    assert weekly_demand["AllocatedHrs"].sum() == demand["AllocatedHrs"].sum()


def test_weekly_gap_for_open_requirement(weekly_demand):
    q = weekly_demand[weekly_demand["RequirementID"] == "Q020"]
    assert list(q["Week"]) == [4, 5, 6, 7, 8, 9, 10]
    assert (q["GapHrs"] == 10).all()


def test_partial_staffing_gives_partial_coverage():
    d = tiny_demand([("Q1", "P1", 10, 1, 2)], [("Q1", 1, 10), ("Q1", 2, 5)])
    r = build_requirement_demand(d).iloc[0]
    assert r["RequiredHrs"] == 20
    assert r["AllocatedHrs"] == 15
    assert r["GapHrs"] == 5
    assert r["Coverage"] == pytest.approx(0.75)


def test_gap_is_never_negative_when_over_staffed():
    d = tiny_demand([("Q1", "P1", 10, 1, 1)], [("Q1", 1, 6), ("Q1", 1, 8)])
    r = build_requirement_demand(d).iloc[0]
    assert r["AllocatedHrs"] == 14
    assert r["GapHrs"] == 0
    assert r["Coverage"] == pytest.approx(1.4)


def test_over_staffed_week_does_not_hide_a_gap_in_another_week():
    # week 1 over-staffed by 4h, week 2 short by 10h: the gap must stay 10h, not net to 6h
    d = tiny_demand([("Q1", "P1", 10, 1, 2)], [("Q1", 1, 14)])
    r = build_requirement_demand(d).iloc[0]
    assert r["GapHrs"] == 10


def test_zero_hour_requirement_has_no_coverage_value():
    d = tiny_demand([("Q1", "P1", 0, 1, 2)], [])
    r = build_requirement_demand(d).iloc[0]
    assert r["RequiredHrs"] == 0
    assert r["GapHrs"] == 0
    assert math.isnan(r["Coverage"])


def test_requirement_with_no_allocation_rows_at_all():
    d = tiny_demand([("Q1", "P1", 8, 2, 4)], [])
    r = build_requirement_demand(d).iloc[0]
    assert (r["RequiredHrs"], r["AllocatedHrs"], r["GapHrs"], r["Coverage"]) == (24, 0, 24, 0)


def test_demand_does_not_modify_input():
    data = load_data()
    before = copy.deepcopy(data)
    build_requirement_demand(data)
    build_requirement_week_table(data)
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])


# ---------- Unavailability overlay (scenario shock) ----------
def test_unavailability_sets_capacity_to_zero_only_for_those_weeks():
    d = tiny(40, [["X", 1, 10]], weeks=(1, 2, 3))
    d["Unavailability"] = pd.DataFrame({"ResourceID": ["X"], "Week": [2]})
    t = build_capacity_table(d).set_index("Week")
    assert t.loc[1, "CapacityHrs"] == 40
    assert t.loc[2, "CapacityHrs"] == 0
    assert t.loc[3, "CapacityHrs"] == 40
    assert t.loc[2, "AvailableHrs"] == 0
    assert math.isnan(t.loc[2, "Utilization"])  # no capacity: utilization is undefined, not infinite


def test_unavailability_does_not_touch_other_resources():
    d = tiny(40, [["X", 1, 10]], weeks=(1, 2))
    d["Resources"] = pd.DataFrame({"ResourceID": ["X", "Y"], "WeeklyCapacityHrs": [40, 30]})
    d["Unavailability"] = pd.DataFrame({"ResourceID": ["X"], "Week": [1]})
    t = build_capacity_table(d)
    assert row(t, "Y", 1)["CapacityHrs"] == 30
    assert row(t, "X", 1)["CapacityHrs"] == 0


def test_empty_or_missing_unavailability_changes_nothing():
    d = tiny(40, [["X", 1, 10]])
    plain = build_capacity_table(d)
    d["Unavailability"] = pd.DataFrame({"ResourceID": [], "Week": []})
    pd.testing.assert_frame_equal(build_capacity_table(d), plain)


def test_unavailability_overlay_does_not_modify_the_input():
    d = tiny(40, [["X", 1, 10]])
    d["Unavailability"] = pd.DataFrame({"ResourceID": ["X"], "Week": [1]})
    before = d["Resources"].copy()
    build_capacity_table(d)
    pd.testing.assert_frame_equal(d["Resources"], before)

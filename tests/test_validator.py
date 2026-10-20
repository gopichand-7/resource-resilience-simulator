import copy

import pandas as pd
import pytest

from src.data_loader import load_data
from src.validator import validate_data


@pytest.fixture
def data():
    return load_data()


def set_value(data, table, row, column, value):
    """Set one cell. The column is cast to object first so any value type can be injected."""
    df = data[table]
    df[column] = df[column].astype(object)
    df.loc[row, column] = value


def has_error(errors, *fragments):
    return any(all(f in e for f in fragments) for e in errors)


def add_dependency(data, pred, succ, dep_id):
    row = pd.DataFrame([{"DependencyID": dep_id, "PredecessorID": pred, "SuccessorID": succ, "Type": "Blocks"}])
    data["Dependencies"] = pd.concat([data["Dependencies"], row], ignore_index=True)


# 1. valid dataset
def test_valid_v2_dataset_has_no_errors(data):
    assert validate_data(data) == []


# 2. required columns / tables
def test_missing_required_column(data):
    data["Resource_Skills"] = data["Resource_Skills"].drop(columns=["Skill"])
    assert has_error(validate_data(data), "Resource_Skills", "missing required column", "Skill")


def test_missing_table(data):
    del data["Calendar"]
    assert has_error(validate_data(data), "Missing table", "Calendar")


# 3. duplicate IDs
def test_duplicate_resource_id(data):
    set_value(data, "Resources", 1, "ResourceID", "R01")
    assert has_error(validate_data(data), "Resources", "duplicate ResourceID R01")


def test_duplicate_composite_key(data):
    data["Resource_Skills"] = pd.concat(
        [data["Resource_Skills"], data["Resource_Skills"].iloc[[0]]], ignore_index=True
    )
    assert has_error(validate_data(data), "Resource_Skills", "duplicate", "Skill")


# 4. missing values
def test_missing_required_value(data):
    set_value(data, "Projects", 0, "ProjectName", None)
    assert has_error(validate_data(data), "Projects row 2", "'ProjectName' is blank")


def test_blank_string_counts_as_missing(data):
    set_value(data, "Resources", 3, "Role", "   ")
    assert has_error(validate_data(data), "Resources row 5", "'Role' is blank")


def test_description_is_optional(data):
    set_value(data, "Projects", 0, "Description", None)
    assert validate_data(data) == []


# 5. proficiency ranges
@pytest.mark.parametrize("bad", [0, 6, 3.5])
def test_invalid_proficiency(data, bad):
    set_value(data, "Resource_Skills", 0, "Proficiency", bad)
    assert has_error(validate_data(data), "Resource_Skills row 2", "'Proficiency'")


def test_invalid_min_proficiency(data):
    set_value(data, "Project_Requirements", 0, "MinProficiency", 9)
    assert has_error(validate_data(data), "Project_Requirements", "'MinProficiency'")


# 6. negative / non-numeric hours
def test_negative_planned_hours(data):
    set_value(data, "Allocations", 0, "PlannedHrs", -5)
    assert has_error(validate_data(data), "Allocations row 2", "'PlannedHrs' -5")


def test_negative_capacity_and_required_hours(data):
    set_value(data, "Resources", 0, "WeeklyCapacityHrs", -1)
    set_value(data, "Project_Requirements", 0, "HoursPerWeek", -2)
    errors = validate_data(data)
    assert has_error(errors, "'WeeklyCapacityHrs' -1")
    assert has_error(errors, "'HoursPerWeek' -2")


def test_non_numeric_hours(data):
    set_value(data, "Allocations", 0, "PlannedHrs", "ten")
    assert has_error(validate_data(data), "'PlannedHrs'", "not a number")


# 7. week validity
def test_week_outside_calendar(data):
    set_value(data, "Allocations", 0, "Week", 15)
    assert has_error(validate_data(data), "Allocations row 2", "'Week' 15 is not a week in Calendar")


def test_project_deadline_outside_calendar(data):
    set_value(data, "Projects", 0, "DeadlineWeek", 20)
    assert has_error(validate_data(data), "Projects row 2", "'DeadlineWeek' 20", "Calendar")


# 8. foreign keys
@pytest.mark.parametrize(
    "table, column, bad, fragment",
    [
        ("Allocations", "ResourceID", "R99", "Resources.ResourceID"),
        ("Allocations", "RequirementID", "Q999", "Project_Requirements.RequirementID"),
        ("Resource_Skills", "ResourceID", "R99", "Resources.ResourceID"),
        ("Project_Requirements", "ProjectID", "P99", "Projects.ProjectID"),
        ("Dependencies", "SuccessorID", "P99", "Projects.ProjectID"),
        ("Dependencies", "PredecessorID", "P99", "Projects.ProjectID"),
    ],
)
def test_invalid_foreign_key(data, table, column, bad, fragment):
    set_value(data, table, 0, column, bad)
    assert has_error(validate_data(data), table, column, bad, fragment)


# 9. periods
def test_invalid_requirement_period(data):
    set_value(data, "Project_Requirements", 0, "StartWeek", 9)
    set_value(data, "Project_Requirements", 0, "EndWeek", 3)
    assert has_error(validate_data(data), "Project_Requirements row 2", "StartWeek 9 is after EndWeek 3")


def test_invalid_project_period(data):
    set_value(data, "Projects", 0, "StartWeek", 11)
    assert has_error(validate_data(data), "Projects row 2", "StartWeek 11 is after DeadlineWeek")


# 10. allocation integrity
def test_allocation_project_must_match_requirement(data):
    set_value(data, "Allocations", 0, "ProjectID", "P09")  # first allocation row belongs to a P01 requirement
    assert has_error(validate_data(data), "Allocations row 2", "does not match requirement")


def test_allocation_week_must_fall_inside_requirement_window(data):
    row = data["Allocations"].index[data["Allocations"]["RequirementID"] == "Q001"][0]
    set_value(data, "Allocations", row, "Week", 12)  # valid calendar week, but Q001 runs weeks 1-10
    errors = validate_data(data)
    assert has_error(errors, "Week 12 is outside requirement Q001")
    assert not has_error(errors, "not a week in Calendar")


# 11. dependency cycles
def test_dependency_cycle_detected(data):
    add_dependency(data, "P05", "P01", "D05")  # P01 -> P05 already exists
    assert has_error(validate_data(data), "cycle detected", "P01", "P05")


def test_three_step_cycle_detected(data):
    add_dependency(data, "P05", "P07", "D05")
    add_dependency(data, "P07", "P01", "D06")  # P01 -> P05 -> P07 -> P01
    assert has_error(validate_data(data), "cycle detected", "P01 -> P05 -> P07 -> P01")


def test_self_dependency_is_a_cycle(data):
    add_dependency(data, "P03", "P03", "D05")
    assert has_error(validate_data(data), "cycle detected", "P03 -> P03")


def test_long_chain_without_cycle_is_valid(data):
    add_dependency(data, "P05", "P07", "D05")  # P01 -> P05 -> P07
    add_dependency(data, "P07", "P08", "D06")  # ... -> P08
    assert validate_data(data) == []


# 12. multiple errors in one run
def test_multiple_errors_are_all_returned(data):
    set_value(data, "Projects", 0, "ProjectName", None)         # missing value
    set_value(data, "Resource_Skills", 0, "Proficiency", 7)     # invalid proficiency
    set_value(data, "Allocations", 1, "ResourceID", "R99")      # unknown resource
    set_value(data, "Allocations", 2, "Week", 15)               # invalid week
    add_dependency(data, "P05", "P01", "D05")                   # dependency cycle
    errors = validate_data(data)
    assert len(errors) >= 5
    assert has_error(errors, "'ProjectName' is blank")
    assert has_error(errors, "'Proficiency' 7")
    assert has_error(errors, "R99")
    assert has_error(errors, "'Week' 15")
    assert has_error(errors, "cycle detected")


# purity and robustness
def test_validator_does_not_modify_input(data):
    set_value(data, "Allocations", 0, "PlannedHrs", -5)
    before = copy.deepcopy(data)
    validate_data(data)
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])


def test_missing_calendar_week_column_does_not_crash(data):
    data["Calendar"] = data["Calendar"].drop(columns=["Week"])
    errors = validate_data(data)  # week checks are skipped, nothing raises
    assert has_error(errors, "Calendar", "missing required column", "Week")


# ---- review follow-ups ----
@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "table, column",
    [("Allocations", "PlannedHrs"), ("Resources", "WeeklyCapacityHrs"), ("Project_Requirements", "HoursPerWeek")],
)
def test_infinite_hours_are_rejected(data, table, column, bad):
    set_value(data, table, 0, column, bad)
    assert has_error(validate_data(data), table, f"'{column}'")


def test_infinite_planned_hours_message(data):
    set_value(data, "Allocations", 0, "PlannedHrs", float("inf"))
    assert has_error(validate_data(data), "Allocations row 2", "'PlannedHrs'", "finite")


def test_infinite_week_is_rejected_without_crashing(data):
    set_value(data, "Allocations", 0, "Week", float("inf"))
    assert has_error(validate_data(data), "Allocations row 2", "'Week'")


def test_overlapping_cycles_are_each_reported_once(data):
    # P05->P07->P01 and P05->P08->P01 both close cycles through P01->P05
    for pred, succ, dep_id in [("P05", "P07", "D05"), ("P07", "P01", "D06"), ("P05", "P08", "D07"), ("P08", "P01", "D08")]:
        add_dependency(data, pred, succ, dep_id)
    cycle_errors = [e for e in validate_data(data) if "cycle detected" in e]
    assert len(cycle_errors) == 2
    assert len(set(cycle_errors)) == 2


@pytest.mark.parametrize("bad", [None, "week two", float("nan")])
def test_malformed_requirement_dates_do_not_crash_allocation_check(data, bad):
    set_value(data, "Project_Requirements", 0, "StartWeek", bad)  # Q001 has allocations
    errors = validate_data(data)  # must not raise
    assert isinstance(errors, list)
    assert has_error(errors, "Project_Requirements row 2", "StartWeek")


def test_duplicate_requirement_ids_do_not_crash(data):
    set_value(data, "Project_Requirements", 1, "RequirementID", "Q001")
    errors = validate_data(data)
    assert has_error(errors, "duplicate RequirementID Q001")


def test_infinite_calendar_week_does_not_crash(data):
    set_value(data, "Calendar", 0, "Week", float("inf"))
    errors = validate_data(data)  # must not raise OverflowError
    assert has_error(errors, "Calendar row 2", "'Week'", "finite")

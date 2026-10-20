import copy

import pandas as pd
import pytest

from src.data_loader import load_data
from src.dependencies import (
    EXPOSURE_COLUMNS,
    downstream_exposure,
    exposed_project_ids,
    projects_supported_by_resource,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


def graph(edges, projects=None, allocations=None):
    """Hand-made data: projects plus Blocks edges, and optional allocations (resource, project, week, hours)."""
    ids = projects or sorted({p for e in edges for p in e} | {"Z"})
    return {
        "Projects": pd.DataFrame({"ProjectID": ids}),
        "Dependencies": pd.DataFrame(
            [(f"D{i}", a, b, "Blocks") for i, (a, b) in enumerate(edges, 1)],
            columns=["DependencyID", "PredecessorID", "SuccessorID", "Type"],
        ),
        "Allocations": pd.DataFrame(allocations or [], columns=["ResourceID", "ProjectID", "Week", "PlannedHrs"]),
    }


def pairs(table):
    return [(r.From, r.ProjectID, r.Depth) for r in table.itertuples()]


# ---------- real data (dependencies D01-D04, checked by hand from the raw sheet) ----------
def test_p01_blocks_p05(data):
    t = downstream_exposure(data, ["P01"])
    assert pairs(t) == [("P01", "P05", 1)]
    assert t.iloc[0]["Path"] == "P01 -> P05"


def test_p04_blocks_p03(data):
    assert pairs(downstream_exposure(data, ["P04"])) == [("P04", "P03", 1)]


def test_project_with_nothing_downstream_has_no_exposure(data):
    t = downstream_exposure(data, ["P05"])
    assert t.empty
    assert list(t.columns) == EXPOSURE_COLUMNS


def test_exposed_set_for_the_r15_story(data):
    # R15 works on P01, P02 and P06 in weeks 4-5 -> those block P05 and P09
    affected = projects_supported_by_resource(data, "R15", weeks=range(4, 6))
    assert affected == ["P01", "P02", "P06"]
    assert exposed_project_ids(data, affected) == ["P05", "P09"]


def test_the_open_p04_requirement_exposes_p03(data):
    assert exposed_project_ids(data, ["P04"]) == ["P03"]


def test_two_affected_projects_reaching_the_same_downstream_project(data):
    t = downstream_exposure(data, ["P01", "P06"])
    assert pairs(t) == [("P01", "P05", 1), ("P06", "P05", 1)]
    assert exposed_project_ids(data, ["P01", "P06"]) == ["P05"]


# ---------- resource to projects ----------
def test_projects_supported_by_resource_all_weeks(data):
    assert projects_supported_by_resource(data, "R15") == ["P01", "P02", "P06"]
    assert projects_supported_by_resource(data, "R08") == ["P05", "P06"]


def test_week_filter_narrows_the_projects(data):
    assert projects_supported_by_resource(data, "R15", weeks=[10]) == ["P01"]  # P02, P06 end earlier


def test_resource_with_no_allocation_supports_no_project(data):
    assert projects_supported_by_resource(data, "R99") == []


# ---------- transitive exposure (hand-made graphs) ----------
def test_chain_is_transitive_with_depths():
    d = graph([("A", "B"), ("B", "C"), ("C", "D")])
    t = downstream_exposure(d, ["A"])
    assert pairs(t) == [("A", "B", 1), ("A", "C", 2), ("A", "D", 3)]
    assert t.iloc[2]["Path"] == "A -> B -> C -> D"


def test_middle_of_chain_only_exposes_what_is_after_it():
    d = graph([("A", "B"), ("B", "C"), ("C", "D")])
    assert pairs(downstream_exposure(d, ["B"])) == [("B", "C", 1), ("B", "D", 2)]


def test_branching_exposure():
    d = graph([("A", "B"), ("A", "C"), ("C", "D")])
    assert pairs(downstream_exposure(d, ["A"])) == [("A", "B", 1), ("A", "C", 1), ("A", "D", 2)]


def test_diamond_reports_each_project_once_at_its_shortest_depth():
    d = graph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    t = downstream_exposure(d, ["A"])
    assert pairs(t) == [("A", "B", 1), ("A", "C", 1), ("A", "D", 2)]


def test_dependency_direction_matters():
    d = graph([("A", "B")])
    assert downstream_exposure(d, ["B"]).empty  # B blocks nothing; being blocked is not exposure


def test_affected_project_inside_another_affected_projects_downstream():
    d = graph([("A", "B"), ("B", "C")])
    assert exposed_project_ids(d, ["A", "B"]) == ["C"]  # B is affected itself, so it is not listed as exposed


def test_cycle_does_not_loop_forever():
    d = graph([("A", "B"), ("B", "A")])  # invalid per the validator, but must not hang
    assert pairs(downstream_exposure(d, ["A"])) == [("A", "B", 1)]


def test_no_dependencies_at_all():
    d = graph([], projects=["A", "B"])
    assert downstream_exposure(d, ["A"]).empty


def test_unknown_project_raises_a_clear_error():
    d = graph([("A", "B")])
    with pytest.raises(ValueError, match="Unknown project.*X"):
        downstream_exposure(d, ["X"])


def test_duplicate_affected_projects_are_handled():
    d = graph([("A", "B")])
    assert pairs(downstream_exposure(d, ["A", "A"])) == [("A", "B", 1)]


def test_inputs_are_not_modified(data):
    before = copy.deepcopy(data)
    downstream_exposure(data, ["P01", "P04"])
    projects_supported_by_resource(data, "R15", weeks=[4, 5])
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])

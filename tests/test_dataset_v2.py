"""Step 6: dataset facts for the V2 workbook.

The four planted situations are NOT tested here. They are discovered and tested against the real
engine functions in tests/test_allocation.py (Step 17), so no discovery logic is duplicated.
"""
import pytest

from src.allocation import get_qualified_resources
from src.capacity import build_capacity_table
from src.data_loader import load_data
from src.validator import validate_data


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_dataset_passes_validator(data):
    assert validate_data(data) == []


def test_dataset_counts(data):
    assert len(data["Resources"]) == 28
    assert len(data["Projects"]) == 9
    assert data["Resource_Skills"]["Skill"].nunique() == 12
    assert data["Project_Requirements"]["Skill"].isin(data["Resource_Skills"]["Skill"]).all()
    assert len(data["Calendar"]) == 12
    assert len(data["Project_Requirements"]) == 50
    assert len(data["Allocations"]) == 380
    assert len(data["Dependencies"]) == 4


def test_no_derived_or_label_columns_are_stored(data):
    banned = ("conflict", "risk", "util", "score", "fit", "cost", "overload", "flag")
    stored = [c for df in data.values() for c in df.columns if any(w in c.lower() for w in banned)]
    assert stored == []


def test_every_requirement_has_a_qualified_candidate(data):
    for rid in data["Project_Requirements"]["RequirementID"]:
        assert get_qualified_resources(data, rid), f"{rid} has nobody qualified"


def test_no_resource_is_overbooked_in_the_baseline_plan(data):
    # The tension comes from unmet requirements, not from the baseline allocations exceeding capacity.
    assert (build_capacity_table(data)["AvailableHrs"] >= 0).all()

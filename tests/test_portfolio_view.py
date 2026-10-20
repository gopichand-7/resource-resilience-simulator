"""Portfolio Map. Expected numbers were hand-calculated from the raw Excel file, not from the engine."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pathlib import Path

from src.data_loader import load_data
from src.portfolio_view import (
    conflict_display,
    coverage_figure,
    dependency_edges,
    exposure_from_gaps,
    headline_metrics,
    open_conflicts,
    project_summary,
    scarce_requirements,
    single_points,
)
from src.scenarios import apply_resource_unavailable, take_snapshot

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def snap(data):
    return take_snapshot(data)


@pytest.fixture(scope="module")
def summary(data, snap):
    return project_summary(data, snap).set_index("ProjectID")


# ---------- headline tiles ----------
def test_headline_tiles(data, snap):
    tiles = {label: value for label, value, _ in headline_metrics(data, snap)}
    assert tiles == {"Required hours": 5430, "Staffed hours": 5142, "Unmet hours": 288,
                     "Reqs with gaps": 3, "Single points": 3}   # R15, R06, R11


def test_allocated_tile_explains_the_coverage_percentage(data, snap):
    help_text = {label: h for label, _, h in headline_metrics(data, snap)}["Staffed hours"]
    assert "94.7%" in help_text                                                        # 5142 / 5430


# ---------- project summary ----------
def test_one_row_per_project_and_totals_add_up(summary, snap):
    assert len(summary) == 9
    assert summary["RequiredHrs"].sum() == 5430
    assert summary["AllocatedHrs"].sum() == 5142
    assert summary["GapHrs"].sum() == 288


def test_hand_calculated_project_rows(summary):
    p08, p04 = summary.loc["P08"], summary.loc["P04"]
    assert (p08["RequiredHrs"], p08["AllocatedHrs"], p08["GapHrs"]) == (482, 264, 218)
    assert p08["Coverage"] == pytest.approx(264 / 482)
    assert (p04["RequiredHrs"], p04["AllocatedHrs"], p04["GapHrs"]) == (180, 110, 70)
    assert p04["Coverage"] == pytest.approx(110 / 180)
    assert p08["OpenRequirements"] == 2 and p04["OpenRequirements"] == 1


def test_fully_staffed_projects_show_full_coverage(summary):
    assert summary.loc["P01", "Coverage"] == 1.0 and summary.loc["P01", "GapHrs"] == 0


def test_summary_facts_about_scarcity_and_shortage(summary):
    assert summary.loc["P01", "SinglePointRequirements"] == 1     # Q001 only R15
    assert summary.loc["P04", "SinglePointRequirements"] == 1     # Q020
    assert summary.loc["P08", "CapacityShortageHrs"] == 82
    assert summary.loc["P04", "CapacityShortageHrs"] == 40
    assert summary.loc["P02", "CapacityShortageHrs"] == 0


def test_summary_dependency_columns(summary):
    assert summary.loc["P05", "BlockedBy"] == "P01, P06"
    assert summary.loc["P01", "Blocks"] == "P05" and summary.loc["P04", "Blocks"] == "P03"
    assert summary.loc["P07", "Blocks"] == "" and summary.loc["P07", "BlockedBy"] == ""


def test_no_score_or_rating_column_exists(summary):
    for name in summary.columns:
        assert not any(w in name.lower() for w in ["score", "risk", "health", "rating", "rank"])


# ---------- conflicts, scarcity, dependencies ----------
def test_conflicts_are_listed_in_a_fixed_order(snap):
    c = open_conflicts(snap)
    assert list(c["ConflictType"]) == ["Staffing gap"] * 3 + ["Capacity shortage"] * 2 + ["Competing requirements"] * 2
    assert list(c["RequirementID"][:3]) == ["Q020", "Q043", "Q045"]


def test_scarce_requirements_exclude_the_broad_pools_and_put_the_thinnest_first(snap):
    s = scarce_requirements(snap)
    assert "Broader pool" not in set(s["Category"])
    cats = list(s["Category"])
    assert cats == sorted(cats, key=lambda c: {"No qualified resource": 0, "Single point of failure": 1}.get(c, 2))
    singles = set(s.loc[s["Category"] == "Single point of failure", "RequirementID"])
    assert {"Q001", "Q012", "Q020", "Q043"} <= singles


def test_single_points_of_failure_people(data):
    p = single_points(data).set_index("ResourceID")
    assert set(p.index) == {"R15", "R06", "R11"}
    assert p.loc["R15", "Requirements"] == "Q001, Q020" and p.loc["R15", "Projects"] == "P01, P04"


def test_dependency_edges_and_unmet_hours(data, snap):
    e = dependency_edges(data, snap)
    assert len(e) == 4
    pairs = set(zip(e["Predecessor"], e["Successor"]))
    assert pairs == {("P01", "P05"), ("P06", "P05"), ("P02", "P09"), ("P04", "P03")}
    unmet = dict(zip(e["Predecessor"], e["PredecessorUnmetHrs"]))
    assert unmet["P04"] == 70 and unmet["P01"] == 0


def test_baseline_exposure_is_p03_through_p04(data, snap):
    x = exposure_from_gaps(data, snap)
    assert len(x) == 1
    assert (x.iloc[0]["ProjectID"], x.iloc[0]["From"], x.iloc[0]["Path"]) == ("P03", "P04", "P04 -> P03")


def test_the_same_views_work_on_a_shocked_plan(data):
    shocked = apply_resource_unavailable(data, "R15", [4, 5])
    s = take_snapshot(shocked)
    x = exposure_from_gaps(shocked, s)
    assert set(x["ProjectID"]) == {"P05", "P09", "P03"}
    assert project_summary(shocked, s).set_index("ProjectID").loc["P02", "GapHrs"] == 28


# ---------- chart ----------
def test_coverage_chart_has_two_series_with_the_right_hours(summary):
    fig = coverage_figure(summary.reset_index())
    assert [t.name for t in fig.data] == ["Staffed hours", "Unmet hours"]
    assert sum(fig.data[0].x) == 5142 and sum(fig.data[1].x) == 288
    assert len(fig.data[0].y) == 9
    assert fig.layout.barmode == "stack"
    assert fig.data[0].marker.color != fig.data[1].marker.color


def test_only_projects_with_unmet_hours_get_a_label(summary):
    fig = coverage_figure(summary.reset_index())
    labelled = [t for t in fig.data[1].text if t]
    assert len(labelled) == 2 and "218h unmet" in labelled and "70h unmet" in labelled


# ---------- the real page, run headlessly ----------
@pytest.fixture(autouse=True)
def fresh_cache():
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_page_renders_with_tiles_and_sections():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.header[0].value == "Portfolio Map"
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Unmet hours"] == "288" and tiles["Required hours"] == "5,430"
    assert [s.value for s in at.main.subheader] == [
        "Coverage by project", "Open problems the engine found", "Scarce skills", "Dependencies"]


def test_page_names_the_single_point_people():
    at = AppTest.from_file(APP, default_timeout=60).run()
    text = " ".join(m.value for m in at.markdown)
    assert "Rohit Sharma" in text and "Q001, Q020" in text
    assert "P03" in text and "P04 -> P03" in text


# ---------- conflicts shown as a wrapping table ----------
def test_conflict_display_has_friendly_headers_and_whole_hours(snap):
    d = conflict_display(open_conflicts(snap))
    assert list(d.columns) == ["Type", "Requirement", "Project", "Weeks", "Hours", "Detail"]   # blank person column dropped
    assert d["Hours"].dtype.kind == "i" and list(d["Hours"]) == [70, 162, 56, 40, 82, 190, 258]


def test_conflict_display_keeps_the_person_column_when_it_has_values(data):
    from src.allocation import detect_conflicts
    import copy
    d = copy.deepcopy(data)
    d["Resources"].loc[d["Resources"]["ResourceID"] == "R15", "WeeklyCapacityHrs"] = 30    # R15 now over-allocated
    shown = conflict_display(detect_conflicts(d))
    assert "Person" in shown.columns and "R15" in set(shown["Person"])


def test_the_page_shows_full_conflict_text_in_a_table():
    at = AppTest.from_file(APP, default_timeout=60).run()
    table = next(t for t in at.table if "Detail" in t.value.columns)
    details = list(table.value["Detail"])
    assert "Hours of qualified resources are already held by: Q024 (P05) 128h; Q031 (P06) 50h; Q039 (P07) 80h" in details


def test_no_tile_label_is_long_enough_to_be_cut_off():
    at = AppTest.from_file(APP, default_timeout=60).run()
    for m in at.metric:
        assert len(m.label) <= 14, m.label
        assert len(m.value) <= 16, m.value

"""Portfolio Map: the tables and the chart behind the page (no Streamlit code, so it can be tested).

Everything here is a plain summary of what the engine already derived. Nothing is scored or ranked:
a project is never given a "health score" or a "risk level", only facts (hours, coverage, which
single people it depends on, which projects it blocks).
"""
import pandas as pd
import plotly.graph_objects as go

from src.allocation import (
    BROADER_POOL,
    CAPACITY_SHORTAGE,
    COMPETING_REQUIREMENTS,
    NO_QUALIFIED,
    SINGLE_POINT,
    STAFFING_GAP,
    find_single_points_of_failure,
)
from src.dependencies import downstream_exposure

EPS = 1e-9
CONFLICT_ORDER = {STAFFING_GAP: 0, CAPACITY_SHORTAGE: 1, COMPETING_REQUIREMENTS: 2}
SCARCITY_ORDER = {NO_QUALIFIED: 0, SINGLE_POINT: 1}
ALLOCATED_COLOR = "#3987e5"   # categorical slot 1 (blue)
UNMET_COLOR = "#B45309"       # amber: unmet hours (Program Office style)


def headline_metrics(data, snapshot):
    """(label, value, help) tiles for the top of the page."""
    m = snapshot["metrics"]
    required, allocated = m["Required hours"], m["Allocated hours"]
    coverage = allocated / required if required else float("nan")
    return [
        ("Required hours", required, "Hours the plan asks for across all requirements"),
        ("Staffed hours", allocated, f"{coverage:.1%} of required hours are staffed"),
        ("Unmet hours", m["Unmet hours (staffing gap)"], "Required hours that nobody is planned to work"),
        ("Reqs with gaps", m["Requirements with gaps"], "Requirements with at least one unstaffed hour"),
        ("Single points", len(find_single_points_of_failure(data)),
         "Single points of failure: people who are the only qualified person for an active requirement"),
    ]


def project_summary(data, snapshot):
    """One row per project: size, coverage, and the facts that matter for pressure."""
    projects = data["Projects"][["ProjectID", "ProjectName", "BusinessPriority", "Status", "StartWeek", "DeadlineWeek"]].copy()
    demand = snapshot["demand"].groupby("ProjectID").agg(
        RequiredHrs=("RequiredHrs", "sum"), AllocatedHrs=("AllocatedHrs", "sum"), GapHrs=("GapHrs", "sum"),
        OpenRequirements=("GapHrs", lambda s: int((s > EPS).sum())),
    )
    out = projects.merge(demand, left_on="ProjectID", right_index=True, how="left").fillna(
        {"RequiredHrs": 0.0, "AllocatedHrs": 0.0, "GapHrs": 0.0, "OpenRequirements": 0})
    out["Coverage"] = out["AllocatedHrs"] / out["RequiredHrs"].where(out["RequiredHrs"] > 0)

    scarcity = snapshot["scarcity"]
    fragile = scarcity[scarcity["Category"].isin([SINGLE_POINT, NO_QUALIFIED])].groupby("ProjectID").size()
    out["SinglePointRequirements"] = out["ProjectID"].map(fragile).fillna(0).astype(int)

    conflicts = snapshot["conflicts"]
    short = conflicts[conflicts["ConflictType"] == CAPACITY_SHORTAGE].groupby("ProjectID")["Hours"].sum()
    out["CapacityShortageHrs"] = out["ProjectID"].map(short).fillna(0.0)

    deps = data["Dependencies"]
    blocks = deps.groupby("PredecessorID")["SuccessorID"].apply(lambda s: ", ".join(sorted(set(s))))
    blocked_by = deps.groupby("SuccessorID")["PredecessorID"].apply(lambda s: ", ".join(sorted(set(s))))
    out["Blocks"] = out["ProjectID"].map(blocks).fillna("")
    out["BlockedBy"] = out["ProjectID"].map(blocked_by).fillna("")
    return out.sort_values("ProjectID").reset_index(drop=True)


def open_conflicts(snapshot):
    """The conflicts the engine found, in a fixed readable order (type, then requirement)."""
    c = snapshot["conflicts"].copy()
    if c.empty:
        return c
    c["_order"] = c["ConflictType"].map(CONFLICT_ORDER).fillna(9)
    return c.sort_values(["_order", "RequirementID", "ResourceID"]).drop(columns="_order").reset_index(drop=True)


def scarce_requirements(snapshot):
    """Requirements whose skill pool is thin (everything except 'Broader pool'), thinnest first."""
    s = snapshot["scarcity"]
    s = s[s["Category"] != BROADER_POOL].copy()
    s["_order"] = s["Category"].map(SCARCITY_ORDER).fillna(2)
    return s.sort_values(["_order", "RequirementID"]).drop(columns="_order").reset_index(drop=True)


def single_points(data):
    return find_single_points_of_failure(data)


def dependency_edges(data, snapshot):
    """Every 'blocks' link, with how many hours the blocking project still has unstaffed."""
    names = data["Projects"].set_index("ProjectID")["ProjectName"]
    gap = snapshot["demand"].groupby("ProjectID")["GapHrs"].sum()
    deps = data["Dependencies"]
    out = pd.DataFrame({
        "Predecessor": deps["PredecessorID"], "PredecessorName": deps["PredecessorID"].map(names),
        "Successor": deps["SuccessorID"], "SuccessorName": deps["SuccessorID"].map(names),
        "PredecessorUnmetHrs": deps["PredecessorID"].map(gap).fillna(0.0),
    })
    return out.sort_values(["Predecessor", "Successor"]).reset_index(drop=True)


def exposure_from_gaps(data, snapshot):
    """Projects that depend (directly or through a chain) on a project that has unmet hours.

    A project that has unmet hours of its own is not listed here, it is already a gap project.
    """
    gap = snapshot["demand"].groupby("ProjectID")["GapHrs"].sum()
    gap_projects = sorted(gap[gap > EPS].index)
    table = downstream_exposure(data, gap_projects)
    if table.empty:
        return table
    return table[~table["ProjectID"].isin(gap_projects)].reset_index(drop=True)


def coverage_figure(summary):
    """Stacked horizontal bars: staffed hours and unmet hours per project (same hours, same axis)."""
    labels = [f"{r.ProjectID}  {r.ProjectName}" for r in summary.itertuples()]
    fig = go.Figure()
    fig.add_bar(y=labels, x=summary["AllocatedHrs"], name="Staffed hours", orientation="h",
                marker_color=ALLOCATED_COLOR,
                hovertemplate="%{y}<br>Staffed: %{x:,.0f}h<extra></extra>")
    fig.add_bar(y=labels, x=summary["GapHrs"], name="Unmet hours", orientation="h", marker_color=UNMET_COLOR,
                text=[f"{g:,.0f}h unmet" if g > EPS else "" for g in summary["GapHrs"]], textposition="outside",
                cliponaxis=False, hovertemplate="%{y}<br>Unmet: %{x:,.0f}h<extra></extra>")
    fig.update_layout(
        barmode="stack", bargap=0.35, height=60 + 38 * len(labels),
        margin=dict(l=0, r=40, t=40, b=0), legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"),
        xaxis=dict(title="Hours", showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        yaxis=dict(autorange="reversed", title=""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def conflict_display(conflicts):
    """Conflicts as a plain, wrap-friendly table: friendly headers, whole hours, blank resource column dropped."""
    out = conflicts.copy()
    if "Hours" in out:
        out["Hours"] = out["Hours"].round().astype(int)
    if "ResourceID" in out and (out["ResourceID"].fillna("") == "").all():
        out = out.drop(columns=["ResourceID"])
    return out.rename(columns={"ConflictType": "Type", "RequirementID": "Requirement", "ProjectID": "Project",
                               "ResourceID": "Person"}).reset_index(drop=True)

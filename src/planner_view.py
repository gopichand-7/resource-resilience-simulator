"""Allocation Planner: tables and chart behind the page (no Streamlit code, so it can be tested).

The planner lists every person who could work on one requirement, with the reason behind each fit.
It never marks a "best" or "recommended" person: the fit score is a transparent ranking aid, and the
page says in plain words where it can mislead (for example, when the highest-scoring people are all
below the minimum proficiency).
"""
import pandas as pd
import plotly.graph_objects as go

from src.allocation import evaluate_candidates
from src.portfolio_view import ALLOCATED_COLOR, UNMET_COLOR

EPS = 1e-9


def _requirement_row(data, requirement_id):
    req = data["Project_Requirements"]
    match = req[req["RequirementID"] == requirement_id]
    if match.empty:
        raise ValueError(f"Unknown requirement: {requirement_id!r}")
    return match.iloc[0]


def requirement_options(data, snapshot, only_unmet=True):
    """Requirements to choose from, with a readable label. Unmet ones first (largest gap first)."""
    names = data["Projects"].set_index("ProjectID")["ProjectName"]
    demand = snapshot["demand"].set_index("RequirementID")
    req = data["Project_Requirements"]
    rows = []
    for r in req.itertuples():
        gap = float(demand.loc[r.RequirementID, "GapHrs"])
        if only_unmet and gap <= EPS:
            continue
        label = f"{r.RequirementID} | {r.ProjectID} {names[r.ProjectID]} | {r.Skill} (min {r.MinProficiency})"
        if gap > EPS:
            label += f" | {gap:,.0f}h unmet"
        rows.append({"RequirementID": r.RequirementID, "Label": label, "GapHrs": gap})
    out = pd.DataFrame(rows, columns=["RequirementID", "Label", "GapHrs"])
    return out.sort_values(["GapHrs", "RequirementID"], ascending=[False, True]).reset_index(drop=True)


def requirement_facts(data, snapshot, requirement_id):
    """Everything the page shows about the requirement itself."""
    r = _requirement_row(data, requirement_id)
    d = snapshot["demand"].set_index("RequirementID").loc[requirement_id]
    s = snapshot["scarcity"].set_index("RequirementID").loc[requirement_id]
    names = data["Projects"].set_index("ProjectID")["ProjectName"]
    return {
        "RequirementID": requirement_id, "ProjectID": r["ProjectID"], "ProjectName": names[r["ProjectID"]],
        "Role": r["Role"], "Skill": r["Skill"], "MinProficiency": int(r["MinProficiency"]),
        "HoursPerWeek": float(r["HoursPerWeek"]), "StartWeek": int(r["StartWeek"]), "EndWeek": int(r["EndWeek"]),
        "RequiredHrs": float(d["RequiredHrs"]), "AllocatedHrs": float(d["AllocatedHrs"]), "GapHrs": float(d["GapHrs"]),
        "QualifiedCount": int(s["QualifiedCount"]), "Scarcity": s["Category"],
    }


def candidate_view(data, snapshot, requirement_id):
    """Every candidate with the numbers behind the fit score. Sorted by fit score (highest first)."""
    c = evaluate_candidates(data, requirement_id, capacity=snapshot["capacity"]).copy()
    minimum = int(_requirement_row(data, requirement_id)["MinProficiency"])
    c["MeetsMinimum"] = c["Proficiency"] >= minimum
    return c


def candidate_summary(facts, candidates):
    """Plain-words observations about the candidate list. Facts only: no recommendation."""
    n = len(candidates)
    if n == 0:
        return [f"Nobody in the dataset has the {facts['Skill']} skill."]
    meets = candidates[candidates["MeetsMinimum"]]
    verb = "meets" if len(meets) == 1 else "meet"
    lines = [f"{n} {'person has' if n == 1 else 'people have'} the {facts['Skill']} skill; "
             f"{len(meets)} of them {verb} the minimum level ({facts['MinProficiency']})."]
    if meets.empty:
        lines.append("Nobody meets the minimum proficiency, so every candidate is below the requirement.")
        return lines
    enough = meets[meets["Availability"] >= 1 - EPS]
    if facts["GapHrs"] <= EPS:
        lines.append("This requirement is fully staffed. Free hours below are counted after all planned work, "
                     "including this requirement, so the person already doing it shows little free time.")
    elif enough.empty:
        best_free = float(meets["TightestFreeHrs"].max())
        lines.append(
            f"None of the people who meet the minimum has {facts['HoursPerWeek']:g}h free every week of the "
            f"window; the most free hours in their tightest week is {best_free:g}h.")
    else:
        lines.append(f"{len(enough)} of them have the full {facts['HoursPerWeek']:g}h free in every week of the window.")
    top = candidates.iloc[0]
    if not top["MeetsMinimum"]:
        lines.append(
            f"The highest fit score ({top['FitScore']:.1f}, {top['ResourceName']}) belongs to someone below the "
            "minimum proficiency. A high score is not the same as being qualified.")
    return lines


def current_assignees(data, requirement_id):
    """Who the baseline plan already has on this requirement, with total planned hours."""
    a = data["Allocations"]
    a = a[a["RequirementID"] == requirement_id]
    out = a.groupby("ResourceID", as_index=False)["PlannedHrs"].sum()
    names = data["Resources"].set_index("ResourceID")["ResourceName"]
    out["ResourceName"] = out["ResourceID"].map(names)
    return out.sort_values("ResourceID").reset_index(drop=True)


def assignee_line(assignees):
    """One plain sentence naming who is already planned on the requirement, or None."""
    if assignees.empty:
        return None
    parts = [f"{r.ResourceName} ({r.ResourceID}), {r.PlannedHrs:,.0f}h" for r in assignees.itertuples()]
    return "**Currently planned:** " + "; ".join(parts) + "."


def candidate_table(candidates, assignee_ids=()):
    """The candidate list as a readable table: the answer columns first, no empty cells, no checkboxes."""
    t = candidates.copy()
    flags = []
    for row in t.itertuples():
        parts = []
        if row.ResourceID in set(assignee_ids):
            parts.append("Currently planned")
        if row.Constraints:
            parts.append(row.Constraints)
        flags.append("; ".join(parts) if parts else "None")
    out = pd.DataFrame({
        "ID": t["ResourceID"], "Name": t["ResourceName"], "Fit score": t["FitScore"].map(lambda v: f"{v:.1f}"),
        "Fit": t["FitLabel"], "Flags": flags, "Meets min": t["MeetsMinimum"].map({True: "Yes", False: "No"}),
        "Level": t["Proficiency"].astype(int), "Role": t["Role"],
        "Free h/wk (tightest)": t["TightestFreeHrs"].map(lambda v: f"{v:g}"),
        "Peak load": t["PeakUtilization"].map(lambda v: f"{v:.0%}"),
    })
    return out.reset_index(drop=True)


def requirement_conflicts(snapshot, requirement_id):
    c = snapshot["conflicts"]
    return c[c["RequirementID"] == requirement_id].reset_index(drop=True)


def weekly_figure(snapshot, requirement_id):
    """Staffed and unmet hours in each week of the requirement's window."""
    w = snapshot["weekly_demand"]
    w = w[w["RequirementID"] == requirement_id].sort_values("Week")
    weeks = [f"Wk {int(x)}" for x in w["Week"]]
    fig = go.Figure()
    fig.add_bar(x=weeks, y=w["AllocatedHrs"], name="Staffed hours", marker_color=ALLOCATED_COLOR,
                hovertemplate="%{x}<br>Staffed: %{y:,.1f}h<extra></extra>")
    fig.add_bar(x=weeks, y=w["GapHrs"], name="Unmet hours", marker_color=UNMET_COLOR,
                hovertemplate="%{x}<br>Unmet: %{y:,.1f}h<extra></extra>")
    fig.update_layout(
        barmode="stack", bargap=0.3, height=300, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"),
        yaxis=dict(title="Hours", gridcolor="rgba(128,128,128,0.2)", zeroline=False), xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

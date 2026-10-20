"""Trade-offs: choosing a response to the active shock and reading what it fixed and what it cost.

No Streamlit code here, so everything can be tested. Three plans are compared:

    baseline  (the original plan)  ->  shocked plan  ->  plan after the response

A response is always applied to the SHOCKED plan (responses do not stack). The page reports facts
(hours recovered, conflicts created, warnings) and never ranks the responses or picks a best one.
"""
import pandas as pd
import plotly.graph_objects as go

from src.allocation import FLAG_BELOW_PROFICIENCY
from src.planner_view import candidate_view, requirement_options
from src.portfolio_view import ALLOCATED_COLOR, UNMET_COLOR
from src.responses import (
    apply_reallocate,
    apply_substitute,
    apply_timeline_extension,
    evaluate_response,
)
from src.scenarios import take_snapshot
from src.shock_view import BASELINE_COLOR, calendar_weeks, project_options

EPS = 1e-9

RESPONSE_SUBSTITUTE = "Bring in another person"
RESPONSE_REALLOCATE = "Move someone's hours"
RESPONSE_TIMELINE = "Extend a deadline"
RESPONSE_KINDS = [RESPONSE_SUBSTITUTE, RESPONSE_REALLOCATE, RESPONSE_TIMELINE]

# (row label, metric key). Row labels stay short so a table never has to cut them off.
MEASURES = [
    ("Unmet hours", "Unmet hours (staffing gap)"),
    ("Requirements with gaps", "Requirements with gaps"),
    ("Capacity-shortage hours", "Capacity-shortage hours"),
    ("Over-allocated hours", "Over-allocated hours"),
    ("Downstream projects exposed", "Downstream projects exposed"),
]


# ---------------------------------------------------------------------------
# choices (all of them read the SHOCKED plan)
# ---------------------------------------------------------------------------
def unmet_options(data, snapshot):
    """Requirements that still have unmet hours in this plan (largest gap first)."""
    return requirement_options(data, snapshot, only_unmet=True)


def gap_weeks(snapshot, requirement_id):
    """Weeks in which the requirement has unmet hours."""
    w = snapshot["weekly_demand"]
    w = w[(w["RequirementID"] == requirement_id) & (w["GapHrs"] > EPS)]
    return sorted(int(x) for x in w["Week"])


def unavailable_people(data, weeks):
    """People with zero capacity (marked unavailable) in at least one of the weeks."""
    table = data.get("Unavailability")
    if table is None or table.empty or not weeks:
        return set()
    t = table[pd.to_numeric(table["Week"]).isin(set(weeks))]
    return set(t["ResourceID"])


def substitute_people(data, snapshot, requirement_id, weeks=None):
    """People with the skill who are not unavailable in the chosen weeks, best fit score first, as a labelled
    table. A label is not a recommendation."""
    c = candidate_view(data, snapshot, requirement_id)
    if weeks:
        c = c[~c["ResourceID"].isin(unavailable_people(data, weeks))]
    if c.empty:
        return pd.DataFrame(columns=["ResourceID", "Label"])
    labels = []
    for r in c.itertuples():
        text = f"{r.ResourceID} {r.ResourceName} | fit {r.FitScore:.0f} ({r.FitLabel})"
        if not r.MeetsMinimum:
            text += " | below minimum level"
        labels.append(text)
    return pd.DataFrame({"ResourceID": c["ResourceID"].values, "Label": labels})


def _hours_held(data, resource_id, requirement_id, week):
    a = data["Allocations"]
    m = ((a["ResourceID"] == resource_id) & (a["RequirementID"] == requirement_id)
         & (pd.to_numeric(a["Week"]) == week))
    return float(pd.to_numeric(a.loc[m, "PlannedHrs"]).sum())


def reallocate_sources(data, resource_id, target_requirement):
    """Requirements (other than the target) on which this person currently holds hours, with a label."""
    a = data["Allocations"]
    a = a[(a["ResourceID"] == resource_id) & (a["RequirementID"] != target_requirement)]
    held = a.groupby("RequirementID")["PlannedHrs"].apply(lambda s: float(pd.to_numeric(s).sum()))
    req = data["Project_Requirements"].set_index("RequirementID")
    rows = [{"RequirementID": rid,
             "Label": f"{rid} | {req.loc[rid, 'ProjectID']} {req.loc[rid, 'Skill']} | {hrs:,.0f}h planned"}
            for rid, hrs in held.items() if hrs > EPS]
    return pd.DataFrame(rows, columns=["RequirementID", "Label"]).sort_values("RequirementID").reset_index(drop=True)


def reallocate_limit(data, snapshot, resource_id, from_requirement, to_requirement, weeks):
    """Most hours per week that can move over every chosen week: the smaller of the person's hours on the
    source and the target's unmet hours, taken across the weeks (0 means the move is impossible)."""
    w = snapshot["weekly_demand"]
    gaps = w[w["RequirementID"] == to_requirement].set_index("Week")["GapHrs"]
    limits = [min(float(gaps.get(week, 0.0)), _hours_held(data, resource_id, from_requirement, week)) for week in weeks]
    return min(limits) if limits else 0.0


def extendable_projects(data):
    """Projects whose deadline can still move later (the calendar has a later week)."""
    _, last = calendar_weeks(data)
    p = project_options(data)
    deadline = data["Projects"].set_index("ProjectID")["DeadlineWeek"].astype(int)
    p = p[p["ProjectID"].map(deadline) < last].copy()
    p["Deadline"] = p["ProjectID"].map(deadline).astype(int)
    return p.reset_index(drop=True)


# ---------------------------------------------------------------------------
# applying a response
# ---------------------------------------------------------------------------
def _weeks_text(weeks):
    weeks = sorted(weeks)
    return f"week {weeks[0]}" if len(weeks) == 1 else f"weeks {weeks[0]}-{weeks[-1]}"


def apply_response(shocked, kind, resource_id=None, requirement_id=None, from_requirement=None,
                   weeks=None, hours_per_week=None, project_id=None, new_deadline=None):
    """Returns (responded_data, plain-words label). Raises ValueError for a move the rules do not allow."""
    if kind == RESPONSE_SUBSTITUTE:
        data = apply_substitute(shocked, requirement_id, resource_id, weeks)
        return data, f"{resource_id} covers the unmet hours of {requirement_id}, {_weeks_text(weeks)}"
    if kind == RESPONSE_REALLOCATE:
        data = apply_reallocate(shocked, resource_id, from_requirement, requirement_id, weeks, hours_per_week)
        return data, (f"{resource_id} moves {hours_per_week:g}h a week from {from_requirement} to "
                      f"{requirement_id}, {_weeks_text(weeks)}")
    if kind == RESPONSE_TIMELINE:
        data = apply_timeline_extension(shocked, project_id, new_deadline)
        return data, f"{project_id} deadline moves to week {new_deadline}"
    raise ValueError(f"Unknown response type: {kind!r}")


def run_response(baseline, shocked, kind, **params):
    """Apply a response to the shocked plan and measure it. Returns what the page keeps in session state."""
    responded, label = apply_response(shocked, kind, **params)
    return {
        "response_data": responded,
        "response_label": label,
        "response_kind": kind,
        "response_result": {
            "snapshot": take_snapshot(responded),
            "evaluation": evaluate_response(baseline["data"], shocked, responded),
        },
    }


# ---------------------------------------------------------------------------
# reading the result
# ---------------------------------------------------------------------------
def compress_weeks(text):
    """'3, 4, 5, 9' -> '3-5, 9' so a long run of weeks stays short."""
    nums = sorted({int(x) for x in str(text).replace(" ", "").split(",") if x})
    parts, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            parts.append((start, prev)); start = prev = n
        if i == len(nums) - 1:
            parts.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in parts)


def _signed(x):
    return "no change" if abs(x) < EPS else f"{x:+,.0f}"


def comparison_table(baseline_snapshot, shocked_snapshot, response_snapshot):
    """Five headline measures across the three plans, plus the change the response made."""
    rows = []
    for label, key in MEASURES:
        b, s, r = (baseline_snapshot["metrics"][key], shocked_snapshot["metrics"][key],
                   response_snapshot["metrics"][key])
        rows.append({"Measure": label, "Baseline": f"{b:,.0f}", "After shock": f"{s:,.0f}",
                     "After response": f"{r:,.0f}", "Change from the response": _signed(r - s)})
    return pd.DataFrame(rows)


def _conflict_name(row):
    where = row.RequirementID if str(row.RequirementID).strip() else row.ResourceID
    return f"{row.ConflictType} on {where}"


def fixed_lines(evaluation):
    """What the response recovered. Facts only."""
    vs = evaluation["vs_shock"]
    m = vs["metrics"].set_index("Metric")
    unmet = m.loc["Unmet hours (staffing gap)"]
    lines = []
    if unmet["Delta"] < -EPS:
        lines.append(f"Unmet hours fall by {-unmet['Delta']:,.0f}h ({unmet['Baseline']:,.0f}h to {unmet['Scenario']:,.0f}h).")
    elif unmet["Delta"] > EPS:
        lines.append(f"Unmet hours do not fall; they rise by {unmet['Delta']:,.0f}h.")
    else:
        lines.append(f"Unmet hours do not change ({unmet['Scenario']:,.0f}h).")
    c = vs["conflict_changes"]
    better = c[c["Status"].isin(["Resolved", "Improved"])]
    if not better.empty:
        parts = [f"{_conflict_name(r)} ({r.BaselineHours:,.0f}h to {r.ScenarioHours:,.0f}h)" for r in better.itertuples()]
        lines.append("Conflicts resolved or reduced: " + "; ".join(parts) + ".")
    return lines


def cost_lines(evaluation, kind):
    """What the response cost: conflicts that appeared or grew, and the soft warnings. Facts only."""
    vs = evaluation["vs_shock"]
    lines = []
    c = vs["conflict_changes"]
    worse = c[c["Status"].isin(["New", "Worsened"])]
    if not worse.empty:
        parts = [f"{_conflict_name(r)} ({r.BaselineHours:,.0f}h to {r.ScenarioHours:,.0f}h)" for r in worse.itertuples()]
        lines.append("Conflicts that are new or bigger after the response: " + "; ".join(parts) + ".")
    for w in evaluation["warnings"].itertuples():
        if w.Warning == FLAG_BELOW_PROFICIENCY:
            lines.append(f"Warning, {w.ResourceID}: {w.Hours:,.0f}h on {w.RequirementID} (weeks {compress_weeks(w.Weeks)}) are staffed "
                         "by someone below the required proficiency level. They count as staffed, but qualification risk remains.")
        else:
            lines.append(f"Warning, {w.ResourceID}: {w.Warning.lower()} ({w.RequirementID}, weeks {compress_weeks(w.Weeks)}, {w.Hours:,.0f}h).")
    exposed = vs["metrics"].set_index("Metric").loc["Downstream projects exposed"]
    if exposed["Delta"] > EPS:
        lines.append(f"Downstream projects exposed rise from {exposed['Baseline']:,.0f} to {exposed['Scenario']:,.0f}.")
    if kind == RESPONSE_REALLOCATE and not worse.empty:
        lines.append("The hours were moved, not created: the problem moved to another requirement.")
    if not lines:
        lines.append("No new conflicts or warnings were created.")
    return lines


def still_open_line(evaluation):
    """Where the response leaves the plan compared with the ORIGINAL plan."""
    m = evaluation["vs_baseline"]["metrics"].set_index("Metric").loc["Unmet hours (staffing gap)"]
    if m["Delta"] > EPS:
        return f"Compared with the original plan, {m['Delta']:,.0f}h more is still unmet ({m['Scenario']:,.0f}h against {m['Baseline']:,.0f}h)."
    if m["Delta"] < -EPS:
        return f"Compared with the original plan, {-m['Delta']:,.0f}h less is unmet."
    return "Unmet hours are back to the original plan's level."


def qualification_note(evaluation):
    """One line under the summary when covered hours rest on below-level proficiency (calculated, never hardcoded)."""
    w = evaluation["warnings"]
    w = w[w["Warning"] == FLAG_BELOW_PROFICIENCY]
    if w.empty:
        return ""
    hours = float(w["Hours"].sum())
    return (f"Qualification: {hours:,.0f}h of the hours this response staffs are with people below the required proficiency. "
            "They count as staffed, so unmet hours fall, but the plan is not fully recovered.")


def timeline_note(kind):
    if kind == RESPONSE_TIMELINE:
        return ("A later deadline adds nobody and does not re-plan anyone. The same total effort is spread over more weeks, "
                "so weekly demand falls. People already planned at the old, higher rate now cover more than is needed in the "
                "old weeks, and the new weeks start with nobody planned. Unmet hours can therefore rise until staff are "
                "planned into the new weeks; what the extension buys is lower weekly demand, which people with less free "
                "time may be able to cover.")
    return ""


def person_names(data):
    """{'R12': 'R12 Lakshmi Devi'} so tables can show a name next to the ID."""
    res = data["Resources"]
    return {r.ResourceID: f"{r.ResourceID} {r.ResourceName}" for r in res.itertuples()}


def changes_display(evaluation, names=None):
    """Hours that moved, one row per person, requirement and run of weeks with the same change.
    Numbers stay numeric (right-aligned); 'Total h' is the change per week times the weeks in the run."""
    ch = evaluation["allocation_changes"]
    columns = ["Person", "Requirement", "Project", "Weeks", "Before h/wk", "After h/wk", "Change h/wk", "Total h"]
    if ch.empty:
        return ch
    names = names or {}
    ch = ch.copy()
    ch["BeforeHrs"], ch["AfterHrs"], ch["DeltaHrs"] = (ch[c].round(1) for c in ["BeforeHrs", "AfterHrs", "DeltaHrs"])
    ch = ch.sort_values(["ResourceID", "RequirementID", "Week"])
    rows, run = [], None
    for r in ch.itertuples():
        key = (r.ResourceID, r.RequirementID, r.ProjectID, r.BeforeHrs, r.AfterHrs)
        if run and run["key"] == key and r.Week == run["weeks"][-1] + 1:
            run["weeks"].append(r.Week)
        else:
            run = {"key": key, "weeks": [r.Week]}
            rows.append(run)
    out = []
    for run in rows:
        rid, req, proj, before, after = run["key"]
        delta = round(after - before, 1)
        out.append({"Person": names.get(rid, rid), "Requirement": req, "Project": proj,
                    "Weeks": compress_weeks(", ".join(str(w) for w in run["weeks"])),
                    "Before h/wk": before, "After h/wk": after, "Change h/wk": delta,
                    "Total h": round(delta * len(run["weeks"]), 1)})
    table = pd.DataFrame(out, columns=columns)
    return table.sort_values(["Person", "Change h/wk", "Requirement"], ascending=[True, True, True],
                             kind="stable").reset_index(drop=True)


def style_changes(table):
    """Right-aligned whole or one-decimal numbers, signed for the changes."""
    return table.style.hide(axis="index").format(
        {"Before h/wk": "{:g}", "After h/wk": "{:g}", "Change h/wk": "{:+g}", "Total h": "{:+g}"})


def warnings_display(evaluation, names=None):
    w = evaluation["warnings"]
    if w.empty:
        return w
    out = w.copy()
    out["Hours"] = out["Hours"].round().astype(int)
    out["Weeks"] = out["Weeks"].apply(compress_weeks)
    if names:
        out["ResourceID"] = out["ResourceID"].map(lambda r: names.get(r, r))
    return out.rename(columns={"ResourceID": "Person", "RequirementID": "Requirement"})


def three_way_figure(baseline_snapshot, shocked_snapshot, response_snapshot):
    """Unmet hours per project in the three plans (only projects with unmet hours in any of them)."""
    series = {name: snap["demand"].groupby("ProjectID")["GapHrs"].sum()
              for name, snap in [("Baseline", baseline_snapshot), ("After shock", shocked_snapshot),
                                 ("After response", response_snapshot)]}
    both = pd.DataFrame(series).fillna(0.0)
    both = both[(both > EPS).any(axis=1)].sort_index()
    colors = {"Baseline": BASELINE_COLOR, "After shock": UNMET_COLOR, "After response": ALLOCATED_COLOR}
    fig = go.Figure()
    for name in series:
        fig.add_bar(x=both.index, y=both[name], name=name, marker_color=colors[name],
                    hovertemplate="%{x}<br>" + name + ": %{y:,.0f}h<extra></extra>")
    fig.update_layout(
        barmode="group", bargap=0.3, height=320, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"),
        yaxis=dict(title="Unmet hours", gridcolor="rgba(128,128,128,0.2)", zeroline=False), xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

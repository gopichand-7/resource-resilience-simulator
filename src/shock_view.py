"""Shock Simulator: choosing a shock, applying it, and telling the story of what it hit.

No Streamlit code here, so everything can be tested. The story is built from facts the engine already
derived (hours, projects, dependencies). It reports EXPOSURE, never predicted delay, and never
recommends a response.
"""
import pandas as pd
import plotly.graph_objects as go

from src.dependencies import projects_supported_by_resource
from src.portfolio_view import ALLOCATED_COLOR, UNMET_COLOR
from src.scenarios import (
    apply_deadline_acceleration,
    apply_effort_increase,
    apply_resource_unavailable,
    compare_scenario,
    take_snapshot,
)

SHOCK_RESOURCE = "A person becomes unavailable"
SHOCK_DEADLINE = "A project's deadline moves earlier"
SHOCK_EFFORT = "A project needs more work"
SHOCK_KINDS = [SHOCK_RESOURCE, SHOCK_DEADLINE, SHOCK_EFFORT]

EPS = 1e-9
BASELINE_COLOR = "#9aa0a6"   # neutral grey: the baseline is context, the shock is the story


# ---------------------------------------------------------------------------
# choices
# ---------------------------------------------------------------------------
def resource_options(data):
    r = data["Resources"]
    out = pd.DataFrame({
        "ResourceID": r["ResourceID"],
        "Label": [f"{rid} {name} ({role})" for rid, name, role in zip(r["ResourceID"], r["ResourceName"], r["Role"])],
    })
    return out.reset_index(drop=True)


def project_options(data):
    p = data["Projects"]
    out = pd.DataFrame({
        "ProjectID": p["ProjectID"],
        "Label": [f"{pid} {name}" for pid, name in zip(p["ProjectID"], p["ProjectName"])],
    })
    return out.reset_index(drop=True)


def calendar_weeks(data):
    weeks = pd.to_numeric(data["Calendar"]["Week"]).astype(int)
    return int(weeks.min()), int(weeks.max())


def deadline_range(data, project_id):
    """(earliest, latest) allowed new deadline for an earlier-deadline shock, or None if it cannot move earlier."""
    p = data["Projects"].set_index("ProjectID").loc[project_id]
    start, current = int(p["StartWeek"]), int(p["DeadlineWeek"])
    if current - 1 < start:
        return None
    return start, current - 1


def resource_preview(data, resource_id, weeks):
    """What the person is planned to do in the chosen weeks: (projects, planned hours)."""
    wanted = set(int(w) for w in weeks)
    a = data["Allocations"]
    mine = a[(a["ResourceID"] == resource_id) & pd.to_numeric(a["Week"]).isin(wanted)]
    return projects_supported_by_resource(data, resource_id, weeks=wanted), float(pd.to_numeric(mine["PlannedHrs"]).sum())


def _name(data, table, key_col, key, name_col):
    t = data[table]
    return t.loc[t[key_col] == key, name_col].iloc[0]


def _weeks_text(weeks):
    weeks = sorted(int(w) for w in weeks)
    return f"week {weeks[0]}" if len(weeks) == 1 else f"weeks {weeks[0]}-{weeks[-1]}"


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------
def apply_shock(data, kind, resource_id=None, weeks=None, project_id=None, new_deadline=None, percent=None):
    """Apply one shock. Returns (scenario_data, plain-words label). Raises ValueError with a clear message."""
    if kind == SHOCK_RESOURCE:
        scenario = apply_resource_unavailable(data, resource_id, list(weeks))
        name = _name(data, "Resources", "ResourceID", resource_id, "ResourceName")
        return scenario, f"{name} ({resource_id}) unavailable, {_weeks_text(weeks)}"
    if kind == SHOCK_DEADLINE:
        scenario = apply_deadline_acceleration(data, project_id, new_deadline)
        name = _name(data, "Projects", "ProjectID", project_id, "ProjectName")
        old = int(_name(data, "Projects", "ProjectID", project_id, "DeadlineWeek"))
        return scenario, f"{name} ({project_id}) deadline moved from week {old} to week {int(new_deadline)}"
    if kind == SHOCK_EFFORT:
        scenario = apply_effort_increase(data, project_id, percent)
        name = _name(data, "Projects", "ProjectID", project_id, "ProjectName")
        return scenario, f"{name} ({project_id}) needs {percent:g}% more work"
    raise ValueError(f"Unknown shock type: {kind!r}")


def run_shock(baseline, kind, **params):
    """Apply a shock and compare it with the baseline. `baseline` is the dict from load_baseline()."""
    scenario, label = apply_shock(baseline["data"], kind, **params)
    scenario_snapshot = take_snapshot(scenario)
    comparison = compare_scenario(baseline["data"], scenario, baseline["snapshot"], scenario_snapshot)
    return {"scenario_data": scenario, "scenario_label": label,
            "scenario_snapshot": scenario_snapshot, "comparison": comparison}


# ---------------------------------------------------------------------------
# reading the result
# ---------------------------------------------------------------------------
TILES = [
    ("Unmet hours", "Unmet hours (staffing gap)", "Required hours nobody is planned to work"),
    ("Reqs with gaps", "Requirements with gaps", "Requirements with at least one unstaffed hour"),
    ("Shortage hours", "Capacity-shortage hours", "Capacity-shortage hours: unmet hours that all qualified people's free time cannot cover"),
    ("Over-allocated", "Over-allocated hours", "Over-allocated hours: hours planned beyond a person's weekly capacity"),
    ("Exposed", "Downstream projects exposed", "Downstream projects exposed: projects that depend on a project with unmet hours"),
]


def metric_tiles(comparison):
    """(label, scenario value, delta vs baseline, help) for the headline tiles."""
    m = comparison["metrics"].set_index("Metric")
    return [(label, float(m.loc[key, "Scenario"]), float(m.loc[key, "Delta"]), help_text)
            for label, key, help_text in TILES]


def shock_story(comparison):
    """The cascade in plain sentences: capacity -> gaps -> projects -> skills -> downstream exposure."""
    m = comparison["metrics"].set_index("Metric")
    unmet_b, unmet_s = m.loc["Unmet hours (staffing gap)", "Baseline"], m.loc["Unmet hours (staffing gap)", "Scenario"]
    reqs, projs = comparison["affected_requirements"], comparison["affected_projects"]
    skills, exposed = comparison["skill_shortages"], comparison["exposed_projects"]

    if reqs.empty and abs(unmet_s - unmet_b) < EPS:
        return ["This shock does not create any new unmet hours in the plan."]

    lines = [f"Unmet hours go from {unmet_b:,.0f}h to {unmet_s:,.0f}h ({unmet_s - unmet_b:+,.0f}h)."]
    if not reqs.empty:
        parts = [f"{r.RequirementID} ({r.ProjectID}, {r.Skill}) +{r.DeltaGapHrs:,.0f}h" for r in reqs.itertuples()]
        lines.append("Requirements that lose coverage: " + "; ".join(parts) + ".")
    if not projs.empty:
        parts = [f"{r.ProjectID} {r.BaselineGapHrs:,.0f}h to {r.ScenarioGapHrs:,.0f}h" for r in projs.itertuples()]
        lines.append("Projects with more unmet hours: " + "; ".join(parts) + ".")
    if not skills.empty:
        parts = [f"{r.Skill} ({r.Requirements}) +{r.ExtraShortfallHrs:,.0f}h" for r in skills.itertuples()]
        lines.append("Skills nobody else is free to cover: " + "; ".join(parts) + ".")
    if not exposed.empty:
        parts = [f"{r.ProjectID} (via {r.Path})" for r in exposed.itertuples()]
        lines.append("Downstream projects exposed: " + "; ".join(parts) + ". A dependency, not a predicted delay.")
    return lines


def before_after_figure(baseline_demand, scenario_demand):
    """Unmet hours per project, baseline next to the shocked plan. Only projects with unmet hours in either."""
    b = baseline_demand.groupby("ProjectID")["GapHrs"].sum()
    s = scenario_demand.groupby("ProjectID")["GapHrs"].sum()
    both = pd.DataFrame({"Baseline": b, "Shocked plan": s}).fillna(0.0)
    both = both[(both > EPS).any(axis=1)].sort_index()
    fig = go.Figure()
    fig.add_bar(x=both.index, y=both["Baseline"], name="Baseline", marker_color=BASELINE_COLOR,
                hovertemplate="%{x}<br>Baseline unmet: %{y:,.0f}h<extra></extra>")
    fig.add_bar(x=both.index, y=both["Shocked plan"], name="After the shock", marker_color=UNMET_COLOR,
                hovertemplate="%{x}<br>After the shock: %{y:,.0f}h<extra></extra>")
    fig.update_layout(
        barmode="group", bargap=0.35, height=320, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"),
        yaxis=dict(title="Unmet hours", gridcolor="rgba(128,128,128,0.2)", zeroline=False), xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


CHANGE_ORDER = {"New": 0, "Worsened": 1, "Improved": 2, "Resolved": 3}


def conflict_change_display(comparison):
    """Conflict changes as a plain, wrap-friendly table."""
    c = comparison["conflict_changes"]
    if c.empty:
        return c
    out = c[["Status", "ConflictType", "RequirementID", "ProjectID", "BaselineHours", "ScenarioHours", "DeltaHours"]].copy()
    for col in ["BaselineHours", "ScenarioHours", "DeltaHours"]:
        out[col] = out[col].round().astype(int)
    return out.rename(columns={"ConflictType": "Type", "RequirementID": "Requirement", "ProjectID": "Project",
                               "BaselineHours": "Before h", "ScenarioHours": "After h", "DeltaHours": "Change h"}).reset_index(drop=True)


def default_resource(data):
    """The person who is the only qualified one for the most requirements (data-derived), else the first person."""
    from src.allocation import find_single_points_of_failure
    spof = find_single_points_of_failure(data)
    if spof.empty:
        return data["Resources"].iloc[0]["ResourceID"]
    return spof.sort_values(["RequirementCount", "ResourceID"], ascending=[False, True]).iloc[0]["ResourceID"]


# ---------------------------------------------------------------------------
# the program-owner view: bottom line, project status, questions for the owner
# ---------------------------------------------------------------------------
def _join(ids):
    ids = list(ids)
    if len(ids) <= 1:
        return "".join(ids)
    return ", ".join(ids[:-1]) + " and " + ids[-1]


def bottom_line(comparison, label):
    """One short paragraph: what happened to the plan. Facts only, no advice."""
    m = comparison["metrics"].set_index("Metric")
    before, after = m.loc["Unmet hours (staffing gap)", "Baseline"], m.loc["Unmet hours (staffing gap)", "Scenario"]
    if abs(after - before) < EPS:
        return f"{label}. This shock does not create any new unmet hours in the plan."
    text = f"{label}. Unmet hours rise from {before:,.0f}h to {after:,.0f}h ({after - before:+,.0f}h)."
    projects = list(comparison["affected_projects"]["ProjectID"])
    if projects:
        text += f" {_join(projects)} {'loses' if len(projects) == 1 else 'lose'} cover."
    exposed = list(dict.fromkeys(comparison["exposed_projects"]["ProjectID"]))
    if exposed:
        text += (f" {_join(exposed)} {'becomes' if len(exposed) == 1 else 'become'} exposed through "
                 f"{'its' if len(exposed) == 1 else 'their'} dependencies.")
    return text


def _dependency_text(data, project_id):
    d = data["Dependencies"]
    after = sorted(d.loc[d["SuccessorID"] == project_id, "PredecessorID"])
    feeds = sorted(d.loc[d["PredecessorID"] == project_id, "SuccessorID"])
    parts = []
    if feeds:
        parts.append("feeds " + ", ".join(feeds))
    if after:
        parts.append("after " + ", ".join(after))
    return "; ".join(parts) if parts else "independent"


def project_status_table(data, baseline_snapshot, scenario_snapshot):
    """One row per project: window, hours, change from the baseline, a status word with its shape, dependencies."""
    from src.dependencies import exposed_project_ids
    from src.theme import project_status

    demand = scenario_snapshot["demand"].groupby("ProjectID")[["RequiredHrs", "GapHrs"]].sum()
    base_gap = baseline_snapshot["demand"].groupby("ProjectID")["GapHrs"].sum()
    gap_projects = sorted(demand.index[demand["GapHrs"] > EPS])
    exposed = set(exposed_project_ids(data, gap_projects))
    rows = []
    for p in data["Projects"].itertuples():
        required = float(demand.loc[p.ProjectID, "RequiredHrs"])
        unmet = float(demand.loc[p.ProjectID, "GapHrs"])
        change = unmet - float(base_gap.get(p.ProjectID, 0.0))
        rows.append({
            "Project": f"{p.ProjectID} {p.ProjectName}",
            "Window": f"Wk {int(p.StartWeek)}-{int(p.DeadlineWeek)}",
            "Required h": f"{required:,.0f}",
            "Unmet h": f"{unmet:,.0f}",
            "Change h": "no change" if abs(change) < EPS else f"{change:+,.0f}",
            "Status": project_status(unmet, required, p.ProjectID in exposed),
            "Dependencies": _dependency_text(data, p.ProjectID),
        })
    return pd.DataFrame(rows)


def owner_questions(baseline_snapshot, scenario_snapshot, comparison, scenario_data=None):
    """Up to three questions the program owner has to answer. Each says what is known; none recommends an answer."""
    reqs, projects = comparison["affected_requirements"], comparison["affected_projects"]
    exposed = list(dict.fromkeys(comparison["exposed_projects"]["ProjectID"]))
    if reqs.empty and projects.empty:
        return []
    out = []
    if not reqs.empty:
        scarcity = baseline_snapshot["scarcity"].set_index("RequirementID")
        worse = reqs[reqs["DeltaGapHrs"] > EPS]
        worse = worse if not worse.empty else reqs
        # Lead with the requirement that has the fewest qualified people; ties go to the largest increase
        worse = worse.assign(_qualified=worse["RequirementID"].map(
            lambda r: int(scarcity.loc[r, "QualifiedCount"]) if r in scarcity.index else 0))
        top = worse.sort_values(["_qualified", "DeltaGapHrs", "RequirementID"], ascending=[True, False, True]).iloc[0]
        n = int(scarcity.loc[top["RequirementID"], "QualifiedCount"]) if top["RequirementID"] in scarcity.index else 0
        qualified, gone = [], []
        if n and scenario_data is not None:
            qualified = [x.strip() for x in str(scarcity.loc[top["RequirementID"], "QualifiedResources"]).split(",") if x.strip()]
            table = scenario_data.get("Unavailability")
            out_ids = set(table["ResourceID"]) if table is not None and not table.empty else set()
            gone = [x for x in qualified if x in out_ids]
        if n == 0:
            detail = f"Nobody meets the minimum level for {top['Skill']}."
        elif gone:
            people = scenario_data["Resources"].set_index("ResourceID")["ResourceName"]
            who = _join([f"{people.get(x, x)} ({x})" for x in gone])
            left = [x for x in qualified if x not in gone]
            if not left:
                detail = (f"{who} {'is' if len(gone) == 1 else 'are'} the only qualified "
                          f"{'person' if len(gone) == 1 else 'people'} for {top['Skill']} and unavailable, "
                          "so anyone else would be below the minimum level.")
            else:
                detail = (f"{len(left)} other {'person meets' if len(left) == 1 else 'people meet'} the minimum level "
                          f"for {top['Skill']} ({', '.join(left)}); {who} {'is' if len(gone) == 1 else 'are'} unavailable.")
        elif n == 1:
            detail = f"Only one person meets the minimum level for {top['Skill']}, so anyone else would be below it."
        else:
            detail = f"{n} people meet the minimum level for {top['Skill']}."
        out.append((f"Who covers {top['RequirementID']} ({top['DeltaGapHrs']:,.0f}h more unmet)?", detail))
    ids = list(projects["ProjectID"])
    if exposed:
        out.append((f"Is a slip on {_join(ids)} acceptable?",
                    f"{_join(exposed)} depend{'s' if len(exposed) == 1 else ''} on "
                    f"{'it' if len(ids) == 1 else 'them'}. Exposure shows what depends on what, not when anything will slip."))
    elif ids:
        out.append((f"Can {_join(ids)} absorb the extra unmet hours?", "No other project depends on them through a dependency."))
    base_demand = baseline_snapshot["demand"]
    base_total = float(base_demand["GapHrs"].sum())
    if base_total > EPS:
        worst = base_demand.sort_values(["GapHrs", "RequirementID"], ascending=[False, True]).iloc[0]
        out.append((f"What about the {base_total:,.0f}h already unmet?",
                    f"The largest is {worst['RequirementID']} ({worst['GapHrs']:,.0f}h on {worst['ProjectID']}). This shock did not cause it."))
    return out

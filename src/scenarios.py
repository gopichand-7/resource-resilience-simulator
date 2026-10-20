"""Steps 19-22: baseline snapshot, the three shocks, and the baseline-vs-scenario comparison.

PURE FUNCTIONS. A shock never changes the baseline: it returns a NEW data dictionary (copies of
the tables), and everything is recalculated from that copy.

    baseline data --shock--> scenario data --take_snapshot--> compare_scenario(baseline, scenario)

The three shocks (frozen specification)
---------------------------------------
1. Resource unavailable   apply_resource_unavailable(data, resource, weeks)
     The resource has zero capacity in those weeks (via an "Unavailability" table) and its planned
     hours in those weeks are removed, so they become unmet demand. Nothing is re-assigned.

2. Deadline acceleration  apply_deadline_acceleration(data, project, new_deadline_week)
     Total effort of each requirement stays constant while its delivery window shortens:
         new weekly demand = original total effort / new number of weeks
     Allocations after the new deadline are removed. All remaining allocations stay exactly as they
     were (they are NOT redistributed), so
         new staffing gap = scenario demand - existing allocation.
     A requirement that starts after the new deadline is compressed into the deadline week.

3. Effort increase        apply_effort_increase(data, project, percent)
     Weekly demand of every requirement of the project grows by `percent`. Allocations are unchanged.

Impact is reported as EXPOSURE, not delay: the model shows which projects lose capacity and which
downstream projects depend on them; it does not predict schedule slips.
"""
import pandas as pd

from src.allocation import (
    CAPACITY_SHORTAGE,
    OVER_ALLOCATION,
    STAFFING_GAP,
    analyze_scarce_skills,
    detect_conflicts,
)
from src.capacity import build_capacity_table, build_requirement_demand, build_requirement_week_table
from src.dependencies import downstream_exposure

EPS = 1e-9

METRIC_NAMES = [
    "Required hours",
    "Allocated hours",
    "Unmet hours (staffing gap)",
    "Requirements with gaps",
    "Projects with gaps",
    "Over-allocated resources",
    "Over-allocated hours",
    "Capacity-shortage requirements",
    "Capacity-shortage hours",
    "Downstream projects exposed",
]
METRIC_COLUMNS = ["Metric", "Baseline", "Scenario", "Delta"]
CONFLICT_CHANGE_COLUMNS = [
    "ConflictType", "ResourceID", "RequirementID", "ProjectID",
    "BaselineHours", "ScenarioHours", "DeltaHours", "Status", "Weeks",
]
AFFECTED_REQUIREMENT_COLUMNS = ["RequirementID", "ProjectID", "Skill", "BaselineGapHrs", "ScenarioGapHrs", "DeltaGapHrs"]
AFFECTED_PROJECT_COLUMNS = ["ProjectID", "BaselineGapHrs", "ScenarioGapHrs", "DeltaGapHrs"]
SKILL_SHORTAGE_COLUMNS = ["Skill", "Requirements", "ExtraShortfallHrs"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _copy(data):
    """Independent copy of every table, so a shock can never touch the baseline."""
    return {name: df.copy(deep=True) for name, df in data.items()}


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _require_project(data, project_id):
    projects = data["Projects"]
    match = projects[projects["ProjectID"] == project_id]
    if match.empty:
        raise ValueError(f"Unknown project: {project_id!r}")
    return match.iloc[0]


def _calendar_weeks(data):
    return {int(w) for w in _num(data["Calendar"]["Week"]).dropna()}


# ---------------------------------------------------------------------------
# Shock 1: resource unavailable
# ---------------------------------------------------------------------------
def apply_resource_unavailable(data, resource_id, weeks):
    """Take one resource out for the given weeks. Returns new data; the input is untouched."""
    if resource_id not in set(data["Resources"]["ResourceID"]):
        raise ValueError(f"Unknown resource: {resource_id!r}")
    weeks = sorted({int(w) for w in weeks})
    if not weeks:
        raise ValueError("Choose at least one week")
    outside = [w for w in weeks if w not in _calendar_weeks(data)]
    if outside:
        raise ValueError(f"Weeks not in the calendar: {outside}")

    scenario = _copy(data)
    alloc = scenario["Allocations"]
    drop = (alloc["ResourceID"] == resource_id) & _num(alloc["Week"]).isin(weeks)
    scenario["Allocations"] = alloc[~drop].reset_index(drop=True)

    new_rows = pd.DataFrame({"ResourceID": resource_id, "Week": weeks})
    existing = scenario.get("Unavailability")
    if existing is None or existing.empty:
        scenario["Unavailability"] = new_rows
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        scenario["Unavailability"] = combined.drop_duplicates().reset_index(drop=True)
    return scenario


# ---------------------------------------------------------------------------
# Shock 2: deadline acceleration
# ---------------------------------------------------------------------------
def apply_deadline_acceleration(data, project_id, new_deadline_week):
    """Move a project's deadline earlier. Effort is constant, the window shrinks, allocations are not redistributed."""
    project = _require_project(data, project_id)
    current = int(_num(pd.Series([project["DeadlineWeek"]])).iloc[0])
    start = int(_num(pd.Series([project["StartWeek"]])).iloc[0])
    new_deadline = int(new_deadline_week)
    if new_deadline >= current:
        raise ValueError(f"New deadline (week {new_deadline}) must be earlier than the current deadline (week {current})")
    if new_deadline < start:
        raise ValueError(f"New deadline (week {new_deadline}) is before the project's start (week {start})")

    scenario = _copy(data)
    projects = scenario["Projects"]
    projects.loc[projects["ProjectID"] == project_id, "DeadlineWeek"] = new_deadline

    req = scenario["Project_Requirements"]
    req["HoursPerWeek"] = _num(req["HoursPerWeek"]).astype(float)
    for idx in req.index[req["ProjectID"] == project_id]:
        s, e = int(req.at[idx, "StartWeek"]), int(req.at[idx, "EndWeek"])
        total_effort = req.at[idx, "HoursPerWeek"] * (e - s + 1)
        new_s, new_e = min(s, new_deadline), min(e, new_deadline)  # starts after the deadline -> deadline week only
        req.at[idx, "StartWeek"] = new_s
        req.at[idx, "EndWeek"] = new_e
        req.at[idx, "HoursPerWeek"] = total_effort / (new_e - new_s + 1)

    alloc = scenario["Allocations"]
    drop = (alloc["ProjectID"] == project_id) & (_num(alloc["Week"]) > new_deadline)
    scenario["Allocations"] = alloc[~drop].reset_index(drop=True)
    return scenario


# ---------------------------------------------------------------------------
# Shock 3: effort increase
# ---------------------------------------------------------------------------
def apply_effort_increase(data, project_id, percent):
    """Raise the weekly demand of every requirement of a project by `percent`. Allocations are unchanged."""
    _require_project(data, project_id)
    if percent < 0:
        raise ValueError("Effort increase must be 0% or more")

    scenario = _copy(data)
    req = scenario["Project_Requirements"]
    req["HoursPerWeek"] = _num(req["HoursPerWeek"]).astype(float)
    mine = req["ProjectID"] == project_id
    req.loc[mine, "HoursPerWeek"] = req.loc[mine, "HoursPerWeek"] * (1 + percent / 100.0)
    return scenario


# ---------------------------------------------------------------------------
# Step 19: baseline snapshot
# ---------------------------------------------------------------------------
def take_snapshot(data):
    """Everything calculated from one data state. Take one from the baseline and one per scenario."""
    capacity = build_capacity_table(data)
    weekly = build_requirement_week_table(data)
    demand = build_requirement_demand(data)
    conflicts = detect_conflicts(data, capacity=capacity, weekly_demand=weekly)
    scarcity = analyze_scarce_skills(data)
    snapshot = {
        "capacity": capacity,
        "weekly_demand": weekly,
        "demand": demand,
        "conflicts": conflicts,
        "scarcity": scarcity,
    }
    snapshot["metrics"] = _metrics(data, snapshot)
    return snapshot


def _projects_with_gaps(demand):
    by_project = demand.groupby("ProjectID")["GapHrs"].sum()
    return sorted(by_project[by_project > EPS].index)


def _metrics(data, snapshot):
    demand, conflicts = snapshot["demand"], snapshot["conflicts"]
    over = conflicts[conflicts["ConflictType"] == OVER_ALLOCATION]
    shortage = conflicts[conflicts["ConflictType"] == CAPACITY_SHORTAGE]
    gap_projects = _projects_with_gaps(demand)
    exposed = downstream_exposure(data, gap_projects)
    exposed_ids = set(exposed["ProjectID"]) - set(gap_projects) if not exposed.empty else set()
    return {
        "Required hours": float(demand["RequiredHrs"].sum()),
        "Allocated hours": float(demand["AllocatedHrs"].sum()),
        "Unmet hours (staffing gap)": float(demand["GapHrs"].sum()),
        "Requirements with gaps": int((demand["GapHrs"] > EPS).sum()),
        "Projects with gaps": len(gap_projects),
        "Over-allocated resources": int(over["ResourceID"].nunique()),
        "Over-allocated hours": float(over["Hours"].sum()),
        "Capacity-shortage requirements": int(shortage["RequirementID"].nunique()),
        "Capacity-shortage hours": float(shortage["Hours"].sum()),
        "Downstream projects exposed": len(exposed_ids),
    }


# ---------------------------------------------------------------------------
# Baseline vs scenario
# ---------------------------------------------------------------------------
def compare_snapshots(baseline, scenario):
    """Metric table: Baseline, Scenario and Delta (scenario minus baseline)."""
    rows = [
        {"Metric": name, "Baseline": baseline["metrics"][name], "Scenario": scenario["metrics"][name],
         "Delta": scenario["metrics"][name] - baseline["metrics"][name]}
        for name in METRIC_NAMES
    ]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _conflict_changes(baseline_conflicts, scenario_conflicts):
    key = ["ConflictType", "ResourceID", "RequirementID"]
    b = baseline_conflicts[key + ["ProjectID", "Weeks", "Hours"]].rename(
        columns={"Hours": "BaselineHours", "ProjectID": "BaseProject", "Weeks": "BaseWeeks"})
    s = scenario_conflicts[key + ["ProjectID", "Weeks", "Hours"]].rename(
        columns={"Hours": "ScenarioHours", "ProjectID": "ScenProject", "Weeks": "ScenWeeks"})
    merged = b.merge(s, on=key, how="outer")

    rows = []
    for r in merged.itertuples():
        base, scen = r.BaselineHours, r.ScenarioHours
        if pd.isna(base):
            status = "New"
        elif pd.isna(scen):
            status = "Resolved"
        elif scen > base + EPS:
            status = "Worsened"
        elif scen < base - EPS:
            status = "Improved"
        else:
            continue
        base_h = 0.0 if pd.isna(base) else float(base)
        scen_h = 0.0 if pd.isna(scen) else float(scen)
        rows.append({
            "ConflictType": r.ConflictType, "ResourceID": r.ResourceID, "RequirementID": r.RequirementID,
            "ProjectID": r.ScenProject if not pd.isna(r.ScenProject) else r.BaseProject,
            "BaselineHours": base_h, "ScenarioHours": scen_h, "DeltaHours": scen_h - base_h, "Status": status,
            "Weeks": r.ScenWeeks if not pd.isna(r.ScenWeeks) else r.BaseWeeks,
        })
    out = pd.DataFrame(rows, columns=CONFLICT_CHANGE_COLUMNS)
    order = {"New": 0, "Worsened": 1, "Improved": 2, "Resolved": 3}
    if out.empty:
        return out
    out["_o"] = out["Status"].map(order)
    return out.sort_values(["_o", "ConflictType", "RequirementID", "ResourceID"]).drop(columns="_o").reset_index(drop=True)


def compare_scenario(baseline_data, scenario_data, baseline_snapshot=None, scenario_snapshot=None):
    """Trace a shock: capacity -> gaps -> affected requirements and projects -> skill shortages -> downstream exposure.

    Returns a dict:
      metrics               Baseline / Scenario / Delta for ten headline numbers
      conflict_changes      conflicts that are New, Worsened, Improved or Resolved
      affected_requirements requirements whose unmet hours increased
      affected_projects     projects whose unmet hours increased
      skill_shortages       skills with a new or worse capacity shortage
      exposed_projects      downstream projects that depend on an affected project
    Pass `baseline_snapshot` / `scenario_snapshot` to avoid recomputing snapshots you already have.
    """
    base = baseline_snapshot or take_snapshot(baseline_data)
    scen = scenario_snapshot or take_snapshot(scenario_data)

    # requirements whose gap grew
    req_info = scenario_data["Project_Requirements"][["RequirementID", "ProjectID", "Skill"]]
    gaps = base["demand"][["RequirementID", "GapHrs"]].rename(columns={"GapHrs": "BaselineGapHrs"}).merge(
        scen["demand"][["RequirementID", "GapHrs"]].rename(columns={"GapHrs": "ScenarioGapHrs"}),
        on="RequirementID", how="outer").fillna(0.0)
    gaps["DeltaGapHrs"] = gaps["ScenarioGapHrs"] - gaps["BaselineGapHrs"]
    affected_requirements = gaps[gaps["DeltaGapHrs"] > EPS].merge(req_info, on="RequirementID", how="left")
    affected_requirements = affected_requirements[AFFECTED_REQUIREMENT_COLUMNS].sort_values(
        "RequirementID").reset_index(drop=True)

    # projects whose gap grew
    by_project = affected_requirements.groupby("ProjectID")[["BaselineGapHrs", "ScenarioGapHrs", "DeltaGapHrs"]].sum()
    project_totals = (
        base["demand"].groupby("ProjectID")["GapHrs"].sum().rename("BaselineGapHrs").to_frame()
        .join(scen["demand"].groupby("ProjectID")["GapHrs"].sum().rename("ScenarioGapHrs"), how="outer").fillna(0.0)
    )
    project_totals["DeltaGapHrs"] = project_totals["ScenarioGapHrs"] - project_totals["BaselineGapHrs"]
    affected_projects = project_totals[project_totals["DeltaGapHrs"] > EPS].reset_index()
    affected_projects = affected_projects.rename(columns={"index": "ProjectID"})[AFFECTED_PROJECT_COLUMNS]
    del by_project

    changes = _conflict_changes(base["conflicts"], scen["conflicts"])

    # skills with a new or worse capacity shortage
    worse = changes[(changes["ConflictType"] == CAPACITY_SHORTAGE) & changes["Status"].isin(["New", "Worsened"])]
    skill_rows = []
    if not worse.empty:
        skills = worse.merge(req_info[["RequirementID", "Skill"]], on="RequirementID", how="left")
        for skill, g in skills.groupby("Skill", sort=True):
            skill_rows.append({"Skill": skill, "Requirements": ", ".join(sorted(g["RequirementID"])),
                               "ExtraShortfallHrs": float(g["DeltaHours"].sum())})
    skill_shortages = pd.DataFrame(skill_rows, columns=SKILL_SHORTAGE_COLUMNS)

    # downstream exposure from the affected projects (excluding projects that are themselves affected)
    affected_ids = list(affected_projects["ProjectID"])
    exposure = downstream_exposure(scenario_data, affected_ids)
    if not exposure.empty:
        exposure = exposure[~exposure["ProjectID"].isin(affected_ids)].reset_index(drop=True)

    return {
        "metrics": compare_snapshots(base, scen),
        "conflict_changes": changes,
        "affected_requirements": affected_requirements,
        "affected_projects": affected_projects,
        "skill_shortages": skill_shortages,
        "exposed_projects": exposure,
    }

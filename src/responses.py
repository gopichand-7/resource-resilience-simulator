"""Response Engine: the moves a planner could make after a shock.

PURE FUNCTIONS, same contract as scenarios.py: each response returns a NEW data dictionary and never
changes its input. The response is then compared with the shocked state using compare_scenario.

    original --shock--> shocked --response--> responded --evaluate_response--> what changed, what it cost

The engine applies the move the PERSON chooses and shows its consequences. It never ranks the
responses and never picks a "best" one: a student-built decision-support prototype shows the
trade-off; the planner decides.

Responses
---------
Substitute        apply_substitute(data, requirement, resource, weeks, hours_per_week=None)
    Bring one more person onto a requirement to cover its UNMET hours. Nobody loses hours. The
    person must have the skill. Being below the minimum proficiency, a role mismatch or going over
    capacity are allowed but reported as warnings, because a planner may knowingly accept them.

Reallocate        apply_reallocate(data, resource, from_requirement, to_requirement, weeks, hours_per_week)
    Move some of one person's existing hours from one requirement to another. Total load for that
    person is unchanged; the donor requirement gets a bigger gap (the problem moves, it is not solved).

Adjust timeline   apply_timeline_extension(data, project, new_deadline_week)
    Move a project's deadline LATER. Requirements that end at the old deadline are stretched to the
    new one with total effort constant, so weekly demand falls. Existing allocations are unchanged,
    so total unmet hours do not fall by themselves: the extension buys room for staffing.

Hard rules (ValueError): unknown ids, weeks outside the requirement window or calendar, a person with
no skill, a person with zero capacity (unavailable) in a chosen week, covering more than the unmet hours,
moving more hours than the person has. Soft issues are warnings, not errors.
"""
import pandas as pd

from src.allocation import (
    BELOW_MINIMUM,
    FLAG_BELOW_PROFICIENCY,
    FLAG_ROLE_MISMATCH,
    NO_SKILL,
    get_resource_qualifications,
)
from src.capacity import build_capacity_table, build_requirement_week_table
from src.scenarios import compare_scenario, take_snapshot

EPS = 1e-9
WARNING_OVER_CAPACITY = "Over capacity"
CHANGE_COLUMNS = ["ResourceID", "RequirementID", "ProjectID", "Week", "BeforeHrs", "AfterHrs", "DeltaHrs"]
WARNING_COLUMNS = ["ResourceID", "RequirementID", "Weeks", "Hours", "Warning"]
NEW_ID_PREFIX = "AX"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _copy(data):
    return {name: df.copy(deep=True) for name, df in data.items()}


def _requirement(data, requirement_id):
    req = data["Project_Requirements"]
    match = req[req["RequirementID"] == requirement_id]
    if match.empty:
        raise ValueError(f"Unknown requirement: {requirement_id!r}")
    return match.iloc[0]


def _check_resource(data, resource_id):
    if resource_id not in set(data["Resources"]["ResourceID"]):
        raise ValueError(f"Unknown resource: {resource_id!r}")


def _weeks_in_window(requirement, weeks):
    weeks = sorted({int(w) for w in weeks})
    if not weeks:
        raise ValueError("Choose at least one week")
    start, end = int(requirement["StartWeek"]), int(requirement["EndWeek"])
    outside = [w for w in weeks if not start <= w <= end]
    if outside:
        raise ValueError(f"Weeks {outside} are outside {requirement['RequirementID']}'s window (weeks {start}-{end})")
    return weeks


def _require_skill(data, resource_id, requirement_id):
    q = get_resource_qualifications(data, requirement_id).set_index("ResourceID")
    if q.loc[resource_id, "Qualification"] == NO_SKILL:
        skill = _requirement(data, requirement_id)["Skill"]
        raise ValueError(f"{resource_id} has no {skill} skill")


def _require_available(data, resource_id, weeks):
    cap = build_capacity_table(data)
    cap = cap[(cap["ResourceID"] == resource_id) & cap["Week"].isin(weeks)]
    off = sorted(int(w) for w in cap.loc[cap["CapacityHrs"] <= 0, "Week"])
    if off:
        raise ValueError(f"{resource_id} is unavailable in week(s) {off}")


def _weekly_gap(data, requirement_id):
    table = build_requirement_week_table(data)
    table = table[table["RequirementID"] == requirement_id]
    return {int(w): float(g) for w, g in zip(table["Week"], table["GapHrs"])}


def _add_hours(alloc, resource_id, project_id, requirement_id, week, hours):
    """Add hours to an existing (resource, requirement, week) row, or append a new row."""
    same = (
        (alloc["ResourceID"] == resource_id)
        & (alloc["RequirementID"] == requirement_id)
        & (pd.to_numeric(alloc["Week"]) == week)
    )
    if same.any():
        alloc["PlannedHrs"] = pd.to_numeric(alloc["PlannedHrs"]).astype(float)       # hours can be fractional
        alloc.loc[same, "PlannedHrs"] = alloc.loc[same, "PlannedHrs"] + hours
        return alloc
    used = [int(i[len(NEW_ID_PREFIX):]) for i in alloc["AllocationID"]
            if str(i).startswith(NEW_ID_PREFIX) and str(i)[len(NEW_ID_PREFIX):].isdigit()]
    new_id = f"{NEW_ID_PREFIX}{(max(used) if used else 0) + 1:03d}"
    row = pd.DataFrame([{"AllocationID": new_id, "ResourceID": resource_id, "ProjectID": project_id,
                         "RequirementID": requirement_id, "Week": week, "PlannedHrs": hours}])
    return pd.concat([alloc, row], ignore_index=True)


# ---------------------------------------------------------------------------
# Response 1: substitute
# ---------------------------------------------------------------------------
def apply_substitute(data, requirement_id, resource_id, weeks, hours_per_week=None):
    """Put another person on a requirement to cover its unmet hours in the chosen weeks.

    hours_per_week=None covers each chosen week's full unmet hours (weeks with nothing unmet are skipped).
    """
    req = _requirement(data, requirement_id)
    _check_resource(data, resource_id)
    weeks = _weeks_in_window(req, weeks)
    _require_skill(data, resource_id, requirement_id)
    _require_available(data, resource_id, weeks)

    gaps = _weekly_gap(data, requirement_id)
    if hours_per_week is not None and hours_per_week <= 0:
        raise ValueError("Hours per week must be more than 0")
    plan = {}
    for week in weeks:
        gap = gaps.get(week, 0.0)
        if hours_per_week is None:
            if gap > EPS:
                plan[week] = gap
        else:
            if hours_per_week > gap + EPS:
                raise ValueError(f"Week {week}: {hours_per_week:g}h is more than the {gap:g}h unmet on {requirement_id}")
            plan[week] = float(hours_per_week)
    if not plan:
        raise ValueError(f"{requirement_id} has no unmet hours in the chosen weeks")

    scenario = _copy(data)
    alloc = scenario["Allocations"]
    for week, hours in plan.items():
        alloc = _add_hours(alloc, resource_id, req["ProjectID"], requirement_id, week, hours)
    scenario["Allocations"] = alloc
    return scenario


# ---------------------------------------------------------------------------
# Response 2: reallocate
# ---------------------------------------------------------------------------
def apply_reallocate(data, resource_id, from_requirement, to_requirement, weeks, hours_per_week):
    """Move part of one person's existing hours from one requirement to another."""
    if from_requirement == to_requirement:
        raise ValueError("Choose two different requirements")
    _requirement(data, from_requirement)
    target = _requirement(data, to_requirement)
    _check_resource(data, resource_id)
    if hours_per_week <= 0:
        raise ValueError("Hours per week must be more than 0")
    weeks = _weeks_in_window(target, weeks)
    _require_skill(data, resource_id, to_requirement)
    _require_available(data, resource_id, weeks)

    gaps = _weekly_gap(data, to_requirement)
    scenario = _copy(data)
    alloc = scenario["Allocations"]

    def held_mask(frame, week):
        return ((frame["ResourceID"] == resource_id) & (frame["RequirementID"] == from_requirement)
                & (pd.to_numeric(frame["Week"]) == week))

    for week in weeks:  # check every week first so a failed request changes nothing
        gap = gaps.get(week, 0.0)
        if hours_per_week > gap + EPS:
            raise ValueError(f"Week {week}: {hours_per_week:g}h is more than the {gap:g}h unmet on {to_requirement}")
        held = float(pd.to_numeric(alloc.loc[held_mask(alloc, week), "PlannedHrs"]).sum())
        if held + EPS < hours_per_week:
            raise ValueError(f"Week {week}: {resource_id} has only {held:g}h on {from_requirement}")

    for week in weeks:
        mask = held_mask(alloc, week)
        alloc["PlannedHrs"] = pd.to_numeric(alloc["PlannedHrs"]).astype(float)
        alloc.loc[mask, "PlannedHrs"] = alloc.loc[mask, "PlannedHrs"] - hours_per_week
        alloc = alloc[pd.to_numeric(alloc["PlannedHrs"]) > EPS].reset_index(drop=True)
        alloc = _add_hours(alloc, resource_id, target["ProjectID"], to_requirement, week, float(hours_per_week))
    scenario["Allocations"] = alloc
    return scenario


# ---------------------------------------------------------------------------
# Response 3: adjust timeline
# ---------------------------------------------------------------------------
def apply_timeline_extension(data, project_id, new_deadline_week):
    """Move a project's deadline later. Effort is constant, windows stretch, allocations are unchanged."""
    projects = data["Projects"]
    match = projects[projects["ProjectID"] == project_id]
    if match.empty:
        raise ValueError(f"Unknown project: {project_id!r}")
    current = int(match.iloc[0]["DeadlineWeek"])
    new_deadline = int(new_deadline_week)
    last_week = int(pd.to_numeric(data["Calendar"]["Week"]).max())
    if new_deadline <= current:
        raise ValueError(f"New deadline (week {new_deadline}) must be later than the current deadline (week {current})")
    if new_deadline > last_week:
        raise ValueError(f"New deadline (week {new_deadline}) is beyond the calendar (last week {last_week})")

    scenario = _copy(data)
    req = scenario["Project_Requirements"]
    req["HoursPerWeek"] = pd.to_numeric(req["HoursPerWeek"]).astype(float)
    stretch = req.index[(req["ProjectID"] == project_id) & (pd.to_numeric(req["EndWeek"]) == current)]
    if len(stretch) == 0:
        raise ValueError(f"No requirement of {project_id} ends at its deadline (week {current})")
    for idx in stretch:
        s, e = int(req.at[idx, "StartWeek"]), int(req.at[idx, "EndWeek"])
        total_effort = req.at[idx, "HoursPerWeek"] * (e - s + 1)
        req.at[idx, "EndWeek"] = new_deadline
        req.at[idx, "HoursPerWeek"] = total_effort / (new_deadline - s + 1)
    scenario["Projects"].loc[scenario["Projects"]["ProjectID"] == project_id, "DeadlineWeek"] = new_deadline
    return scenario


# ---------------------------------------------------------------------------
# What a response changed
# ---------------------------------------------------------------------------
def allocation_changes(before, after):
    """Every (resource, requirement, week) whose planned hours differ between two data states."""
    key = ["ResourceID", "RequirementID", "Week"]

    def prep(data, name):
        a = data["Allocations"][key + ["ProjectID", "PlannedHrs"]].copy()
        a["Week"] = pd.to_numeric(a["Week"]).astype(int)
        a["PlannedHrs"] = pd.to_numeric(a["PlannedHrs"]).astype(float)
        return a.groupby(key, as_index=False).agg(ProjectID=("ProjectID", "first"), **{name: ("PlannedHrs", "sum")})

    merged = prep(before, "BeforeHrs").merge(prep(after, "AfterHrs"), on=key, how="outer", suffixes=("", "_b"))
    merged["ProjectID"] = merged["ProjectID"].fillna(merged.get("ProjectID_b"))
    merged[["BeforeHrs", "AfterHrs"]] = merged[["BeforeHrs", "AfterHrs"]].fillna(0.0)
    merged["DeltaHrs"] = merged["AfterHrs"] - merged["BeforeHrs"]
    out = merged[merged["DeltaHrs"].abs() > EPS][CHANGE_COLUMNS]
    return out.sort_values(["RequirementID", "ResourceID", "Week"]).reset_index(drop=True)


def assignment_warnings(before, after):
    """Soft issues with the people who GAINED hours: below minimum proficiency, role mismatch, over capacity."""
    changes = allocation_changes(before, after)
    gained = changes[changes["DeltaHrs"] > EPS]
    if gained.empty:
        return pd.DataFrame(columns=WARNING_COLUMNS)

    capacity = build_capacity_table(after)
    over = capacity[capacity["PlannedHrs"] > capacity["CapacityHrs"] + EPS].copy()
    over["Excess"] = over["PlannedHrs"] - over["CapacityHrs"]
    roles = after["Resources"].set_index("ResourceID")["Role"]

    rows = []
    for (resource, requirement), g in gained.groupby(["ResourceID", "RequirementID"]):
        weeks = sorted(g["Week"])
        hours = float(g["DeltaHrs"].sum())
        qual = get_resource_qualifications(after, requirement).set_index("ResourceID").loc[resource, "Qualification"]
        if qual == BELOW_MINIMUM:
            rows.append((resource, requirement, weeks, hours, FLAG_BELOW_PROFICIENCY))
        if roles[resource] != _requirement(after, requirement)["Role"]:
            rows.append((resource, requirement, weeks, hours, FLAG_ROLE_MISMATCH))

    # over capacity is a property of the PERSON, so it is reported once per person (never once per requirement)
    for resource, g in gained.groupby("ResourceID"):
        hit = over[(over["ResourceID"] == resource) & over["Week"].isin(set(g["Week"]))]
        if not hit.empty:
            rows.append((resource, ", ".join(sorted(set(g["RequirementID"]))), sorted(int(w) for w in hit["Week"]),
                         float(hit["Excess"].sum()), WARNING_OVER_CAPACITY))
    out = pd.DataFrame(rows, columns=WARNING_COLUMNS)
    if not out.empty:
        out["Weeks"] = out["Weeks"].apply(lambda w: ", ".join(str(x) for x in w))
    return out


def evaluate_response(original_data, shocked_data, responded_data):
    """What a response did, measured two ways. No ranking and no recommendation.

    vs_shock       responded compared with the shocked state: what the move recovered or broke
    vs_baseline    responded compared with the original plan: what is still not back to normal
    allocation_changes   the exact hours that moved
    warnings       soft issues (below minimum proficiency, role mismatch, over capacity)
    """
    return {
        "vs_shock": compare_scenario(shocked_data, responded_data),
        "vs_baseline": compare_scenario(original_data, responded_data, take_snapshot(original_data)),
        "allocation_changes": allocation_changes(shocked_data, responded_data),
        "warnings": assignment_warnings(shocked_data, responded_data),
    }

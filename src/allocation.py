"""Steps 9-14: qualification, candidate evaluation, fit score, constraint explanations,
conflict detection and scarce-skill analysis.

For ONE project requirement this module answers: "which resources could support it, and why?"

Order of evaluation (frozen specification):

    1. Does the resource have the required skill at all?
         no  -> NOT a candidate (excluded)
    2. Does its proficiency meet the requirement's minimum?
         no  -> candidate, flagged "Below required proficiency", fit label capped at Partial
         yes -> normal scoring
    3. Score three dimensions separately, then combine them transparently.

Dimensions (each 0-1), evaluated over the WHOLE requirement window:

    SkillMatch     min(proficiency / minimum proficiency, 1)
    Availability   min(tightest free hours / required hours per week, 1)
                   "tightest" = the week in the window where the resource has the fewest free hours
                   (so 2h free in week 4 is not hidden by 14h free in week 9)
    LoadHeadroom   max(0, 1 - peak utilization in the window)

Fit score = 50*SkillMatch + 30*Availability + 20*LoadHeadroom   (0-100)

    75 or more -> Strong      50 to 74.99 -> Partial      below 50 -> Constraint

IMPORTANT: the weights and thresholds below are simulation assumptions for this project.
They are NOT industry standards, and they can be changed through the `weights` argument.

Free hours are measured against the resource's whole baseline plan, so a resource already
assigned to this same requirement is shown with that assignment counted in its load.

Role mismatch is a soft flag only: it never removes a candidate.
Every function is pure: inputs are never modified.
"""
import pandas as pd

from src.capacity import build_capacity_table, build_requirement_demand, build_requirement_week_table

# ---- simulation assumptions (documented, not industry standards) ----
FIT_WEIGHTS = {"skill": 50, "availability": 30, "headroom": 20}
STRONG_FIT_MIN = 75
PARTIAL_FIT_MIN = 50

# ---- qualification statuses ----
QUALIFIED = "Qualified"
BELOW_MINIMUM = "Below minimum"
NO_SKILL = "No skill"

# ---- fit labels ----
STRONG = "Strong"
PARTIAL = "Partial"
CONSTRAINT = "Constraint"

# ---- constraint explanations ----
FLAG_INSUFFICIENT_HOURS = "Insufficient hours"
FLAG_NO_AVAILABILITY = "No availability"
FLAG_FULLY_ALLOCATED = "Already fully allocated"
FLAG_BELOW_PROFICIENCY = "Below required proficiency"
FLAG_ROLE_MISMATCH = "Role mismatch"
FLAG_UNAVAILABLE = "Unavailable in part of the window"

QUALIFICATION_COLUMNS = ["ResourceID", "ResourceName", "Role", "Proficiency", "MinProficiency", "Qualification"]
CANDIDATE_COLUMNS = [
    "ResourceID", "ResourceName", "Role", "Proficiency", "SkillMatch", "RequiredHrsPerWeek",
    "TightestFreeHrs", "Availability", "PeakUtilization", "LoadHeadroom",
    "FitScore", "FitLabel", "Constraints",
]


def _get_requirement(data, requirement_id):
    req = data["Project_Requirements"]
    match = req[req["RequirementID"] == requirement_id]
    if match.empty:
        raise ValueError(f"Unknown requirement: {requirement_id!r}")
    row = match.iloc[0]
    return {
        "RequirementID": row["RequirementID"],
        "ProjectID": row["ProjectID"],
        "Role": row["Role"],
        "Skill": row["Skill"],
        "MinProficiency": float(pd.to_numeric(row["MinProficiency"])),
        "HoursPerWeek": float(pd.to_numeric(row["HoursPerWeek"])),
        "StartWeek": int(pd.to_numeric(row["StartWeek"])),
        "EndWeek": int(pd.to_numeric(row["EndWeek"])),
    }


def classify_proficiency(proficiency, min_proficiency):
    """Step 9 rule for one resource: NO_SKILL, BELOW_MINIMUM or QUALIFIED."""
    if proficiency is None or pd.isna(proficiency) or proficiency <= 0:
        return NO_SKILL
    return QUALIFIED if proficiency >= min_proficiency else BELOW_MINIMUM


def get_resource_qualifications(data, requirement_id):
    """Step 9: every resource with its proficiency in the requirement's skill and its qualification."""
    req = _get_requirement(data, requirement_id)
    resources = data["Resources"][["ResourceID", "ResourceName", "Role"]].copy()

    skills = data["Resource_Skills"].copy()
    skills = skills[skills["Skill"] == req["Skill"]]
    proficiency = pd.to_numeric(skills["Proficiency"], errors="coerce")
    by_resource = dict(zip(skills["ResourceID"], proficiency))

    resources["Proficiency"] = resources["ResourceID"].map(by_resource).fillna(0).astype(float)
    resources["MinProficiency"] = req["MinProficiency"]
    resources["Qualification"] = [
        classify_proficiency(p, req["MinProficiency"]) for p in resources["Proficiency"]
    ]
    return resources[QUALIFICATION_COLUMNS].reset_index(drop=True)


def get_qualified_resources(data, requirement_id):
    """Resource IDs that meet the requirement's skill AND minimum proficiency."""
    q = get_resource_qualifications(data, requirement_id)
    return sorted(q.loc[q["Qualification"] == QUALIFIED, "ResourceID"])


def fit_label(score, qualification=QUALIFIED):
    """Strong / Partial / Constraint from a fit score. Below-minimum candidates can never be Strong."""
    if score >= STRONG_FIT_MIN:
        label = STRONG
    elif score >= PARTIAL_FIT_MIN:
        label = PARTIAL
    else:
        label = CONSTRAINT
    if qualification == BELOW_MINIMUM and label == STRONG:
        label = PARTIAL
    return label


def constraint_flags(tightest_free, max_free, required_hrs, qualification, role_matches, unavailable=False):
    """Step 12: plain-language reasons a candidate is constrained. The hours flags never overlap."""
    flags = []
    if qualification == BELOW_MINIMUM:
        flags.append(FLAG_BELOW_PROFICIENCY)
    if unavailable:
        flags.append(FLAG_UNAVAILABLE)  # a scenario has taken the resource out of some weeks
    elif tightest_free <= 0:
        flags.append(FLAG_NO_AVAILABILITY if max_free <= 0 else FLAG_FULLY_ALLOCATED)
    elif tightest_free < required_hrs:
        flags.append(FLAG_INSUFFICIENT_HOURS)
    if not role_matches:
        flags.append(FLAG_ROLE_MISMATCH)
    return flags


def _check_weights(weights):
    weights = dict(FIT_WEIGHTS if weights is None else weights)
    if set(weights) != set(FIT_WEIGHTS):
        raise ValueError(f"weights must have exactly these keys: {sorted(FIT_WEIGHTS)}")
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    return weights


def evaluate_candidates(data, requirement_id, capacity=None, weights=None):
    """Steps 10-12: rank every eligible resource for one requirement.

    Returns one row per candidate (resources with no skill are excluded), best fit first,
    with each dimension shown separately so the ranking can be explained.
    Pass `capacity` (from build_capacity_table) to avoid recomputing it for every requirement.
    """
    weights = _check_weights(weights)
    req = _get_requirement(data, requirement_id)
    qualifications = get_resource_qualifications(data, requirement_id)
    candidates = qualifications[qualifications["Qualification"] != NO_SKILL].copy()

    if capacity is None:
        capacity = build_capacity_table(data)
    window = capacity[(capacity["Week"] >= req["StartWeek"]) & (capacity["Week"] <= req["EndWeek"])]
    grouped = window.groupby("ResourceID")
    tightest_free = grouped["AvailableHrs"].min()
    max_free = grouped["AvailableHrs"].max()
    peak_util = grouped["Utilization"].max()
    min_capacity = grouped["CapacityHrs"].min()

    rows = []
    for c in candidates.itertuples():
        free = float(tightest_free.get(c.ResourceID, 0.0))
        best_week_free = float(max_free.get(c.ResourceID, 0.0))
        peak = peak_util.get(c.ResourceID)
        peak = 1.0 if peak is None or pd.isna(peak) else float(peak)  # zero-capacity resource counts as full

        skill_match = min(c.Proficiency / req["MinProficiency"], 1.0)
        if req["HoursPerWeek"] > 0:
            availability = min(max(free, 0.0) / req["HoursPerWeek"], 1.0)
        else:
            availability = 1.0
        headroom = max(0.0, 1.0 - peak)

        score = (
            weights["skill"] * skill_match
            + weights["availability"] * availability
            + weights["headroom"] * headroom
        )
        flags = constraint_flags(
            free, best_week_free, req["HoursPerWeek"], c.Qualification, c.Role == req["Role"],
            unavailable=float(min_capacity.get(c.ResourceID, 1.0)) <= 0,
        )
        rows.append({
            "ResourceID": c.ResourceID,
            "ResourceName": c.ResourceName,
            "Role": c.Role,
            "Proficiency": c.Proficiency,
            "SkillMatch": skill_match,
            "RequiredHrsPerWeek": req["HoursPerWeek"],
            "TightestFreeHrs": free,
            "Availability": availability,
            "PeakUtilization": peak,
            "LoadHeadroom": headroom,
            "FitScore": score,
            "FitLabel": fit_label(score, c.Qualification),
            "Constraints": "; ".join(flags),
        })

    result = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    result = result.sort_values(["FitScore", "ResourceID"], ascending=[False, True])
    return result.reset_index(drop=True)


# ===========================================================================
# Step 13: conflict detection
#
# Nothing here is stored in the workbook. Every conflict is discovered from the raw
# allocations, requirements and skills.
#
#   Over-allocation         a resource's planned hours exceed its weekly capacity
#   Staffing gap            a requirement has unmet hours (allocated < required)
#   Capacity shortage       in a week, a requirement's unmet hours exceed the TOTAL free hours of
#                           every qualified resource, so nobody qualified could close the gap
#                           even using all of their free time
#   Competing requirements  for a capacity shortage: the other requirements that currently hold
#                           the hours of the qualified resources in those weeks
# ===========================================================================
OVER_ALLOCATION = "Over-allocation"
STAFFING_GAP = "Staffing gap"
CAPACITY_SHORTAGE = "Capacity shortage"
COMPETING_REQUIREMENTS = "Competing requirements"

CONFLICT_COLUMNS = ["ConflictType", "ResourceID", "RequirementID", "ProjectID", "Weeks", "Hours", "Detail"]


def format_weeks(weeks):
    """[4, 5, 6, 8] -> '4-6, 8'"""
    weeks = sorted({int(w) for w in weeks})
    if not weeks:
        return ""
    parts, start, prev = [], weeks[0], weeks[0]
    for w in weeks[1:] + [None]:
        if w is not None and w == prev + 1:
            prev = w
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if w is not None:
            start = prev = w
    return ", ".join(parts)


def _num(x):
    return f"{x:g}"


def detect_conflicts(data, capacity=None, weekly_demand=None):
    """Return one row per discovered conflict. An empty table means the baseline has none."""
    if capacity is None:
        capacity = build_capacity_table(data)
    if weekly_demand is None:
        weekly_demand = build_requirement_week_table(data)

    rows = []

    # ---- over-allocation (per resource) ----
    over = capacity[capacity["AvailableHrs"] < 0]
    for resource, g in over.groupby("ResourceID", sort=True):
        excess = -g["AvailableHrs"]
        rows.append({
            "ConflictType": OVER_ALLOCATION, "ResourceID": resource, "RequirementID": "", "ProjectID": "",
            "Weeks": format_weeks(g["Week"]), "Hours": float(excess.sum()),
            "Detail": f"Planned hours exceed capacity by up to {_num(excess.max())}h in a week",
        })

    # ---- staffing gaps (per requirement) ----
    gaps = weekly_demand[weekly_demand["GapHrs"] > 0]
    for requirement, g in gaps.groupby("RequirementID", sort=True):
        required = weekly_demand.loc[weekly_demand["RequirementID"] == requirement, "RequiredHrs"].sum()
        rows.append({
            "ConflictType": STAFFING_GAP, "ResourceID": "", "RequirementID": requirement,
            "ProjectID": g["ProjectID"].iloc[0], "Weeks": format_weeks(g["Week"]),
            "Hours": float(g["GapHrs"].sum()),
            "Detail": f"{_num(g['GapHrs'].sum())}h of {_num(required)}h required hours are unstaffed",
        })

    # ---- capacity shortages and the requirements competing for the same people ----
    alloc = data["Allocations"][["ResourceID", "RequirementID", "ProjectID", "Week", "PlannedHrs"]].copy()
    alloc["Week"] = pd.to_numeric(alloc["Week"], errors="coerce")
    alloc["PlannedHrs"] = pd.to_numeric(alloc["PlannedHrs"], errors="coerce")
    free = capacity.set_index(["ResourceID", "Week"])["AvailableHrs"]

    for requirement, g in gaps.groupby("RequirementID", sort=True):
        qualified = get_qualified_resources(data, requirement)
        short_weeks, short_hours = [], 0.0
        for week, unmet in zip(g["Week"], g["GapHrs"]):
            pool_free = sum(max(0.0, float(free.get((r, int(week)), 0.0))) for r in qualified)
            if unmet > pool_free:
                short_weeks.append(int(week))
                short_hours += float(unmet - pool_free)
        if not short_weeks:
            continue
        project = g["ProjectID"].iloc[0]
        who = ", ".join(qualified) if qualified else "nobody"
        rows.append({
            "ConflictType": CAPACITY_SHORTAGE, "ResourceID": "", "RequirementID": requirement,
            "ProjectID": project, "Weeks": format_weeks(short_weeks), "Hours": short_hours,
            "Detail": f"Unmet hours exceed all free hours of qualified resources ({who}) by {_num(short_hours)}h in total",
        })

        holders = alloc[
            alloc["ResourceID"].isin(qualified)
            & alloc["Week"].isin(short_weeks)
            & (alloc["RequirementID"] != requirement)
        ]
        if not holders.empty:
            by_req = holders.groupby(["RequirementID", "ProjectID"])["PlannedHrs"].sum().sort_index()
            held = "; ".join(f"{rq} ({pj}) {_num(h)}h" for (rq, pj), h in by_req.items())
            rows.append({
                "ConflictType": COMPETING_REQUIREMENTS, "ResourceID": "", "RequirementID": requirement,
                "ProjectID": project, "Weeks": format_weeks(short_weeks), "Hours": float(by_req.sum()),
                "Detail": f"Hours of qualified resources are already held by: {held}",
            })

    return pd.DataFrame(rows, columns=CONFLICT_COLUMNS)


# ===========================================================================
# Step 14: scarce skills and single points of failure
#
# For every requirement of a project that is not Completed ("active" = Active or Planned),
# count how many resources are QUALIFIED (skill present AND proficiency at the minimum).
#
#   0 qualified   No qualified resource
#   1 qualified   Single point of failure
#   2 qualified   Limited backup
#   3 or more     Broader pool           (the 3 is a simulation assumption, not a standard)
# ===========================================================================
NO_QUALIFIED = "No qualified resource"
SINGLE_POINT = "Single point of failure"
LIMITED_BACKUP = "Limited backup"
BROADER_POOL = "Broader pool"
BROADER_POOL_MIN = 3

SCARCITY_COLUMNS = [
    "RequirementID", "ProjectID", "Skill", "MinProficiency", "QualifiedCount", "QualifiedResources", "Category",
]
SPOF_COLUMNS = ["ResourceID", "ResourceName", "RequirementCount", "Requirements", "Projects"]


def scarcity_category(qualified_count):
    if qualified_count <= 0:
        return NO_QUALIFIED
    if qualified_count == 1:
        return SINGLE_POINT
    if qualified_count < BROADER_POOL_MIN:
        return LIMITED_BACKUP
    return BROADER_POOL


def analyze_scarce_skills(data):
    """One row per requirement of a non-completed project, with its qualified pool size and category."""
    projects = data["Projects"]
    active = set(projects.loc[projects["Status"] != "Completed", "ProjectID"])
    requirements = data["Project_Requirements"]
    requirements = requirements[requirements["ProjectID"].isin(active)]

    rows = []
    for r in requirements.itertuples():
        qualified = get_qualified_resources(data, r.RequirementID)
        rows.append({
            "RequirementID": r.RequirementID, "ProjectID": r.ProjectID, "Skill": r.Skill,
            "MinProficiency": r.MinProficiency, "QualifiedCount": len(qualified),
            "QualifiedResources": ", ".join(qualified), "Category": scarcity_category(len(qualified)),
        })
    return pd.DataFrame(rows, columns=SCARCITY_COLUMNS).reset_index(drop=True)


def find_single_points_of_failure(data, scarcity=None):
    """Resources that are the ONLY qualified person for at least one active requirement."""
    if scarcity is None:
        scarcity = analyze_scarce_skills(data)
    sole = scarcity[scarcity["QualifiedCount"] == 1]
    names = data["Resources"].set_index("ResourceID")["ResourceName"]
    rows = []
    for resource, g in sole.groupby("QualifiedResources", sort=True):
        rows.append({
            "ResourceID": resource, "ResourceName": names.get(resource, ""),
            "RequirementCount": len(g), "Requirements": ", ".join(sorted(g["RequirementID"])),
            "Projects": ", ".join(sorted(set(g["ProjectID"]))),
        })
    return pd.DataFrame(rows, columns=SPOF_COLUMNS).reset_index(drop=True)

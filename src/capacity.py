"""Steps 7-8: resource capacity and requirement demand calculations.

For every resource and every calendar week this module calculates:

    CapacityHrs    hours the resource has available that week (Resources.WeeklyCapacityHrs)
    PlannedHrs     sum of that resource's allocations that week
    AvailableHrs   CapacityHrs - PlannedHrs  (negative means the resource is over-allocated)
    Utilization    PlannedHrs / CapacityHrs  (0.95 = 95%; NaN if CapacityHrs is 0)
    LoadHeadroom   max(0, 1 - Utilization)   (never negative; 0 if CapacityHrs is 0)

An optional "Unavailability" table (ResourceID, Week), added only by scenarios, sets CapacityHrs to 0
for those resource-weeks.

Precondition: the data has already passed `validator.validate_data`. Numeric columns
are converted once here, so callers never have to re-coerce them.
The function is pure: it returns a new DataFrame and never modifies its input.
"""
import pandas as pd

CAPACITY_COLUMNS = [
    "ResourceID", "Week", "CapacityHrs", "PlannedHrs", "AvailableHrs", "Utilization", "LoadHeadroom",
]


def build_capacity_table(data):
    """One row per resource per calendar week (including weeks with no allocation)."""
    resources = data["Resources"][["ResourceID", "WeeklyCapacityHrs"]].copy()
    resources["CapacityHrs"] = pd.to_numeric(resources["WeeklyCapacityHrs"], errors="coerce").astype(float)
    resources = resources[["ResourceID", "CapacityHrs"]]

    weeks = pd.to_numeric(data["Calendar"]["Week"], errors="coerce").dropna().astype(int)
    weeks = pd.DataFrame({"Week": sorted(weeks.unique())})

    alloc = data["Allocations"][["ResourceID", "Week", "PlannedHrs"]].copy()
    alloc["Week"] = pd.to_numeric(alloc["Week"], errors="coerce")
    alloc["PlannedHrs"] = pd.to_numeric(alloc["PlannedHrs"], errors="coerce")
    alloc = alloc.dropna(subset=["Week", "PlannedHrs"]).astype({"Week": int})
    planned = alloc.groupby(["ResourceID", "Week"], as_index=False)["PlannedHrs"].sum()

    table = resources.merge(weeks, how="cross")
    table = table.merge(planned, on=["ResourceID", "Week"], how="left")
    table["PlannedHrs"] = table["PlannedHrs"].fillna(0.0).astype(float)

    # Optional scenario overlay: weeks in which a resource is unavailable have zero capacity.
    # The baseline workbook has no such table, so the baseline is unaffected.
    unavailable = data.get("Unavailability")
    if unavailable is not None and len(unavailable):
        weeks_off = pd.to_numeric(unavailable["Week"], errors="coerce")
        keys = set(zip(unavailable["ResourceID"], weeks_off))
        off = [(r, w) in keys for r, w in zip(table["ResourceID"], table["Week"])]
        table.loc[off, "CapacityHrs"] = 0.0

    table["AvailableHrs"] = table["CapacityHrs"] - table["PlannedHrs"]
    table["Utilization"] = table["PlannedHrs"].div(table["CapacityHrs"].where(table["CapacityHrs"] > 0))
    table["LoadHeadroom"] = (1 - table["Utilization"]).clip(lower=0).fillna(0.0)

    return table[CAPACITY_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 8: requirement demand and staffing gap
#
#   RequiredHrs   HoursPerWeek for each week inside the requirement window (else not listed)
#   AllocatedHrs  sum of allocations to that requirement that week
#   GapHrs        max(0, RequiredHrs - AllocatedHrs)   (unmet demand, never negative)
#   Coverage      AllocatedHrs / RequiredHrs           (1.0 = fully staffed; can exceed 1.0
#                                                       if over-staffed; NaN if nothing required)
# No threshold is applied here: "understaffed" is a judgement made by later steps.
# ---------------------------------------------------------------------------
DEMAND_WEEK_COLUMNS = ["RequirementID", "ProjectID", "Week", "RequiredHrs", "AllocatedHrs", "GapHrs", "Coverage"]
DEMAND_COLUMNS = ["RequirementID", "ProjectID", "RequiredHrs", "AllocatedHrs", "GapHrs", "Coverage"]


def _add_gap_and_coverage(df):
    df["GapHrs"] = (df["RequiredHrs"] - df["AllocatedHrs"]).clip(lower=0)
    df["Coverage"] = df["AllocatedHrs"].div(df["RequiredHrs"].where(df["RequiredHrs"] > 0))
    return df


def build_requirement_week_table(data):
    """One row per requirement per week inside its window."""
    req = data["Project_Requirements"][
        ["RequirementID", "ProjectID", "HoursPerWeek", "StartWeek", "EndWeek"]
    ].copy()
    for col in ("HoursPerWeek", "StartWeek", "EndWeek"):
        req[col] = pd.to_numeric(req[col], errors="coerce")
    req = req.dropna(subset=["HoursPerWeek", "StartWeek", "EndWeek"])

    rows = [
        (r.RequirementID, r.ProjectID, week, float(r.HoursPerWeek))
        for r in req.itertuples()
        for week in range(int(r.StartWeek), int(r.EndWeek) + 1)
    ]
    table = pd.DataFrame(rows, columns=["RequirementID", "ProjectID", "Week", "RequiredHrs"])

    alloc = data["Allocations"][["RequirementID", "Week", "PlannedHrs"]].copy()
    alloc["Week"] = pd.to_numeric(alloc["Week"], errors="coerce")
    alloc["PlannedHrs"] = pd.to_numeric(alloc["PlannedHrs"], errors="coerce")
    alloc = alloc.dropna(subset=["Week", "PlannedHrs"]).astype({"Week": int})
    allocated = alloc.groupby(["RequirementID", "Week"], as_index=False)["PlannedHrs"].sum()
    allocated = allocated.rename(columns={"PlannedHrs": "AllocatedHrs"})

    table = table.merge(allocated, on=["RequirementID", "Week"], how="left")
    table["AllocatedHrs"] = table["AllocatedHrs"].fillna(0.0).astype(float)
    return _add_gap_and_coverage(table)[DEMAND_WEEK_COLUMNS].reset_index(drop=True)


def build_requirement_demand(data):
    """One row per requirement: total required, allocated, gap and coverage over its whole window."""
    weekly = build_requirement_week_table(data)
    totals = weekly.groupby(["RequirementID", "ProjectID"], as_index=False, sort=False)[
        ["RequiredHrs", "AllocatedHrs", "GapHrs"]
    ].sum()
    totals["Coverage"] = totals["AllocatedHrs"].div(totals["RequiredHrs"].where(totals["RequiredHrs"] > 0))
    return totals[DEMAND_COLUMNS].reset_index(drop=True)

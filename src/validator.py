"""Step 5: validate the seven loaded DataFrames.

Pure validation: no business calculations, nothing is modified.
`validate_data(data)` returns a list of human-readable problems (all of them,
not just the first). An empty list means the dataset is valid.

Conflicts, gaps, utilization and fit are NOT validated here. They are discovered
later by the engine, so an over-allocated resource is not a validation error.
"""
import math
from collections import defaultdict

import pandas as pd

# Every column listed here is mandatory and may not be blank.
# (Projects.Description is documentation and is optional.)
REQUIRED_COLUMNS = {
    "Resources": ["ResourceID", "ResourceName", "Role", "WeeklyCapacityHrs"],
    "Resource_Skills": ["ResourceID", "Skill", "Proficiency"],
    "Projects": ["ProjectID", "ProjectName", "BusinessPriority", "StartWeek", "DeadlineWeek", "Status"],
    "Project_Requirements": [
        "RequirementID", "ProjectID", "Role", "Skill", "MinProficiency",
        "HoursPerWeek", "StartWeek", "EndWeek",
    ],
    "Allocations": ["AllocationID", "ResourceID", "ProjectID", "RequirementID", "Week", "PlannedHrs"],
    "Calendar": ["Week", "WeekStartDate"],
    "Dependencies": ["DependencyID", "PredecessorID", "SuccessorID", "Type"],
}

# Columns that must be unique (single IDs and composite keys).
UNIQUE_KEYS = [
    ("Resources", ["ResourceID"]),
    ("Projects", ["ProjectID"]),
    ("Project_Requirements", ["RequirementID"]),
    ("Allocations", ["AllocationID"]),
    ("Calendar", ["Week"]),
    ("Dependencies", ["DependencyID"]),
    ("Resource_Skills", ["ResourceID", "Skill"]),
    ("Allocations", ["ResourceID", "RequirementID", "Week"]),
]

# (table, column, minimum, maximum, whole number required)
RANGE_RULES = [
    ("Resources", "WeeklyCapacityHrs", 0, None, False),
    ("Resource_Skills", "Proficiency", 1, 5, True),
    ("Project_Requirements", "MinProficiency", 1, 5, True),
    ("Project_Requirements", "HoursPerWeek", 0, None, False),
    ("Allocations", "PlannedHrs", 0, None, False),
    ("Calendar", "Week", 1, None, True),
]

# Columns holding a week number that must exist in Calendar.
WEEK_COLUMNS = [
    ("Projects", "StartWeek"),
    ("Projects", "DeadlineWeek"),
    ("Project_Requirements", "StartWeek"),
    ("Project_Requirements", "EndWeek"),
    ("Allocations", "Week"),
]

# (child table, child column, parent table, parent column)
FOREIGN_KEYS = [
    ("Resource_Skills", "ResourceID", "Resources", "ResourceID"),
    ("Project_Requirements", "ProjectID", "Projects", "ProjectID"),
    ("Allocations", "ResourceID", "Resources", "ResourceID"),
    ("Allocations", "ProjectID", "Projects", "ProjectID"),
    ("Allocations", "RequirementID", "Project_Requirements", "RequirementID"),
    ("Dependencies", "PredecessorID", "Projects", "ProjectID"),
    ("Dependencies", "SuccessorID", "Projects", "ProjectID"),
]

ID_COLUMN = {
    "Resources": "ResourceID",
    "Projects": "ProjectID",
    "Project_Requirements": "RequirementID",
    "Allocations": "AllocationID",
    "Dependencies": "DependencyID",
}


def _fmt(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_blank(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def _label(table, df, i):
    """'Allocations row 5 (AllocationID A0004)'. Row numbers match Excel (header = row 1)."""
    text = f"{table} row {i + 2}"
    id_col = ID_COLUMN.get(table)
    if id_col in df.columns and not _is_blank(df[id_col])[i]:
        text += f" ({id_col} {_fmt(df[id_col][i])})"
    return text


def validate_data(data):
    """Return a list of every validation problem found. [] means the dataset is valid."""
    errors = []

    # ---- tables present ----
    tables = {}
    for name in REQUIRED_COLUMNS:
        if name not in data:
            errors.append(f"Missing table: {name}")
        elif not isinstance(data[name], pd.DataFrame):
            errors.append(f"{name}: expected a DataFrame")
        else:
            tables[name] = data[name].reset_index(drop=True)  # copy; input is never modified

    def get(table, *cols):
        """Table if it exists and has all the given columns, else None (dependent checks are skipped)."""
        df = tables.get(table)
        if df is not None and all(c in df.columns for c in cols):
            return df
        return None

    # ---- 1. required columns ----
    for table, cols in REQUIRED_COLUMNS.items():
        if table in tables:
            missing = [c for c in cols if c not in tables[table].columns]
            if missing:
                errors.append(f"{table}: missing required column(s): {', '.join(missing)}")

    # ---- 2. duplicate IDs ----
    for table, cols in UNIQUE_KEYS:
        df = get(table, *cols)
        if df is None:
            continue
        keys = df[cols].dropna()
        dups = keys[keys.duplicated(keep=False)]
        for key, group in dups.groupby(cols):
            key = key if isinstance(key, tuple) else (key,)
            where = ", ".join(str(i + 2) for i in group.index)
            label = ", ".join(f"{c} {_fmt(v)}" for c, v in zip(cols, key))
            errors.append(f"{table}: duplicate {label} (rows {where})")

    # ---- 3. missing values ----
    for table, cols in REQUIRED_COLUMNS.items():
        df = tables.get(table)
        if df is None:
            continue
        for col in cols:
            if col in df.columns:
                for i in df.index[_is_blank(df[col])]:
                    errors.append(f"{_label(table, df, i)}: required value '{col}' is blank")

    # ---- numeric conversion (once per column) ----
    num = {}
    numeric_cols = {(t, c) for t, c, *_ in RANGE_RULES} | set(WEEK_COLUMNS)
    for table, col in sorted(numeric_cols):
        df = get(table, col)
        if df is None:
            continue
        raw = df[col]
        num[(table, col)] = pd.to_numeric(raw, errors="coerce")
        for i in df.index[num[(table, col)].isna() & ~_is_blank(raw)]:
            errors.append(f"{_label(table, df, i)}: '{col}' value {raw[i]!r} is not a number")

    # ---- 4. value ranges ----
    for table, col, low, high, whole in RANGE_RULES:
        values = num.get((table, col))
        if values is None:
            continue
        df = tables[table]
        for i, v in values.dropna().items():
            if not math.isfinite(v):
                errors.append(f"{_label(table, df, i)}: '{col}' value {_fmt(v)} must be a finite number")
            elif whole and not float(v).is_integer():
                errors.append(f"{_label(table, df, i)}: '{col}' {_fmt(v)} must be a whole number")
            elif v < low or (high is not None and v > high):
                bound = f"between {low} and {high}" if high is not None else f"at least {low}"
                errors.append(f"{_label(table, df, i)}: '{col}' {_fmt(v)} must be {bound}")

    # ---- 5. week validity (every referenced week must exist in Calendar) ----
    calendar_weeks = None
    if ("Calendar", "Week") in num:
        calendar_weeks = {int(v) for v in num[("Calendar", "Week")].dropna() if float(v).is_integer()}
    for table, col in WEEK_COLUMNS:
        values = num.get((table, col))
        if values is None or calendar_weeks is None:
            continue
        df = tables[table]
        for i, v in values.dropna().items():
            if not float(v).is_integer():
                errors.append(f"{_label(table, df, i)}: '{col}' {_fmt(v)} must be a whole number")
            elif int(v) not in calendar_weeks:
                errors.append(f"{_label(table, df, i)}: '{col}' {_fmt(v)} is not a week in Calendar")

    # ---- 6. period logic (start <= end) ----
    for table, start_col, end_col in [
        ("Projects", "StartWeek", "DeadlineWeek"),
        ("Project_Requirements", "StartWeek", "EndWeek"),
    ]:
        if (table, start_col) in num and (table, end_col) in num:
            df = tables[table]
            start, end = num[(table, start_col)], num[(table, end_col)]
            for i in df.index[(start > end).fillna(False)]:
                errors.append(
                    f"{_label(table, df, i)}: {start_col} {_fmt(start[i])} is after {end_col} {_fmt(end[i])}"
                )

    # ---- 7. foreign keys ----
    for child_t, child_c, parent_t, parent_c in FOREIGN_KEYS:
        child, parent = get(child_t, child_c), get(parent_t, parent_c)
        if child is None or parent is None:
            continue
        valid = set(parent[parent_c].dropna())
        for i in child.index[~_is_blank(child[child_c])]:
            if child[child_c][i] not in valid:
                errors.append(
                    f"{_label(child_t, child, i)}: {child_c} {_fmt(child[child_c][i])} "
                    f"not found in {parent_t}.{parent_c}"
                )

    # ---- 8. allocation integrity (project and week must agree with the requirement) ----
    alloc = get("Allocations", "RequirementID", "ProjectID", "Week")
    req = get("Project_Requirements", "RequirementID", "ProjectID", "StartWeek", "EndWeek")
    if alloc is not None and req is not None:
        req_project = dict(zip(req["RequirementID"], req["ProjectID"]))
        req_start = dict(zip(req["RequirementID"], num.get(("Project_Requirements", "StartWeek"), req["StartWeek"])))
        req_end = dict(zip(req["RequirementID"], num.get(("Project_Requirements", "EndWeek"), req["EndWeek"])))
        weeks = num.get(("Allocations", "Week"))
        for i in alloc.index:
            rid = alloc["RequirementID"][i]
            if rid not in req_project:
                continue  # unknown requirement is already reported by the foreign-key check
            pid = alloc["ProjectID"][i]
            if not pd.isna(pid) and pid != req_project[rid]:
                errors.append(
                    f"{_label('Allocations', alloc, i)}: ProjectID {_fmt(pid)} does not match "
                    f"requirement {_fmt(rid)} (project {_fmt(req_project[rid])})"
                )
            week = weeks[i] if weeks is not None else float("nan")
            if not pd.isna(week) and not pd.isna(req_start[rid]) and not pd.isna(req_end[rid]):
                if week < req_start[rid] or week > req_end[rid]:
                    errors.append(
                        f"{_label('Allocations', alloc, i)}: Week {_fmt(week)} is outside requirement "
                        f"{_fmt(rid)} window ({_fmt(req_start[rid])}-{_fmt(req_end[rid])})"
                    )

    # ---- 9. dependency cycles ----
    deps = get("Dependencies", "PredecessorID", "SuccessorID")
    if deps is not None:
        edges = {
            (str(a), str(b))
            for a, b in zip(deps["PredecessorID"], deps["SuccessorID"])
            if not pd.isna(a) and not pd.isna(b)
        }
        for cycle in _find_cycles(edges):
            errors.append("Dependencies: cycle detected: " + " -> ".join(cycle))

    return errors


def _find_cycles(edges):
    """Return each cycle found as a list like ['P01', 'P02', 'P01']. Chains (A->B->C) are fine."""
    graph = defaultdict(list)
    for a, b in sorted(edges):
        graph[a].append(b)
    state = defaultdict(int)  # 0 = unseen, 1 = on current path, 2 = finished
    path, cycles = [], []

    def visit(node):
        state[node] = 1
        path.append(node)
        for nxt in graph[node]:
            if state[nxt] == 1:
                cycles.append(path[path.index(nxt):] + [nxt])
            elif state[nxt] == 0:
                visit(nxt)
        path.pop()
        state[node] = 2

    for node in sorted(graph):
        if state[node] == 0:
            visit(node)
    return cycles

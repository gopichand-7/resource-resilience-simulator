"""Step 15: dependency analysis ("downstream exposure").

The Dependencies table says which project blocks which:

    PredecessorID --Blocks--> SuccessorID

If a project is affected (for example because it loses a critical resource), every project that
it blocks, directly or through a chain (A -> B -> C), is EXPOSED.

This is exposure, not a schedule delay: the model does not predict that a downstream project
slips, only that it depends on a project that is now under pressure.

Every function is pure: inputs are never modified. Dependency cycles are rejected by the
validator, but the traversal also guards against them so it can never loop forever.
"""
from collections import deque

import pandas as pd

EXPOSURE_COLUMNS = ["From", "ProjectID", "Depth", "Path"]


def _successors(data):
    graph = {}
    deps = data["Dependencies"]
    for pred, succ in zip(deps["PredecessorID"], deps["SuccessorID"]):
        graph.setdefault(pred, []).append(succ)
    for key in graph:
        graph[key] = sorted(set(graph[key]))
    return graph


def _check_projects(data, project_ids):
    known = set(data["Projects"]["ProjectID"])
    unknown = sorted(set(project_ids) - known)
    if unknown:
        raise ValueError(f"Unknown project(s): {', '.join(map(str, unknown))}")


def projects_supported_by_resource(data, resource_id, weeks=None):
    """Projects a resource has planned hours on, optionally only in the given weeks.

    `weeks` can be any collection of week numbers, for example range(4, 6) for weeks 4-5.
    """
    alloc = data["Allocations"][["ResourceID", "ProjectID", "Week", "PlannedHrs"]].copy()
    alloc["Week"] = pd.to_numeric(alloc["Week"], errors="coerce")
    alloc["PlannedHrs"] = pd.to_numeric(alloc["PlannedHrs"], errors="coerce")
    alloc = alloc[(alloc["ResourceID"] == resource_id) & (alloc["PlannedHrs"] > 0)]
    if weeks is not None:
        alloc = alloc[alloc["Week"].isin(list(weeks))]
    return sorted(set(alloc["ProjectID"]))


def downstream_exposure(data, affected_projects):
    """Every project exposed to the affected projects, one row per (From, ProjectID) pair.

    From       the affected project the exposure comes from
    ProjectID  the exposed downstream project
    Depth      1 = directly blocked by From, 2 = blocked through one other project, and so on
    Path       the shortest chain, for example 'P01 -> P05'
    """
    affected = sorted(set(affected_projects))
    _check_projects(data, affected)
    graph = _successors(data)

    rows = []
    for start in affected:
        seen = {start}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            for nxt in graph.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                new_path = path + [nxt]
                rows.append({"From": start, "ProjectID": nxt, "Depth": len(new_path) - 1,
                             "Path": " -> ".join(new_path)})
                queue.append((nxt, new_path))

    result = pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)
    if result.empty:
        return result
    return result.sort_values(["From", "Depth", "ProjectID"]).reset_index(drop=True)


def exposed_project_ids(data, affected_projects):
    """Just the set of downstream project IDs (excluding the affected projects themselves)."""
    affected = set(affected_projects)
    table = downstream_exposure(data, affected)
    return sorted(set(table["ProjectID"]) - affected)

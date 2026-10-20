import streamlit as st

from src.app_state import GUIDE

from src.portfolio_view import (
    conflict_display,
    coverage_figure,
    dependency_edges,
    exposure_from_gaps,
    headline_metrics,
    open_conflicts,
    project_summary,
    scarce_requirements,
    single_points,
)

baseline = st.session_state["baseline"]
data, snapshot = baseline["data"], baseline["snapshot"]

st.header("Portfolio Map")
st.write("Where every project stands in the baseline plan: coverage, gaps, scarce skills and dependencies.")
st.caption(GUIDE["portfolio_map"][0])

# ---------- headline numbers ----------
tiles = headline_metrics(data, snapshot)
for column, (label, value, help_text) in zip(st.columns(len(tiles)), tiles):
    column.metric(label, f"{value:,.0f}", help=help_text)

# ---------- coverage per project ----------
summary = project_summary(data, snapshot)
st.subheader("Coverage by project")
st.caption("Hours the plan requires, split into hours somebody is planned to work and hours nobody is.")
st.plotly_chart(coverage_figure(summary), width="stretch")

with st.expander("Show as a table"):
    table = summary.rename(columns={
        "ProjectID": "Project", "ProjectName": "Name", "BusinessPriority": "Priority", "StartWeek": "Start wk",
        "DeadlineWeek": "Deadline wk", "RequiredHrs": "Required h", "AllocatedHrs": "Staffed h", "GapHrs": "Unmet h",
        "OpenRequirements": "Open reqs", "SinglePointRequirements": "Single-person",
        "CapacityShortageHrs": "Shortage h",
    })
    keep = ["Project", "Name", "Priority", "Start wk", "Deadline wk", "Required h", "Staffed h", "Unmet h",
            "Coverage", "Open reqs", "Single-person", "Shortage h"]
    numbers = {"Required h": "{:,.0f}", "Staffed h": "{:,.0f}", "Unmet h": "{:,.0f}",
               "Shortage h": "{:,.0f}", "Coverage": "{:.0%}"}
    st.table(table[keep].set_index("Project").style.format(numbers))  # a plain table shows every row; a grid would scroll inside the page

# ---------- open problems ----------
st.subheader("Open problems the engine found")
conflicts = open_conflicts(snapshot)
if conflicts.empty:
    st.success("No conflicts in the baseline plan.")
else:
    st.table(conflict_display(conflicts))  # a plain table wraps long text; a scrolling grid would cut it off
with st.expander("What do these types mean?"):
    st.markdown(
        "- **Staffing gap**: required hours nobody is planned to work.\n"
        "- **Capacity shortage**: the unmet hours are bigger than the free hours of *every* qualified person combined, "
        "so reshuffling the qualified people cannot close it.\n"
        "- **Competing requirements**: the qualified people's hours are already held by other requirements.\n"
        "- **Over-allocation**: a person is planned for more hours than their weekly capacity."
    )

# ---------- scarce skills ----------
st.subheader("Scarce skills")
st.caption("Requirements where very few people meet both the skill and the minimum proficiency. Three or more is a broader pool.")
people = single_points(data)
if not people.empty:
    for person in people.itertuples():
        st.write(
            f"**{person.ResourceName}** ({person.ResourceID}) is the only qualified person for "
            f"{person.Requirements} on {person.Projects}."
        )
scarce = scarce_requirements(snapshot).rename(columns={
    "RequirementID": "Requirement", "ProjectID": "Project", "MinProficiency": "Min level",
    "QualifiedCount": "Qualified", "QualifiedResources": "Who"})
st.table(scarce.set_index("Requirement"))

# ---------- dependencies ----------
st.subheader("Dependencies")
st.caption("Each row means the blocking project must finish before the blocked project can. Exposure means a project depends on one with unmet hours. "
           "It is not a prediction that anything will slip.")
edges = dependency_edges(data, snapshot).rename(columns={
    "Predecessor": "Must finish first", "PredecessorName": "Name", "Successor": "Waits for it",
    "SuccessorName": "Waiting project", "PredecessorUnmetHrs": "Its unmet h"})
st.table(edges.set_index("Must finish first").style.format({"Its unmet h": "{:,.0f}"}))
exposure = exposure_from_gaps(data, snapshot)
if exposure.empty:
    st.write("No project is downstream of a project with unmet hours.")
else:
    for row in exposure.itertuples():
        st.write(f"**{row.ProjectID}** is exposed through {row.Path} (depth {row.Depth}).")

st.caption(GUIDE["portfolio_map"][1])

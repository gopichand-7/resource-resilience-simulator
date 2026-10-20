import streamlit as st

from src.app_state import GUIDE

from src.allocation import FIT_WEIGHTS, PARTIAL_FIT_MIN, STRONG_FIT_MIN
from src.portfolio_view import conflict_display
from src.planner_view import (
    assignee_line,
    candidate_summary,
    candidate_table,
    current_assignees,
    candidate_view,
    requirement_conflicts,
    requirement_facts,
    requirement_options,
    weekly_figure,
)

baseline = st.session_state["baseline"]
data, snapshot = baseline["data"], baseline["snapshot"]

st.header("Allocation Planner")
st.write("Pick a requirement and see every person who could work on it, with the reason behind each fit.")
st.caption(GUIDE["allocation_planner"][0])

# ---------- choose a requirement ----------
# Streamlit forgets a widget's value when you leave the page, so the choices are also saved under a plain key
saved = st.session_state.get("planner_saved", {})
if "planner_only_unmet" not in st.session_state:
    st.session_state["planner_only_unmet"] = saved.get("only_unmet", True)
only_unmet = st.checkbox("Only requirements with unmet hours", key="planner_only_unmet")
options = requirement_options(data, snapshot, only_unmet=only_unmet)
if options.empty:
    st.success("Every requirement is fully staffed in the baseline plan.")
    st.stop()
labels = list(options["Label"])
if st.session_state.get("planner_choice") not in labels:
    st.session_state["planner_choice"] = saved.get("choice") if saved.get("choice") in labels else labels[0]
choice = st.selectbox("Requirement", labels, key="planner_choice")
st.session_state["planner_saved"] = {"only_unmet": only_unmet, "choice": choice}
requirement_id = options.loc[options["Label"] == choice, "RequirementID"].iloc[0]
facts = requirement_facts(data, snapshot, requirement_id)

# ---------- the requirement ----------
st.subheader(f"{requirement_id}: {facts['Skill']} on {facts['ProjectName']}")
# Long text (role names) goes in a normal line, never in a big tile: tiles cut long text off with "..."
st.write(f"**Role needed:** {facts['Role']}  |  **Skill pool:** {facts['Scarcity']} ({facts['QualifiedCount']} qualified)")
tiles = [
    ("Minimum level", f"{facts['MinProficiency']} of 5"),
    ("Hours per week", f"{facts['HoursPerWeek']:g}h"),
    ("Window", f"Weeks {facts['StartWeek']}-{facts['EndWeek']}"),
    ("Unmet hours", f"{facts['GapHrs']:,.0f} of {facts['RequiredHrs']:,.0f}h"),
]
for column, (label, value) in zip(st.columns(len(tiles)), tiles):
    column.metric(label, value)
st.markdown("**Hours needed each week**")
st.caption("Blue is planned work; amber is work nobody is planned to do.")
st.plotly_chart(weekly_figure(snapshot, requirement_id), width="stretch")

conflicts = requirement_conflicts(snapshot, requirement_id)
if not conflicts.empty:
    st.markdown("**Problems the engine found on this requirement**")
    st.table(conflict_display(conflicts[["ConflictType", "Weeks", "Hours", "Detail"]]))

# ---------- candidates ----------
st.subheader("Candidates")
assignees = current_assignees(data, requirement_id)
if assignee_line(assignees):
    st.write(assignee_line(assignees))
candidates = candidate_view(data, snapshot, requirement_id)
if candidates.empty:
    st.warning("Nobody in the dataset has this skill, so there are no candidates.")
else:
    for line in candidate_summary(facts, candidates):
        st.write(line)
    # The answer first (score, label, flags), the working behind it to the right
    st.table(candidate_table(candidates, assignees["ResourceID"]).set_index("ID"))

with st.expander("How is the fit score worked out?"):
    st.markdown(
        f"**Fit score = {FIT_WEIGHTS['skill']} × skill match + {FIT_WEIGHTS['availability']} × availability "
        f"+ {FIT_WEIGHTS['headroom']} × load headroom**, a number from 0 to 100.\n\n"
        "- **Skill match**: the person's level divided by the minimum level, capped at 1.\n"
        "- **Availability**: free hours in their *tightest* week of the window divided by the hours per week "
        "the requirement needs, capped at 1.\n"
        "- **Load headroom**: 1 minus their busiest week's utilization.\n\n"
        f"Labels: **Strong** at {STRONG_FIT_MIN}+, **Partial** from {PARTIAL_FIT_MIN}, **Constraint** below that. "
        "Someone below the minimum level can never be better than Partial.\n\n"
        "The weights and thresholds are simulation assumptions chosen for this prototype, not industry standards. "
        "The score orders the list; it does not choose a person for you."
    )

st.caption(GUIDE["allocation_planner"][1])

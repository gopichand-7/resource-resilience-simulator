import streamlit as st

from src.app_state import GUIDE, clear_response
from src.theme import STATUS_NOTE, bottom_line_html, card_html, heading_html

from src.shock_view import (
    SHOCK_DEADLINE,
    SHOCK_EFFORT,
    SHOCK_KINDS,
    SHOCK_RESOURCE,
    before_after_figure,
    bottom_line,
    calendar_weeks,
    conflict_change_display,
    deadline_range,
    default_resource,
    metric_tiles,
    owner_questions,
    project_status_table,
    project_options,
    resource_options,
    resource_preview,
    run_shock,
    shock_story,
)

baseline = st.session_state["baseline"]
data = baseline["data"]

st.header("Shock Simulator")
st.write("Apply one shock to the baseline plan and trace what it hits. Every shock starts from the baseline, so shocks do not stack.")
st.caption(GUIDE["shock_simulator"][0])

# ---------- choose a shock ----------
# Streamlit forgets a widget's value when you leave the page, so the form is also saved under a plain key
# and put back when you return. Defaults are seeded into session state (no value= argument on the widgets).
FORM_KEYS = ["shock_kind", "shock_person", "shock_weeks", "shock_dl_project", "shock_deadline",
             "shock_wp_project", "shock_percent"]
saved_form = st.session_state.get("shock_form", {})
for key, value in saved_form.items():
    st.session_state.setdefault(key, value)


def valid_choice(key, options, fallback):
    """Keep a remembered choice only if it is still on offer."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback


valid_choice("shock_kind", SHOCK_KINDS, SHOCK_KINDS[0])
kind = st.radio("What happens?", SHOCK_KINDS, horizontal=True, key="shock_kind")
params = {}
can_apply = True

if kind == SHOCK_RESOURCE:
    people = resource_options(data)
    labels = list(people["Label"])
    default = people.index[people["ResourceID"] == default_resource(data)][0]
    valid_choice("shock_person", labels, labels[int(default)])
    person = st.selectbox("Who is unavailable?", labels, key="shock_person")
    resource_id = people.loc[people["Label"] == person, "ResourceID"].iloc[0]
    first, last = calendar_weeks(data)
    start_default = min(max(4, first), last)
    kept = st.session_state.get("shock_weeks")
    if not (isinstance(kept, (tuple, list)) and len(kept) == 2 and first <= kept[0] <= kept[1] <= last):
        st.session_state["shock_weeks"] = (start_default, min(start_default + 1, last))
    weeks = st.slider("Which weeks?", first, last, key="shock_weeks")
    chosen = list(range(weeks[0], weeks[1] + 1))
    projects, hours = resource_preview(data, resource_id, chosen)
    if hours > 0:
        st.caption(f"In those weeks this person is planned for {hours:,.0f}h on {', '.join(projects)}. "
                   "Those hours become unmet demand; nobody is reassigned automatically.")
    else:
        st.caption("This person has no planned hours in those weeks, so the shock will change nothing.")
    params = {"resource_id": resource_id, "weeks": chosen}

elif kind == SHOCK_DEADLINE:
    projects = project_options(data)
    project_labels = list(projects["Label"])
    valid_choice("shock_dl_project", project_labels, project_labels[0])
    choice = st.selectbox("Which project?", project_labels, key="shock_dl_project")
    project_id = projects.loc[projects["Label"] == choice, "ProjectID"].iloc[0]
    allowed = deadline_range(data, project_id)
    if allowed is None:
        st.info("This project's deadline is already its start week, so it cannot move earlier.")
        can_apply = False
    else:
        current = allowed[1] + 1
        if allowed[0] == allowed[1]:
            new_deadline = allowed[0]
            st.write(f"The deadline can only move to week {new_deadline} (now week {current}).")
        else:
            kept = st.session_state.get("shock_deadline")
            st.session_state["shock_deadline"] = (allowed[1] if not isinstance(kept, int)
                                                  else min(max(kept, allowed[0]), allowed[1]))
            new_deadline = st.slider(f"New deadline (now week {current})", allowed[0], allowed[1], key="shock_deadline")
        params = {"project_id": project_id, "new_deadline": int(new_deadline)}

else:
    projects = project_options(data)
    project_labels = list(projects["Label"])
    valid_choice("shock_wp_project", project_labels, project_labels[0])
    choice = st.selectbox("Which project?", project_labels, key="shock_wp_project")
    project_id = projects.loc[projects["Label"] == choice, "ProjectID"].iloc[0]
    if st.session_state.get("shock_percent") not in range(5, 101, 5):
        st.session_state["shock_percent"] = 20
    percent = st.slider("Extra work (%)", 5, 100, step=5, key="shock_percent")
    params = {"project_id": project_id, "percent": float(percent)}

st.session_state["shock_form"] = {**saved_form, **{k: st.session_state[k] for k in FORM_KEYS if k in st.session_state}}

if st.button("Apply shock", type="primary", disabled=not can_apply):
    try:
        result = run_shock(baseline, kind, **params)
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state.update(result)
        clear_response(st.session_state)
        st.rerun()

# ---------- what it did ----------
comparison = st.session_state["comparison"]
if comparison is None:
    st.info("Choose a shock and press Apply shock to see what it hits.")
    st.stop()

st.subheader("What the shock did")
st.markdown(bottom_line_html(bottom_line(comparison, st.session_state["scenario_label"])), unsafe_allow_html=True)

tiles = metric_tiles(comparison)
for column, (label, value, delta, help_text) in zip(st.columns(len(tiles)), tiles):
    column.metric(label, f"{value:,.0f}", delta=f"{delta:+,.0f}" if abs(delta) > 1e-9 else "no change",
                  delta_color="inverse" if abs(delta) > 1e-9 else "off", help=help_text)

st.markdown(heading_html("Projects after the shock"), unsafe_allow_html=True)
st.table(project_status_table(st.session_state["scenario_data"], baseline["snapshot"], st.session_state["scenario_snapshot"]))
st.caption(STATUS_NOTE)

questions = owner_questions(baseline["snapshot"], st.session_state["scenario_snapshot"], comparison,
                           st.session_state["scenario_data"])
if questions:
    st.markdown(heading_html("Decisions for the program owner"), unsafe_allow_html=True)
    for column, (title, detail) in zip(st.columns(len(questions)), questions):
        column.markdown(card_html(title, detail), unsafe_allow_html=True)

st.markdown("**How it spreads**")
for line in shock_story(comparison):
    st.write(line)

st.plotly_chart(before_after_figure(baseline["snapshot"]["demand"], st.session_state["scenario_snapshot"]["demand"]),
                width="stretch")

changes = conflict_change_display(comparison)
if not changes.empty:
    st.markdown("**Conflicts that changed**")
    st.table(changes)

st.caption("To test a response to this shock (substitute someone, move hours, or adjust the timeline), open Trade-offs in the sidebar.")

with st.expander("How is each shock modelled?"):
    st.markdown(
        "- **A person becomes unavailable**: their capacity is zero in the chosen weeks and their planned hours in "
        "those weeks are removed, so they become unmet demand. Nobody is reassigned.\n"
        "- **A project's deadline moves earlier**: each requirement keeps its total effort but must fit in fewer weeks, "
        "so weekly demand rises. Allocations after the new deadline are removed and the rest are *not* redistributed. "
        "New unmet hours = new demand minus the hours already allocated.\n"
        "- **A project needs more work**: weekly demand of every requirement of the project grows by the chosen "
        "percentage. Allocations stay as they were.\n\n"
        "These are deliberately simple rules for a student prototype. The tool shows exposure (what depends on what), "
        "not predicted delays."
    )

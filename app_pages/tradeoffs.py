import streamlit as st

from src.app_state import GUIDE, clear_response
from src.shock_view import calendar_weeks
from src.tradeoffs_view import (
    RESPONSE_KINDS,
    RESPONSE_REALLOCATE,
    RESPONSE_SUBSTITUTE,
    RESPONSE_TIMELINE,
    changes_display,
    comparison_table,
    cost_lines,
    extendable_projects,
    fixed_lines,
    gap_weeks,
    person_names,
    reallocate_limit,
    reallocate_sources,
    run_response,
    qualification_note,
    still_open_line,
    style_changes,
    substitute_people,
    three_way_figure,
    timeline_note,
    unmet_options,
    warnings_display,
)

baseline = st.session_state["baseline"]

st.header("Trade-offs")
st.write("Try a response to the shock and compare what it fixes with what it costs. No option is picked for you.")
st.caption(GUIDE["tradeoffs"][0])

shocked = st.session_state["scenario_data"]
if shocked is None:
    st.info("Apply a shock first (Shock Simulator in the sidebar). A response is a move you make after something goes wrong.")
    st.stop()
shocked_snapshot = st.session_state["scenario_snapshot"]

st.write(f"**Active shock:** {st.session_state['scenario_label']}")
st.caption("Each response starts from the shocked plan. Responses do not stack: trying a new one replaces the last.")

# ---------- choose a response ----------
# Streamlit forgets a widget's value when you leave the page, so the form is also saved under a plain key
# and put back when you return. Defaults are seeded into session state (no value= argument on the widgets).
FORM_KEYS = ["tr_kind", "tr_requirement", "tr_weeks", "tr_who", "tr_mover", "tr_source", "tr_hours",
             "tr_project", "tr_deadline", "tr_sig_weeks", "tr_sig_hours", "tr_sig_deadline"]
saved_form = st.session_state.get("tradeoff_form", {})
for key, value in saved_form.items():
    st.session_state.setdefault(key, value)


def valid_choice(key, options, fallback):
    """Keep a remembered choice only if it is still on offer."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback


def reset_when_changed(sig_key, signature, value_key, value):
    """When the thing a slider depends on changes, put the slider back to its full range."""
    if st.session_state.get(sig_key) != signature or value_key not in st.session_state:
        st.session_state[value_key] = value
        st.session_state[sig_key] = signature


valid_choice("tr_kind", RESPONSE_KINDS, RESPONSE_KINDS[0])
kind = st.radio("What will you try?", RESPONSE_KINDS, horizontal=True, key="tr_kind")
params = {}
can_apply = True

if kind in (RESPONSE_SUBSTITUTE, RESPONSE_REALLOCATE):
    options = unmet_options(shocked, shocked_snapshot)
    if options.empty:
        st.success("Nothing is unmet after this shock, so there is nothing to cover.")
        st.stop()
    labels = list(options["Label"])
    valid_choice("tr_requirement", labels, labels[0])
    choice = st.selectbox("Which requirement has the problem?", labels, key="tr_requirement")
    requirement_id = options.loc[options["Label"] == choice, "RequirementID"].iloc[0]
    open_weeks = gap_weeks(shocked_snapshot, requirement_id)
    if len(open_weeks) == 1:
        weeks = (open_weeks[0], open_weeks[0])
        st.write(f"Unmet hours are in week {open_weeks[0]}.")
    else:
        full = (open_weeks[0], open_weeks[-1])
        reset_when_changed("tr_sig_weeks", (requirement_id, full), "tr_weeks", full)
        remembered = st.session_state["tr_weeks"]
        if not (isinstance(remembered, tuple) and full[0] <= remembered[0] <= remembered[1] <= full[1]):
            st.session_state["tr_weeks"] = full
        weeks = st.slider("Which weeks?", full[0], full[1], key="tr_weeks")
    chosen = list(range(weeks[0], weeks[1] + 1))
    params = {"requirement_id": requirement_id, "weeks": chosen}

    people = substitute_people(shocked, shocked_snapshot, requirement_id, chosen)
    if people.empty:
        st.warning("Nobody with this skill is available in those weeks.")
        can_apply = False
    elif kind == RESPONSE_SUBSTITUTE:
        valid_choice("tr_who", list(people["Label"]), people["Label"].iloc[0])
        who = st.selectbox("Who covers it?", list(people["Label"]), key="tr_who")
        params["resource_id"] = people.loc[people["Label"] == who, "ResourceID"].iloc[0]
        st.caption("The person covers all unmet hours of this requirement in the chosen weeks, on top of their current work. "
                   "The list is sorted by fit score, which is not a recommendation: check the level warning.")
    else:
        valid_choice("tr_mover", list(people["Label"]), people["Label"].iloc[0])
        who = st.selectbox("Whose hours move?", list(people["Label"]), key="tr_mover")
        resource_id = people.loc[people["Label"] == who, "ResourceID"].iloc[0]
        sources = reallocate_sources(shocked, resource_id, requirement_id)
        if sources.empty:
            st.warning("This person has no other planned hours to move.")
            can_apply = False
        else:
            valid_choice("tr_source", list(sources["Label"]), sources["Label"].iloc[0])
            source = st.selectbox("Move hours away from", list(sources["Label"]), key="tr_source")
            from_requirement = sources.loc[sources["Label"] == source, "RequirementID"].iloc[0]
            limit = reallocate_limit(shocked, shocked_snapshot, resource_id, from_requirement, requirement_id, chosen)
            if limit < 1:
                st.warning("In these weeks the person has no hours to move, or nothing unmet to move them to. Try fewer weeks.")
                can_apply = False
            else:
                reset_when_changed("tr_sig_hours", (resource_id, from_requirement, requirement_id, tuple(chosen)),
                                   "tr_hours", int(limit))
                if st.session_state["tr_hours"] not in range(1, int(limit) + 1):
                    st.session_state["tr_hours"] = int(limit)
                hours = st.slider("Hours per week to move", 1, int(limit), key="tr_hours")
                params.update({"resource_id": resource_id, "from_requirement": from_requirement,
                               "hours_per_week": float(hours)})
                st.caption("The hours are taken from the other requirement, so its unmet hours grow by the same amount.")
else:
    projects = extendable_projects(shocked)
    if projects.empty:
        st.info("Every project already ends in the last week of the calendar.")
        st.stop()
    valid_choice("tr_project", list(projects["Label"]), projects["Label"].iloc[0])
    choice = st.selectbox("Which project?", list(projects["Label"]), key="tr_project")
    row = projects[projects["Label"] == choice].iloc[0]
    _, last = calendar_weeks(shocked)
    if int(row["Deadline"]) + 1 == last:
        new_deadline = last
        st.write(f"The deadline can only move to week {last} (now week {int(row['Deadline'])}).")
    else:
        first_option = int(row["Deadline"]) + 1
        reset_when_changed("tr_sig_deadline", (row["ProjectID"], first_option), "tr_deadline", first_option)
        if st.session_state["tr_deadline"] not in range(first_option, last + 1):
            st.session_state["tr_deadline"] = first_option
        new_deadline = st.slider(f"New deadline (now week {int(row['Deadline'])})", first_option, last,
                                 key="tr_deadline")
    params = {"project_id": row["ProjectID"], "new_deadline": int(new_deadline)}

st.session_state["tradeoff_form"] = {**saved_form, **{k: st.session_state[k] for k in FORM_KEYS if k in st.session_state}}

if st.button("Try this response", type="primary", disabled=not can_apply):
    try:
        result = run_response(baseline, shocked, kind, **params)
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state.update(result)
        st.rerun()

# ---------- what it fixed and what it cost ----------
result = st.session_state["response_result"]
if st.session_state["response_data"] is None or result is None:
    st.info("Choose a response and press Try this response to see what it fixes and what it costs.")
    st.stop()

evaluation, response_snapshot = result["evaluation"], result["snapshot"]
applied_kind = st.session_state["response_kind"]

st.subheader("What this response did")
st.write(f"**{st.session_state['response_label']}**")
st.table(comparison_table(baseline["snapshot"], shocked_snapshot, response_snapshot))

fixed_column, cost_column = st.columns(2)
with fixed_column:
    st.markdown("**What it fixed**")
    for line in fixed_lines(evaluation):
        st.write(line)
with cost_column:
    st.markdown("**What it cost**")
    for line in cost_lines(evaluation, applied_kind):
        st.write(line)
st.write(still_open_line(evaluation))
qualification = qualification_note(evaluation)
if qualification:
    st.write(qualification)
note = timeline_note(applied_kind)
if note:
    st.caption(note)

st.plotly_chart(three_way_figure(baseline["snapshot"], shocked_snapshot, response_snapshot), width="stretch")

names = person_names(shocked)
changes = changes_display(evaluation, names)
st.markdown("**Exact hours that moved**")
if changes.empty:
    st.write("No one's planned hours changed.")
else:
    st.table(style_changes(changes))
warnings = warnings_display(evaluation, names)
if not warnings.empty:
    st.markdown("**Warnings**")
    st.table(warnings)

if st.button("Undo this response"):
    clear_response(st.session_state)
    st.rerun()

st.caption("The prototype shows consequences, not a verdict. Whether a cost is acceptable is the planner's call.")

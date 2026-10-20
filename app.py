"""Resource Resilience & Project Capacity Simulator - app shell (Step 23).

Run with:  streamlit run app.py

This file only does the things every page needs: page setup, the disclosure banner, loading and
validating the data once, the shared session state, and navigation. The four pages live in app_pages/.
"""
import streamlit as st

from src.theme import apply_theme
from src.app_state import (
    APP_TAGLINE,
    APP_TITLE,
    DISCLOSURE,
    PAGES,
    init_session,
    load_baseline,
    reset_scenario,
)

st.set_page_config(page_title=APP_TITLE, page_icon=None, layout="wide")
apply_theme()


@st.cache_data(show_spinner="Loading and checking the dataset...")
def get_baseline():
    return load_baseline()


baseline = get_baseline()
init_session(st.session_state)

# ---------- header on every page ----------
st.title(APP_TITLE)
st.caption(APP_TAGLINE)
st.info(DISCLOSURE)

# ---------- stop with a plain explanation if the data is not usable ----------
if baseline["errors"]:
    st.error("The dataset has problems, so the simulator cannot start. Fix these in the Excel file and reload:")
    for message in baseline["errors"][:25]:
        st.write(f"- {message}")
    if len(baseline["errors"]) > 25:
        st.write(f"...and {len(baseline['errors']) - 25} more.")
    if st.button("Reload the data"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

st.session_state["baseline"] = baseline

# ---------- sidebar ----------
with st.sidebar:
    st.subheader("Scenario")
    if st.session_state["scenario_data"] is None:
        st.write("Baseline plan (no shock applied)")
    else:
        st.write(f"Shock: {st.session_state['scenario_label']}")
        if st.session_state["response_data"] is not None:
            st.write(f"Response: {st.session_state['response_label']}")
    if st.button("Reset to baseline", disabled=st.session_state["scenario_data"] is None):
        reset_scenario(st.session_state)
        st.rerun()

# ---------- navigation ----------
navigation = st.navigation(
    [st.Page(f"app_pages/{key}.py", title=title, url_path=key, default=(i == 0)) for i, (key, title) in enumerate(PAGES)]
)
try:
    navigation.run()
except Exception as error:  # Streamlit's own stop/rerun signals are not Exceptions, so they pass through
    st.error("Something went wrong while showing this page. Nothing in the data was changed. "
             "Click Reset to baseline in the sidebar and try again.")
    with st.expander("Technical detail"):
        st.code(f"{type(error).__name__}: {error}")

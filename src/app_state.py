"""Step 23: logic shared by every page of the app (no Streamlit code here, so it can be tested).

The app never touches the Excel file directly. It asks for the baseline once, gets back the data, the
validation errors and the baseline snapshot, and every page works from that.
"""
from src.data_loader import DataLoadError, load_data
from src.scenarios import take_snapshot
from src.validator import validate_data

APP_TITLE = "Resource Resilience & Project Capacity Simulator"
APP_TAGLINE = "See which projects a resource shortage, an earlier deadline or extra work puts under pressure, and what each response costs."

DISCLOSURE = (
    "Student-built decision-support prototype. All data is synthetic (28 invented people, 9 invented projects). "
    "It shows consequences and trade-offs; it does not forecast, optimize, use machine learning or recommend a decision."
)

# (key, title) in navigation order. The page files live in app_pages/<key>.py
PAGES = [
    ("portfolio_map", "Portfolio Map"),
    ("allocation_planner", "Allocation Planner"),
    ("shock_simulator", "Shock Simulator"),
    ("tradeoffs", "Trade-offs"),
]

# The one-line "what do I do here" and "what next" for each page (start, next). Plain pointers, never a recommendation.
GUIDE = {
    "portfolio_map": (
        "Start here: read the five numbers, find the projects with an amber (unmet) bar, then open Allocation Planner "
        "to see who could cover them.",
        "Next: open Allocation Planner to look at who could work on a requirement, or Shock Simulator to test what a "
        "shortage would hit.",
    ),
    "allocation_planner": (
        "Start here: pick a requirement with unmet hours and read who could work on it and why.",
        "Next: open Shock Simulator to see what happens if someone becomes unavailable or a project changes.",
    ),
    "shock_simulator": (
        "Start here: choose a shock, press Apply shock, read the bottom line, then open Trade-offs to test a response.",
        "",
    ),
    "tradeoffs": (
        "Start here: pick a response, press Try this response, then compare what it fixed with what it cost.",
        "",
    ),
}

# Everything a page may keep between clicks lives under one of these keys.
SESSION_DEFAULTS = {
    "scenario_data": None,     # data after a shock (None = no shock applied yet)
    "scenario_label": "",      # plain-words description of the shock, e.g. "R15 unavailable, weeks 4-5"
    "scenario_snapshot": None, # snapshot of the shocked plan
    "comparison": None,        # baseline vs shocked plan (compare_scenario result)
    "response_data": None,     # data after a response (None = no response applied yet)
    "response_label": "",
    "response_kind": None,     # which kind of response was tried
    "response_result": None,   # snapshot + evaluation of the response (kept so the page does not recompute)
}


def load_baseline(path=None):
    """Load, validate and summarise the dataset.

    Returns {"data", "errors", "snapshot"}. If the file cannot be read, or fails validation, data errors are
    returned in `errors` and `snapshot` is None, so the app can explain the problem instead of crashing.
    """
    try:
        data = load_data(path)
    except DataLoadError as exc:
        return {"data": None, "errors": [str(exc)], "snapshot": None}
    except Exception as exc:  # a corrupt file must never produce a raw traceback in front of a user
        return {"data": None, "errors": [f"The dataset could not be read: {exc}"], "snapshot": None}

    try:
        errors = validate_data(data)
        if errors:
            return {"data": data, "errors": errors, "snapshot": None}
        return {"data": data, "errors": [], "snapshot": take_snapshot(data)}
    except Exception as exc:  # odd data that the checks did not anticipate is still a message, not a crash
        return {"data": data, "errors": [f"The dataset could not be checked: {type(exc).__name__}: {exc}"],
                "snapshot": None}


def init_session(state):
    """Fill in any missing session key (works on st.session_state or a plain dict)."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in state:
            state[key] = default
    return state


def clear_response(state):
    """Forget the response only; the shock stays."""
    for key in ("response_data", "response_label", "response_kind", "response_result"):
        state[key] = SESSION_DEFAULTS[key]
    return state


def reset_scenario(state):
    """Clear the shock and the response, back to the baseline."""
    for key, default in SESSION_DEFAULTS.items():
        state[key] = default
    return state

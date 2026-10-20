"""The 'Program Office' look: one place for the colours, fonts and small HTML pieces every page shares.

The colours and rules (status always uses words AND a shape, never colour alone) come from the design
the project owner picked. No Streamlit calls except apply_theme, so the rest can be tested.
"""
from html import escape

NAVY = "#1F3A5F"
INK = "#1B2733"
SLATE = "#4F5D6B"
PAPER = "#FAFAF7"
RULE = "#D6D9DC"
AMBER = "#B45309"      # unmet hours / a gap
BRICK = "#A4262C"      # a large gap, or a number that got worse
GREEN = "#2E6B4F"      # staffed / recovered
BLUE = "#3987E5"       # plan after a response

# Status words always travel with a shape, so colour is never the only signal.
STATUS_STAFFED = "● Staffed"
STATUS_GAP = "▲ Gap"
STATUS_LARGE_GAP = "■ Large gap"
STATUS_EXPOSED = "◇ Exposed"
LARGE_GAP_SHARE = 0.25   # a project with 25% or more of its hours unmet is a "Large gap" (an assumption)

STATUS_NOTE = ('Status describes unmet hours and dependencies only. "Large gap" means 25% or more of the project\'s '
               'hours are unmet; "Exposed" means it depends on a project with unmet hours (a dependency, not a '
               "predicted delay).")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:wght@500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp button, .stApp input {
  font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
}
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Source Serif 4', Georgia, serif; color: #1B2733; letter-spacing: -0.01em; }
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; }
.stApp h1 { font-size: 1.3rem; color: #4F5D6B; border-bottom: 1px solid #1F3A5F; padding-bottom: 0.3rem; }
.stApp h2 { font-size: 2.2rem; margin-top: 0.4rem; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D6D9DC; padding: 8px 10px; }
[data-testid="stMetricLabel"] p { font-size: 0.8rem; font-weight: 600; color: #4F5D6B; }
[data-testid="stMetricValue"] { font-weight: 700; }
[data-testid="stAlert"] { border-radius: 0; border: 1px solid #D6D9DC; }
[data-testid="stTable"] table { border: 1px solid #D6D9DC; background: #FFFFFF; }
@media (min-width: 1800px) {
  [data-testid="stTable"] th, [data-testid="stTable"] th * { white-space: nowrap !important; }
}
[data-testid="stTable"] th { background: #EEF0F2; color: #4F5D6B; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.po-bottom { background: #FFFFFF; border: 1px solid #D6D9DC; border-left: 5px solid #1F3A5F; padding: 14px 18px; margin: 6px 0 14px 0; }
.po-bottom .po-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #1F3A5F; }
.po-bottom .po-text { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.15rem; line-height: 1.5; margin-top: 4px; }
.po-card { min-height: 118px; background: #FFFFFF; border: 1px solid #D6D9DC; padding: 10px 14px; margin-bottom: 10px; }
.po-card .po-title { font-weight: 700; }
.po-card .po-detail { font-size: 0.9rem; line-height: 1.45; color: #2C3A47; margin-top: 3px; }
.po-heading { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #1F3A5F;
  border-bottom: 2px solid #1F3A5F; padding-bottom: 6px; margin-bottom: 10px; }
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 0.6rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 170px !important; min-width: 170px !important; width: auto !important; }
}
@media (max-width: 900px) {
  [data-testid="stTable"] { overflow-x: auto; }
  [data-testid="stTable"] td, [data-testid="stTable"] th { white-space: nowrap; }
  [data-testid="stTable"] td:last-child { white-space: normal; min-width: 240px; }
}
"""


def apply_theme():
    """Inject the shared CSS (call once, from app.py)."""
    import streamlit as st
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def bottom_line_html(text, label="Bottom line"):
    """The one-paragraph summary box at the top of a result. Text is escaped."""
    return (f'<div class="po-bottom"><div class="po-label">{escape(label)}</div>'
            f'<div class="po-text">{escape(text)}</div></div>')


def card_html(title, detail):
    return (f'<div class="po-card"><div class="po-title">{escape(title)}</div>'
            f'<div class="po-detail">{escape(detail)}</div></div>')


def heading_html(text):
    return f'<div class="po-heading">{escape(text)}</div>'


def project_status(unmet_hours, required_hours, exposed):
    """One status word (with its shape) for a project. Unmet hours come first, then dependency exposure."""
    if unmet_hours > 1e-9:
        share = unmet_hours / required_hours if required_hours > 0 else 1.0
        return STATUS_LARGE_GAP if share >= LARGE_GAP_SHARE - 1e-9 else STATUS_GAP
    return STATUS_EXPOSED if exposed else STATUS_STAFFED

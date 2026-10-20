"""The shared look. Status words must carry a shape (never colour alone) and the 25% rule must hold at the edge."""
import re

from src.theme import (
    CSS, LARGE_GAP_SHARE, STATUS_EXPOSED, STATUS_GAP, STATUS_LARGE_GAP, STATUS_STAFFED,
    bottom_line_html, card_html, heading_html, project_status,
)


def test_every_status_word_carries_a_shape():
    for word in [STATUS_STAFFED, STATUS_GAP, STATUS_LARGE_GAP, STATUS_EXPOSED]:
        assert word[0] in "●▲■◇" and word[1] == " "
    assert len({w[0] for w in [STATUS_STAFFED, STATUS_GAP, STATUS_LARGE_GAP, STATUS_EXPOSED]}) == 4


def test_a_project_with_no_unmet_hours_is_staffed_or_exposed():
    assert project_status(0, 100, False) == STATUS_STAFFED
    assert project_status(0, 100, True) == STATUS_EXPOSED


def test_unmet_hours_decide_the_status_before_exposure():
    assert project_status(5, 100, True) == STATUS_GAP          # exposed too, but its own gap comes first


def test_large_gap_starts_exactly_at_25_percent():
    assert LARGE_GAP_SHARE == 0.25
    assert project_status(24.9, 100, False) == STATUS_GAP
    assert project_status(25, 100, False) == STATUS_LARGE_GAP
    assert project_status(70, 180, False) == STATUS_LARGE_GAP           # P04: 38.9%
    assert project_status(24, 456, False) == STATUS_GAP                  # P01: 5.3%


def test_a_project_with_no_required_hours_but_unmet_hours_is_a_large_gap():
    assert project_status(10, 0, False) == STATUS_LARGE_GAP


def test_html_pieces_escape_what_they_are_given():
    html = bottom_line_html("A <b>bold</b> & claim")
    assert "<b>" not in html and "&lt;b&gt;" in html and "&amp;" in html
    assert "Bottom line" in html
    assert "&lt;i&gt;" in card_html("<i>t</i>", "d") and "&lt;u&gt;" in heading_html("<u>h</u>")


def test_the_palette_and_fonts_are_the_chosen_ones():
    for token in ["#1F3A5F", "#FAFAF7", "#B45309", "Source Serif 4", "Source Sans 3"]:
        assert token in CSS or token in open("src/theme.py", encoding="utf-8").read()


def test_the_streamlit_theme_file_matches_the_palette():
    text = open(".streamlit/config.toml", encoding="utf-8").read()
    for token in ['primaryColor = "#1F3A5F"', 'backgroundColor = "#FAFAF7"', 'textColor = "#1B2733"']:
        assert token in text


def test_css_has_balanced_braces():
    assert CSS.count("{") == CSS.count("}")
    assert not re.search(r"</?script", CSS, re.I)


def test_small_screens_get_wrapping_tiles_and_scrolling_tables():
    assert "@media (max-width: 1100px)" in CSS and "flex-wrap: wrap" in CSS      # tiles wrap instead of being squeezed
    assert "@media (max-width: 900px)" in CSS and "overflow-x: auto" in CSS      # wide tables scroll instead of growing tall


def test_chart_legends_sit_above_the_plot_so_they_never_cover_bars():
    from src.app_state import load_baseline
    from src.portfolio_view import coverage_figure, project_summary
    baseline = load_baseline()
    fig = coverage_figure(project_summary(baseline["data"], baseline["snapshot"]))
    assert fig.layout.legend.yanchor == "bottom" and fig.layout.legend.y > 1.0
    assert fig.layout.margin.t >= 34                                  # room for a legend that wraps on a phone

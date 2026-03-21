from memetalk.app.ui import _THEME_CSS, _build_theme_css


def test_theme_css_contains_explicit_dark_mode_overrides() -> None:
    css = _build_theme_css("dark")

    assert "color-scheme: dark" in css
    assert "--mt-bg-top: #111721;" in css
    assert "--mt-input-bg: rgba(13, 18, 27, 0.94);" in css


def test_theme_css_styles_shared_shell_components_in_both_themes() -> None:
    css = _build_theme_css("light")

    assert ".stApp {" in _THEME_CSS
    assert "color-scheme: light" in _THEME_CSS
    assert "--mt-input-bg" in css
    assert "--mt-button-secondary-bg" in css
    assert '[data-testid="stAppViewContainer"]' in css
    assert '[data-testid="stMainBlockContainer"]' in css
    assert "background: transparent !important;" in css
    assert 'div[data-testid="stMetric"]' in css
    assert 'div[data-testid="stExpander"]' in css

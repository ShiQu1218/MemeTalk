from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import streamlit as st

_LIGHT_THEME_VARS = {
    "--mt-font-stack": '"Aptos", "Segoe UI", "Microsoft JhengHei", sans-serif',
    "--mt-bg-top": "#fbf2e8",
    "--mt-bg-bottom": "#f4f0ea",
    "--mt-surface": "rgba(255, 252, 248, 0.92)",
    "--mt-surface-strong": "rgba(255, 255, 255, 0.98)",
    "--mt-surface-soft": "rgba(255, 248, 243, 0.95)",
    "--mt-border": "rgba(115, 92, 75, 0.14)",
    "--mt-border-strong": "rgba(115, 92, 75, 0.22)",
    "--mt-shadow": "0 18px 48px rgba(83, 57, 44, 0.08)",
    "--mt-ink": "#22303a",
    "--mt-muted": "#5d6874",
    "--mt-accent": "#c96742",
    "--mt-accent-soft": "rgba(201, 103, 66, 0.12)",
    "--mt-accent-strong": "#215e63",
    "--mt-success": "#2d7a58",
    "--mt-warning": "#b86a1b",
    "--mt-danger": "#b84d4d",
    "--mt-button-secondary-bg": "rgba(255, 255, 255, 0.95)",
    "--mt-button-secondary-ink": "#22303a",
    "--mt-input-bg": "rgba(255, 255, 255, 0.96)",
    "--mt-code-bg": "rgba(233, 221, 212, 0.52)",
    "--mt-overlay-top": "rgba(201, 103, 66, 0.14)",
    "--mt-overlay-right": "rgba(33, 94, 99, 0.10)",
    "--mt-hero-top": "rgba(255, 255, 255, 0.97)",
    "--mt-hero-bottom": "rgba(255, 245, 235, 0.92)",
    "--mt-info-band": "rgba(33, 94, 99, 0.18)",
    "--mt-success-band": "rgba(45, 122, 88, 0.18)",
    "--mt-warning-band": "rgba(184, 106, 27, 0.18)",
    "--mt-danger-band": "rgba(184, 77, 77, 0.16)",
    "--mt-button-shadow": "0 10px 24px rgba(83, 57, 44, 0.08)",
    "--mt-button-shadow-hover": "0 14px 28px rgba(83, 57, 44, 0.12)",
    "--mt-sidebar-bg": "rgba(255, 251, 246, 0.88)",
}

_DARK_THEME_VARS = {
    "--mt-font-stack": '"Aptos", "Segoe UI", "Microsoft JhengHei", sans-serif',
    "--mt-bg-top": "#111721",
    "--mt-bg-bottom": "#0a0f17",
    "--mt-surface": "rgba(18, 24, 34, 0.9)",
    "--mt-surface-strong": "rgba(23, 30, 42, 0.96)",
    "--mt-surface-soft": "rgba(29, 37, 51, 0.94)",
    "--mt-border": "rgba(240, 138, 93, 0.16)",
    "--mt-border-strong": "rgba(240, 138, 93, 0.28)",
    "--mt-shadow": "0 22px 56px rgba(0, 0, 0, 0.34)",
    "--mt-ink": "#f4ede4",
    "--mt-muted": "#c8c0b7",
    "--mt-accent": "#f08a5d",
    "--mt-accent-soft": "rgba(240, 138, 93, 0.16)",
    "--mt-accent-strong": "#8dc9cc",
    "--mt-success": "#7fc4a0",
    "--mt-warning": "#efbb78",
    "--mt-danger": "#ef8f8f",
    "--mt-button-secondary-bg": "rgba(26, 33, 45, 0.96)",
    "--mt-button-secondary-ink": "#f4ede4",
    "--mt-input-bg": "rgba(13, 18, 27, 0.94)",
    "--mt-code-bg": "rgba(58, 69, 88, 0.5)",
    "--mt-overlay-top": "rgba(240, 138, 93, 0.16)",
    "--mt-overlay-right": "rgba(141, 201, 204, 0.12)",
    "--mt-hero-top": "rgba(24, 32, 46, 0.97)",
    "--mt-hero-bottom": "rgba(15, 21, 31, 0.95)",
    "--mt-info-band": "rgba(141, 201, 204, 0.22)",
    "--mt-success-band": "rgba(127, 196, 160, 0.22)",
    "--mt-warning-band": "rgba(239, 187, 120, 0.22)",
    "--mt-danger-band": "rgba(239, 143, 143, 0.2)",
    "--mt-button-shadow": "0 12px 26px rgba(0, 0, 0, 0.28)",
    "--mt-button-shadow-hover": "0 16px 32px rgba(0, 0, 0, 0.36)",
    "--mt-sidebar-bg": "rgba(16, 22, 31, 0.88)",
}


def _theme_var_block(theme_type: str) -> str:
    values = _DARK_THEME_VARS if theme_type == "dark" else _LIGHT_THEME_VARS
    return "\n".join(f"  {name}: {value};" for name, value in values.items())


def _resolve_theme_type() -> str | None:
    try:
        theme = st.context.theme
    except Exception:
        return None
    theme_type = getattr(theme, "type", None)
    return theme_type if theme_type in {"light", "dark"} else None


def _build_theme_css(theme_type: str | None = None) -> str:
    resolved_theme = theme_type or _resolve_theme_type() or "light"
    fallback_media = ""
    if theme_type is None:
        fallback_media = """
@media (prefers-color-scheme: dark) {
  :root,
  .stApp {
    color-scheme: dark;
  }
}
"""
    return f"""
<style>
:root,
.stApp {{
{_theme_var_block(resolved_theme)}
  color-scheme: {resolved_theme};
}}
{fallback_media}

html, body, [class*="css"] {{
  font-family: var(--mt-font-stack);
  color: var(--mt-ink);
}}

html,
body {{
  background-color: var(--mt-bg-bottom) !important;
}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
section.main {{
  background-color: var(--mt-bg-bottom) !important;
  background:
    radial-gradient(circle at top left, var(--mt-overlay-top), transparent 34%),
    radial-gradient(circle at top right, var(--mt-overlay-right), transparent 28%),
    linear-gradient(180deg, var(--mt-bg-top) 0%, var(--mt-bg-bottom) 100%) !important;
}}

[data-testid="stMainBlockContainer"],
.main .block-container {{
  background-color: transparent !important;
}}

[data-testid="stMain"] > div,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
  background: transparent !important;
}}

section[data-testid="stSidebar"] > div {{
  background: var(--mt-sidebar-bg);
  border-right: 1px solid var(--mt-border);
}}

.block-container {{
  max-width: 1180px;
  padding-top: 2rem;
  padding-bottom: 2.5rem;
}}

h1, h2, h3 {{
  color: var(--mt-ink);
  letter-spacing: -0.01em;
}}

p, li, label, .stCaption {{
  color: var(--mt-muted);
}}

.mt-hero {{
  border: 1px solid var(--mt-border);
  background:
    linear-gradient(135deg, var(--mt-hero-top), var(--mt-hero-bottom)),
    var(--mt-surface);
  border-radius: 28px;
  box-shadow: var(--mt-shadow);
  padding: 1.75rem 1.5rem;
  margin-bottom: 1.1rem;
}}

.mt-hero-eyebrow {{
  color: var(--mt-accent-strong);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.45rem;
}}

.mt-hero-title {{
  color: var(--mt-ink);
  font-size: 2.15rem;
  font-weight: 700;
  margin: 0;
}}

.mt-hero-subtitle {{
  color: var(--mt-muted);
  font-size: 1rem;
  line-height: 1.7;
  margin: 0.6rem 0 0;
}}

.mt-chip-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}}

.mt-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  background: var(--mt-accent-soft);
  color: var(--mt-accent-strong);
  border-radius: 999px;
  padding: 0.38rem 0.78rem;
  font-size: 0.88rem;
  font-weight: 600;
}}

.mt-section {{
  margin: 0.35rem 0 0.7rem;
}}

.mt-section h2 {{
  font-size: 1.22rem;
  margin-bottom: 0.15rem;
}}

.mt-section p {{
  margin: 0;
  line-height: 1.6;
}}

.mt-inline-card {{
  border: 1px solid var(--mt-border);
  background: var(--mt-surface-strong);
  border-radius: 18px;
  box-shadow: var(--mt-shadow);
  padding: 1rem 1.05rem;
  margin: 0.25rem 0 1rem;
}}

.mt-inline-card strong {{
  display: block;
  color: var(--mt-ink);
  margin-bottom: 0.25rem;
}}

.mt-inline-card.info {{
  border-left: 4px solid var(--mt-accent-strong);
  background:
    linear-gradient(135deg, var(--mt-info-band), transparent 42%),
    var(--mt-surface-strong);
}}

.mt-inline-card.success {{
  border-left: 4px solid var(--mt-success);
  background:
    linear-gradient(135deg, var(--mt-success-band), transparent 42%),
    var(--mt-surface-strong);
}}

.mt-inline-card.warning {{
  border-left: 4px solid var(--mt-warning);
  background:
    linear-gradient(135deg, var(--mt-warning-band), transparent 42%),
    var(--mt-surface-strong);
}}

.mt-inline-card.danger {{
  border-left: 4px solid var(--mt-danger);
  background:
    linear-gradient(135deg, var(--mt-danger-band), transparent 42%),
    var(--mt-surface-strong);
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 22px;
  border: 1px solid var(--mt-border);
  background: linear-gradient(180deg, var(--mt-surface-strong), var(--mt-surface-soft));
  box-shadow: var(--mt-shadow);
}}

div[data-testid="stMetric"] {{
  background: linear-gradient(180deg, var(--mt-surface-strong), var(--mt-surface-soft));
  border: 1px solid var(--mt-border);
  border-radius: 20px;
  padding: 0.85rem 1rem;
  box-shadow: var(--mt-shadow);
}}

div[data-testid="stMetric"] label {{
  color: var(--mt-muted);
  font-weight: 600;
}}

div[data-testid="stMetricValue"] {{
  color: var(--mt-ink);
}}

.stButton > button,
.stDownloadButton > button {{
  border-radius: 999px;
  border: 1px solid var(--mt-border-strong);
  box-shadow: var(--mt-button-shadow);
  font-weight: 600;
  transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: var(--mt-button-shadow-hover);
  border-color: rgba(240, 138, 93, 0.34);
}}

.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, var(--mt-accent) 0%, #df8a5c 100%);
  color: #fff7f3;
}}

.stButton > button[kind="secondary"] {{
  background: var(--mt-button-secondary-bg);
  color: var(--mt-button-secondary-ink);
}}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
.stNumberInput input,
div[data-baseweb="base-input"] > div {{
  border-radius: 16px;
  border: 1px solid var(--mt-border);
  background: var(--mt-input-bg);
  color: var(--mt-ink);
}}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
div[data-baseweb="base-input"] > div:focus-within {{
  border-color: rgba(201, 103, 66, 0.55);
  box-shadow: 0 0 0 1px rgba(201, 103, 66, 0.12);
}}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder {{
  color: var(--mt-muted);
}}

div[data-testid="stExpander"] {{
  border-radius: 18px;
  border: 1px solid var(--mt-border);
  background: var(--mt-surface-strong);
}}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {{
  color: var(--mt-ink);
}}

.stMarkdown code,
.stCaption code {{
  background: var(--mt-code-bg);
  border-radius: 8px;
  padding: 0.14rem 0.38rem;
}}

.stAlert,
div[data-testid="stStatusWidget"] {{
  border-radius: 18px;
  border: 1px solid var(--mt-border);
}}

div[data-baseweb="select"] *,
div[data-baseweb="base-input"] * {{
  color: var(--mt-ink);
}}

hr {{
  border-color: var(--mt-border);
}}

/* --- Dark mode: override Streamlit internal white backgrounds --- */

.stApp [data-testid="stForm"],
.stApp .stTabs [data-baseweb="tab-panel"],
.stApp .stTabs [data-baseweb="tab-list"],
.stApp [data-testid="stBottom"],
.stApp [data-testid="stBottom"] > div,
.stApp [data-testid="stStatusWidget"],
.stApp .stSpinner > div,
.stApp [data-testid="stImage"],
.stApp [data-testid="stCaptionContainer"],
.stApp .stRadio > div,
.stApp .stCheckbox > label,
.stApp .stSelectbox > div,
.stApp .stMultiSelect > div {{
  background: transparent !important;
}}

.stApp .stAlert {{
  background: var(--mt-surface-soft) !important;
  color: var(--mt-ink);
}}

.stApp .stAlert p,
.stApp .stAlert span {{
  color: var(--mt-ink) !important;
}}

.stApp .stSpinner > div > span {{
  color: var(--mt-muted) !important;
}}

.stApp .stMarkdown,
.stApp .stMarkdown p,
.stApp .stMarkdown li,
.stApp .stMarkdown span,
.stApp .stCaption {{
  color: var(--mt-muted);
}}

.stApp .stMarkdown h1,
.stApp .stMarkdown h2,
.stApp .stMarkdown h3,
.stApp .stMarkdown strong {{
  color: var(--mt-ink);
}}

div[data-testid="stVerticalBlockBorderWrapper"] > div {{
  background: transparent !important;
}}

div[data-baseweb="select"] [data-baseweb="popover"],
div[data-baseweb="select"] ul,
div[data-baseweb="select"] li {{
  background: var(--mt-surface-strong) !important;
  color: var(--mt-ink) !important;
}}

div[data-baseweb="select"] li:hover {{
  background: var(--mt-accent-soft) !important;
}}

.stApp .stSuccess,
.stApp [data-testid="stNotification"] {{
  background: var(--mt-surface-soft) !important;
  color: var(--mt-ink) !important;
}}

.stApp .stSuccess p,
.stApp [data-testid="stNotification"] p {{
  color: var(--mt-ink) !important;
}}

div[data-testid="stMetricValue"] > div {{
  color: var(--mt-ink) !important;
}}
</style>
"""


_THEME_CSS = _build_theme_css("light")


def setup_page(
    *,
    page_title: str,
    page_icon: str,
    title: str,
    subtitle: str,
    eyebrow: str | None = None,
    chips: Iterable[str] = (),
) -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    st.markdown(_build_theme_css(), unsafe_allow_html=True)
    render_hero(title=title, subtitle=subtitle, eyebrow=eyebrow, chips=chips)


def render_hero(
    *,
    title: str,
    subtitle: str,
    eyebrow: str | None = None,
    chips: Iterable[str] = (),
) -> None:
    eyebrow_html = ""
    if eyebrow:
        eyebrow_html = f"<div class='mt-hero-eyebrow'>{escape(eyebrow)}</div>"
    chip_html = "".join(f"<span class='mt-chip'>{escape(chip)}</span>" for chip in chips)
    chip_row_html = f"<div class='mt-chip-row'>{chip_html}</div>" if chip_html else ""
    st.markdown(
        (
            "<section class='mt-hero'>"
            f"{eyebrow_html}"
            f"<h1 class='mt-hero-title'>{escape(title)}</h1>"
            f"<p class='mt-hero-subtitle'>{escape(subtitle)}</p>"
            f"{chip_row_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_section(title: str, description: str | None = None) -> None:
    description_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f"<section class='mt-section'><h2>{escape(title)}</h2>{description_html}</section>",
        unsafe_allow_html=True,
    )


def render_notice(title: str, body: str, *, tone: str = "info") -> None:
    safe_tone = tone if tone in {"info", "success", "warning", "danger"} else "info"
    st.markdown(
        (
            f"<div class='mt-inline-card {safe_tone}'>"
            f"<strong>{escape(title)}</strong>"
            f"<span>{escape(body)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def format_path(value: str | Path | None) -> str:
    if value is None:
        return "未設定"
    text = str(value).strip()
    return text or "未設定"

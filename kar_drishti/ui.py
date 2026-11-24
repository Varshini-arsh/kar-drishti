"""Small UI kit for the Streamlit app: CSS, KPI cards, badges, finding cards."""
from __future__ import annotations

import html

import streamlit as st

INK, MUTED, BORDER, SOFT = "#1C1917", "#78716C", "#E7E5E4", "#FAFAF9"
TEAL, TEAL_SOFT, AMBER, RED, STONE = "#0F766E", "#F0FDFA", "#D97706", "#DC2626", "#A8A29E"
NAVY = TEAL  # kept for older imports
LEVEL_COLOR = {"HIGH": RED, "MEDIUM": AMBER, "LOW": STONE}
LEVEL_BG = {"HIGH": "#FEF2F2", "MEDIUM": "#FFFBEB", "LOW": "#F5F5F4"}

CSS = f"""
<style>
section[data-testid="stSidebar"] {{ width: 240px !important; min-width: 240px !important; }}
.block-container {{ max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {BORDER}; border-radius: 12px; padding: 8px 10px 4px; background: #fff; }}
div[data-testid="stExpander"] details {{ border-radius: 10px; }}

.kd-head {{ margin: 0 0 1.4rem; }}
.kd-head h1 {{ font-size: 1.7rem; font-weight: 700; margin: 0; padding: 0; color: {INK}; }}
.kd-head p {{ color: {MUTED}; margin: .3rem 0 0; font-size: .95rem; }}
.kd-section {{ font-size: 1.02rem; font-weight: 650; color: {INK}; margin: 1.6rem 0 .6rem; }}

.kd-kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 0 0 .4rem; }}
@media (max-width: 900px) {{ .kd-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kd-kpi {{ border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; background: #fff; min-width: 0; }}
.kd-kpi .l {{ font-size: .82rem; color: {MUTED}; font-weight: 500; }}
.kd-kpi .v {{ font-size: 1.6rem; font-weight: 700; color: {INK}; margin-top: 2px; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }}
.kd-kpi .s {{ font-size: .8rem; color: {MUTED}; margin-top: 2px; }}
.kd-kpi.red .v {{ color: {RED}; }} .kd-kpi.amber .v {{ color: {AMBER}; }} .kd-kpi.teal .v {{ color: {TEAL}; }}

.kd-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 700;
            letter-spacing: .03em; vertical-align: middle; margin-left: 6px; }}
.kd-chip {{ display: inline-block; padding: 2px 9px; margin: 0 6px 6px 0; border-radius: 999px; font-size: .76rem;
           background: {TEAL_SOFT}; color: {TEAL}; font-weight: 600; }}

.kd-find {{ border: 1px solid {BORDER}; border-left: 4px solid {RED}; border-radius: 10px; padding: 12px 16px;
           background: #fff; margin-bottom: 8px; }}
.kd-find .t {{ font-weight: 650; color: {INK}; font-size: .97rem; }}
.kd-find .d {{ color: #44403C; font-size: .92rem; margin-top: 4px; line-height: 1.5; }}
.kd-note {{ background: {SOFT}; border: 1px solid {BORDER}; border-left: 4px solid {TEAL}; border-radius: 10px;
           padding: 12px 16px; font-size: .93rem; color: #292524; line-height: 1.55; margin: 6px 0 14px; }}
.kd-ai {{ font-size: .82rem; color: {MUTED}; padding: 9px 11px; border: 1px solid {BORDER}; border-radius: 10px;
         background: #fff; margin-top: .3rem; line-height: 1.4; }}
.kd-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
.kd-profile {{ margin: .2rem 0 1rem; }}
.kd-profile .n {{ font-size: 1.5rem; font-weight: 700; color: {INK}; }}
.kd-profile .sub {{ color: {MUTED}; font-size: .9rem; margin-top: 2px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def header(title: str, subtitle: str = ""):
    st.markdown(f"<div class='kd-head'><h1>{esc(title)}</h1>"
                + (f"<p>{esc(subtitle)}</p>" if subtitle else "") + "</div>", unsafe_allow_html=True)


def section(title: str):
    st.markdown(f"<div class='kd-section'>{esc(title)}</div>", unsafe_allow_html=True)


def kpis(items: list[tuple]):
    """items: (label, value, sub=None, tone=None). Up to 4 per row."""
    cells = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else None
        tone = it[3] if len(it) > 3 else ""
        cells.append(f"<div class='kd-kpi {tone or ''}'><div class='l'>{esc(label)}</div>"
                     f"<div class='v'>{esc(value)}</div>" + (f"<div class='s'>{esc(sub)}</div>" if sub else "")
                     + "</div>")
    cols = min(4, max(1, len(cells)))
    st.markdown(f"<div class='kd-kpis' style='grid-template-columns:repeat({cols},minmax(0,1fr))'>{''.join(cells)}</div>",
                unsafe_allow_html=True)


def badge(level: str) -> str:
    return (f"<span class='kd-badge' style='background:{LEVEL_BG.get(level, '#F5F5F4')};"
            f"color:{LEVEL_COLOR.get(level, STONE)}'>{esc(level)}</span>")


def finding_card(title: str, text: str, severity: float, meta: str = ""):
    color = RED if severity >= 0.75 else AMBER if severity >= 0.5 else STONE
    st.markdown(f"<div class='kd-find' style='border-left-color:{color}'>"
                f"<div class='t'>{esc(title)}</div><div class='d'>{esc(text)}</div></div>", unsafe_allow_html=True)


def note(text: str):
    st.markdown(f"<div class='kd-note'>{esc(text).replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)


def ai_status(ok: bool, model: str, msg: str):
    color = "#16A34A" if ok else AMBER
    label = f"AI model: {model}" if ok else "AI model: offline"
    st.markdown(f"<div class='kd-ai'><span class='kd-dot' style='background:{color}'></span>{esc(label)}"
                f"<div style='margin-top:3px'>{esc('Running on this computer' if ok else msg)}</div></div>",
                unsafe_allow_html=True)


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

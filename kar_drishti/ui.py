"""Small UI kit for the Streamlit app: CSS, KPI cards, badges, finding cards.

Colours come from branding.py. The styles are theme-neutral (transparent cards,
see-through borders, inherited text colour) so they read well in light and dark mode.
"""
from __future__ import annotations

import html

import streamlit as st

import branding

TEAL, TEAL_SOFT = branding.ACCENT, "rgba(20,184,166,.14)"
RED, AMBER, STONE = branding.RISK_COLOURS["HIGH"], branding.RISK_COLOURS["MEDIUM"], branding.RISK_COLOURS["LOW"]
BORDER = "rgba(120,113,108,.28)"     # works on white and on dark backgrounds
SOFT = "rgba(120,113,108,.08)"
MUTED = "#8A8580"                    # mid grey, readable in both modes
INK = "inherit"
NAVY = TEAL  # kept for older imports
LEVEL_COLOR = {"HIGH": RED, "MEDIUM": AMBER, "LOW": STONE}
LEVEL_BG = {"HIGH": "rgba(220,38,38,.12)", "MEDIUM": "rgba(217,119,6,.14)", "LOW": "rgba(168,162,158,.18)"}

CSS = f"""
<style>
section[data-testid="stSidebar"] {{ width: 240px !important; min-width: 240px !important; }}
.block-container {{ max-width: 1100px; padding-top: 3.2rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {BORDER}; border-radius: 12px; padding: 8px 10px 4px; background: transparent; }}
div[data-testid="stExpander"] details {{ border-radius: 10px; }}

.kd-head {{ margin: 0 0 1.4rem; }}
.kd-head h1 {{ font-size: 1.7rem; font-weight: 700; margin: 0; padding: 0; }}
.kd-head p {{ color: {MUTED}; margin: .3rem 0 0; font-size: .95rem; }}
.kd-section {{ font-size: 1.02rem; font-weight: 650; color: {INK}; margin: 1.6rem 0 .6rem; }}

.kd-kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 0 0 .4rem; }}
@media (max-width: 900px) {{ .kd-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kd-kpi {{ border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; background: transparent; min-width: 0; }}
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
           background: transparent; margin-bottom: 8px; }}
.kd-find .t {{ font-weight: 650; color: {INK}; font-size: .97rem; }}
.kd-find .d {{ color: inherit; opacity: .88; font-size: .92rem; margin-top: 4px; line-height: 1.5; }}
.kd-note {{ background: {SOFT}; border: 1px solid {BORDER}; border-left: 4px solid {TEAL}; border-radius: 10px;
           padding: 12px 16px; font-size: .93rem; color: inherit; line-height: 1.55; margin: 6px 0 14px; }}
.kd-ai {{ font-size: .82rem; color: {MUTED}; padding: 9px 11px; border: 1px solid {BORDER}; border-radius: 10px;
         background: transparent; margin-top: .3rem; line-height: 1.4; }}
.kd-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
.kd-profile {{ margin: .2rem 0 1rem; }}
.kd-score {{ border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; background: transparent; margin: .4rem 0 .8rem; }}
.kd-score .h {{ font-size: .82rem; color: {MUTED}; font-weight: 600; margin-bottom: 6px; }}
.kd-score .r {{ display: grid; grid-template-columns: minmax(0, 1fr) 140px 64px; gap: 12px; align-items: center;
  padding: 5px 0; border-top: 1px solid {SOFT}; font-size: .9rem; }}
.kd-score .r:first-of-type {{ border-top: 0; }}
.kd-score .bar {{ height: 6px; border-radius: 3px; background: {BORDER}; position: relative; overflow: hidden; }}
.kd-score .bar b {{ position: absolute; left: 0; top: 0; bottom: 0; background: {TEAL}; border-radius: 3px; }}
.kd-score .p {{ text-align: right; font-weight: 650; font-variant-numeric: tabular-nums; }}
.kd-score .tot {{ display: flex; justify-content: space-between; border-top: 1px solid {BORDER}; margin-top: 6px;
  padding-top: 8px; font-size: .9rem; font-weight: 650; }}
.kd-score .tot span {{ color: {MUTED}; font-weight: 500; }}
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


def score_breakdown(breakdown: list, points: float, score: float, floored: bool, high: int, medium: int):
    """'Why this score' card: points per red flag (weight x strength), total, cap / floor."""
    if not breakdown:
        st.markdown("<div class='kd-score'><div class='h'>WHY THIS SCORE</div>No red flags were found, so the score "
                    "is 0.</div>", unsafe_allow_html=True)
        return
    rows = []
    for b in breakdown:
        pct = 100 * b["points"] / max(b["weight"], 1)
        rows.append(f"<div class='r'><div>{esc(b['label'])}</div>"
                    f"<div title='strength {b['strength']:.2f} of max {b['weight']} points'>"
                    f"<div class='bar'><b style='width:{pct:.0f}%'></b></div></div>"
                    f"<div class='p'>+{b['points']:.0f} <span style='color:{MUTED};font-weight:400'>/ {b['weight']}"
                    f"</span></div></div>")
    if points > 100:
        end = f"{points:.0f} points <span>→ capped at 100</span>"
    elif floored:
        end = f"{points:.0f} points <span>→ raised to {medium}: every red flag must be reviewed</span>"
    else:
        end = f"{points:.0f} points"
    st.markdown(f"<div class='kd-score'><div class='h'>WHY THIS SCORE · each red flag adds up to its maximum points, "
                f"scaled by how strong the evidence is</div>{''.join(rows)}"
                f"<div class='tot'><div>Total: {end}</div><div>Score {score:.0f} / 100 "
                f"<span>(High ≥ {high}, Medium ≥ {medium})</span></div></div></div>", unsafe_allow_html=True)


def quote_card(q: dict):
    """An exact quote from a loaded document: document, number, date, the sentence(s)."""
    later = " · <b>later document</b>" if q.get("later") else ""
    st.markdown(f"<div class='kd-note'><div style='font-size:.8rem;color:{MUTED};margin-bottom:4px'>"
                f"{esc(q.get('doc', ''))} · No. {esc(q.get('number', ''))} · {esc(q.get('date', ''))}{later}</div>"
                f"“{esc(q.get('text', ''))}”</div>", unsafe_allow_html=True)

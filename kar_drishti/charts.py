"""Plotly charts used by the dashboard and the agent (minimal style, no toolbars)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from .data import Case

NAVY, TEAL, AMBER, RED, SLATE, INK, MUTED = "#0F766E", "#14B8A6", "#D97706", "#DC2626", "#A8A29E", "#1C1917", "#78716C"
FONT = "Source Sans Pro, Segoe UI, Arial, sans-serif"


def _layout(fig, h=320, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=4, r=12, t=8, b=4), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT, size=12, color="#44403C"), hoverlabel=dict(font_family=FONT, bgcolor="white"),
        showlegend=legend, legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        bargap=0.35)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E7E5E4", ticks="", automargin=True)
    fig.update_yaxes(gridcolor="#F5F5F4", zeroline=False, ticks="", automargin=True)
    return fig


def monthly_flows(case: Case, holder_id: str):
    tx = case.transactions[case.transactions.holder_id == holder_id].copy()
    tx["month"] = tx.txn_date.dt.to_period("M").dt.to_timestamp()
    g = tx.groupby(["month", "direction"])["amount"].sum().unstack(fill_value=0).reindex(columns=["CR", "DR"], fill_value=0)
    fig = go.Figure()
    fig.add_bar(x=g.index, y=g["CR"] / 1e5, name="Money in", marker_color=NAVY, marker_cornerradius=4,
                hovertemplate="%{x|%b %Y}<br>In: ₹%{y:,.2f} L<extra></extra>")
    fig.add_bar(x=g.index, y=g["DR"] / 1e5, name="Money out", marker_color="#D6D3D1", marker_cornerradius=4,
                hovertemplate="%{x|%b %Y}<br>Out: ₹%{y:,.2f} L<extra></extra>")
    fig.update_yaxes(title=dict(text="₹ lakh", font=dict(size=11, color=MUTED)))
    fig.update_xaxes(tickformat="%b")
    return _layout(fig, 300)


def cash_deposits_scatter(case: Case, holder_id: str):
    tx = case.transactions
    c = tx[(tx.holder_id == holder_id) & (tx["mode"] == "CASH") & (tx.direction == "CR")]
    if c.empty:
        return None
    fig = go.Figure(go.Scatter(x=c.txn_date, y=c.amount, mode="markers",
                               marker=dict(size=8, color=NAVY, opacity=.75, line=dict(width=1, color="white")),
                               customdata=c[["txn_id", "account_id"]],
                               hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=config.CASH_PAN_THRESHOLD, line_dash="dot", line_color=RED, line_width=1.5,
                  annotation_text="₹50,000 PAN limit", annotation_position="top left",
                  annotation_font=dict(color=RED, size=11))
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    fig.update_xaxes(tickformat="%b")
    return _layout(fig, 300, legend=False)


def sankey(edges: pd.DataFrame, title=None):
    if edges.empty:
        return None
    labels = sorted(set(edges.source) | set(edges.target), key=lambda s: (s.split("|")[0], s))
    idx = {l: i for i, l in enumerate(labels)}
    disp = [l.split("|", 1)[1].replace("[outside] ", "↗ ") for l in labels]
    start = labels[0].split("|", 1)[1] if labels else ""
    colors = [RED if l.split("|", 1)[1].startswith("[outside]") else (NAVY if d == start else TEAL)
              for l, d in zip(labels, disp)]
    link_colors = ["rgba(220,38,38,.22)" if t.split("|", 1)[1] == start else "rgba(15,118,110,.16)"
                   for t in edges.target]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=disp, pad=22, thickness=14, color=colors, line=dict(width=0)),
        link=dict(source=edges.source.map(idx), target=edges.target.map(idx), value=edges.amount, color=link_colors,
                  customdata=[f"₹{a / 1e5:,.2f} L in {t} txn(s)" for a, t in zip(edges.amount, edges.txns)],
                  hovertemplate="%{source.label} → %{target.label}<br>%{customdata}<extra></extra>")))
    return _layout(fig, 460, legend=False)


def risk_donut(risk: pd.DataFrame):
    order = ["HIGH", "MEDIUM", "LOW"]
    g = risk.risk_level.value_counts().reindex(order, fill_value=0)
    flagged = int(g["HIGH"] + g["MEDIUM"])
    fig = go.Figure(go.Pie(labels=["High", "Medium", "Low"], values=g.values, hole=.72, sort=False,
                           marker=dict(colors=[RED, AMBER, "#E7E5E4"], line=dict(color="white", width=2)),
                           textinfo="none", hovertemplate="%{label}: %{value} holders<extra></extra>"))
    fig.add_annotation(text=f"<b style='font-size:26px;color:{INK}'>{flagged}</b><br>"
                            f"<span style='font-size:12px;color:{MUTED}'>need review</span>",
                       showarrow=False, x=.5, y=.5)
    fig = _layout(fig, 280)
    fig.update_layout(legend=dict(orientation="h", y=-0.05, x=.5, xanchor="center"))
    return fig


def pattern_counts(findings: pd.DataFrame, labels: dict):
    g = findings.pattern.map(lambda p: labels.get(p, p)).value_counts().sort_values()
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=NAVY, marker_cornerradius=4,
                           text=g.values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x} holders<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, max(g.values) * 1.15 if len(g) else 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color="#44403C"))
    return _layout(fig, max(280, 26 * len(g) + 30), legend=False)


def gst_monthly(r: pd.DataFrame):
    fig = go.Figure()
    fig.add_bar(x=r.period, y=r.output_tax / 1e5, name="Output tax", marker_color=NAVY, marker_cornerradius=3)
    fig.add_bar(x=r.period, y=r.itc_claimed / 1e5, name="ITC claimed", marker_color="#D6D3D1", marker_cornerradius=3)
    fig.add_scatter(x=r.period, y=r.tax_paid_cash / 1e5, name="Paid in cash", mode="lines+markers",
                    line=dict(color=RED, width=2), marker=dict(size=6))
    fig.update_yaxes(title=dict(text="₹ lakh", font=dict(size=11, color=MUTED)))
    fig.update_layout(barmode="group")
    return _layout(fig, 320)


def auto_chart(df: pd.DataFrame):
    """Pick a sensible chart for an SQL result, or None."""
    if df is None or df.empty or len(df.columns) < 2 or len(df) < 2:
        return None
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.lower().endswith("_id")]
    if not num:
        return None
    y = num[0]
    for c in df.columns:
        if c == y:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or "month" in c.lower() \
                or "period" in c.lower():
            try:
                x = pd.to_datetime(df[c])
                d = df.assign(**{c: x}).sort_values(c)
                fig = go.Figure(go.Bar(x=d[c], y=d[y], marker_color=NAVY, marker_cornerradius=4,
                                       hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>"))
                fig.update_xaxes(tickformat="%b %Y")
                fig.update_yaxes(title=dict(text=y.replace("_", " "), font=dict(size=11, color=MUTED)))
                return _layout(fig, 300, legend=False)
            except Exception:
                pass
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if cat and len(df) <= 25:
        label = next((c for c in cat if "name" in c.lower()), cat[0])
        d = df.sort_values(y)
        fig = go.Figure(go.Bar(x=d[y], y=d[label].astype(str), orientation="h", marker_color=NAVY,
                               marker_cornerradius=4, hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
        fig.update_xaxes(title=dict(text=y.replace("_", " "), font=dict(size=11, color=MUTED)), gridcolor="#F5F5F4")
        fig.update_yaxes(showgrid=False)
        return _layout(fig, max(260, 24 * len(d) + 40), legend=False)
    return None


# backwards-compatible name
risk_distribution = risk_donut

"""Charts for Pragya screens (same quiet style as the Kar Drishti charts, light + dark friendly)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import branding
from kar_drishti.charts import _layout

GREY = "#A8A29E"


def revenue_chart(rev, where: dict | None = None):
    """Cumulative collections this year vs target, with the year-end projection and last year for comparison."""
    m = rev.monthly(where)
    s = rev.summary(where)
    x = [pd.Timestamp(v + "-01") for v in m["month"]]
    done = m["collected"].notna()
    cum = m["collected"].fillna(0).cumsum() / 1e7
    ly = m["last_year"].cumsum() / 1e7
    fig = go.Figure()
    fig.add_scatter(x=x, y=ly, name="Last year", mode="lines", line=dict(color=GREY, width=2),
                    hovertemplate="%{x|%b}: ₹%{y:,.2f} crore<extra>last year</extra>")
    fig.add_scatter(x=[d for d, k in zip(x, done) if k], y=cum[done], name="Collected", mode="lines+markers",
                    line=dict(color=branding.ACCENT, width=3), marker=dict(size=6),
                    hovertemplate="%{x|%b %Y}: ₹%{y:,.2f} crore<extra>collected</extra>")
    if (~done).any() and done.any():
        last_i = int(done[done].index[-1])
        # projected path: remaining months follow last year's monthly pattern, scaled to reach the projection
        rest_ly = m.loc[last_i + 1:, "last_year"]
        remaining = s["projected"] - s["collected"]
        steps = rest_ly / rest_ly.sum() * remaining if rest_ly.sum() else pd.Series(remaining / len(rest_ly), rest_ly.index)
        proj = [cum.iloc[last_i]] + list((s["collected"] + steps.cumsum()) / 1e7)
        fig.add_scatter(x=x[last_i:], y=proj, name="Projected", mode="lines",
                        line=dict(color=branding.ACCENT, width=2, dash="dot"),
                        hovertemplate="%{x|%b %Y}: ₹%{y:,.2f} crore<extra>projected</extra>")
    fig.add_hline(y=s["target"] / 1e7, line=dict(color=branding.RISK_COLOURS["HIGH"], width=1.5, dash="dash"),
                  annotation_text=f"Target ₹{s['target'] / 1e7:,.0f} crore", annotation_position="top left",
                  annotation_font=dict(size=11, color=branding.RISK_COLOURS["HIGH"]))
    fig.update_yaxes(title=dict(text="₹ crore, cumulative", font=dict(size=11, color="#8A8580")))
    fig.update_xaxes(tickformat="%b")
    fig = _layout(fig, 330)
    fig.update_layout(margin=dict(l=4, r=12, t=8, b=28))
    return fig

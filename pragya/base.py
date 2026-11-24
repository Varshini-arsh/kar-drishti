"""Shared pieces for Pragya's agents: the result every specialist returns, and number formatting."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CANT_ANSWER = "I can't answer this from the loaded data. Here is what I can do:"


@dataclass
class AgentResult:
    """What a specialist agent hands back to the Lead agent.

    facts:     the tool output, as plain lines. The answer may only use what is here.
    template:  a ready-made answer written by code from the same facts (used without a model,
               or when the model's answer fails the checker).
    checked:   what was looked at, e.g. "177 people and firms, 47,996 transactions".
    """
    agent: str
    tool: str
    ok: bool
    facts: list = field(default_factory=list)
    template: str = ""
    checked: str = ""
    records: pd.DataFrame | None = None
    fig: object = None
    html: str | None = None
    sql: str | None = None
    quotes: list = field(default_factory=list)          # Law agent: [{doc, number, date, text}]
    must: list = field(default_factory=list)            # facts a model-written answer MUST contain word for word
    extra: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def sources(self) -> list[str]:
        """Everything the checker may match against."""
        out = list(self.facts) + [self.template]
        if self.records is not None and len(self.records):
            out.append(self.records.head(200).to_csv(index=False))
        out += [q.get("text", "") + " " + q.get("number", "") + " " + q.get("date", "") for q in self.quotes]
        return out


def indian(n: float, decimals: int = 0) -> str:
    """Indian digit grouping: 16900000 -> '1,69,00,000'."""
    neg, n = n < 0, abs(float(n))
    whole, _, frac = f"{n:.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + (f".{frac}" if decimals else "")


def rupees(n: float) -> str:
    """₹ amount in words people use: ₹1.69 crore, ₹47.50 lakh, ₹9,500."""
    n = float(n)
    if abs(n) >= 1e7:
        return f"₹{n / 1e7:,.2f} crore"
    if abs(n) >= 1e5:
        return f"₹{n / 1e5:,.2f} lakh"
    return f"₹{indian(n)}"


def df_lines(df: pd.DataFrame, n: int = 8) -> list[str]:
    """First rows of a table as 'col: value' lines (numbers in Indian format)."""
    out = []
    for _, row in df.head(n).iterrows():
        bits = []
        for c, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v):
                v = indian(v, 0 if float(v).is_integer() else 2)
            bits.append(f"{str(c).replace('_', ' ')}: {v}")
        out.append(" · ".join(bits))
    return out

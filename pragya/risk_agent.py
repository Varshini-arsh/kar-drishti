"""Risk agent - Task 1, Fraud & Risk Detection.

Wraps the existing Kar Drishti engine (rule-based detectors + points-based risk score).
It runs code only (no model call): the tools return their own facts and a template answer.
Tools: risk ranking, holder profile, money trail, links/network, GST signals, and a
complete list for one red-flag pattern ("showing N of M" when the list is long).
"""
from __future__ import annotations

import re

import pandas as pd

from kar_drishti import risk
from kar_drishti.agents import KarDrishti, resolve_holders

from .base import AgentResult, rupees

LIST_LIMIT = 15

# question words -> red-flag pattern, for "who are the ..." list questions
PATTERN_WORDS = [
    (r"non.?filer", "non_filer"),
    (r"round.?figure|round cash|large cash withdrawal", "round_cash"),
    (r"pass.?through|mule", "pass_through"),
    (r"structur|just below|below (rs\.? ?|₹)?50,?000", "structuring"),
    (r"round.?trip", "round_trip"),
    (r"income mismatch", "income_mismatch"),
]

RISK_WORDS = re.compile(r"investigat|priorit|risk|flagged|flag\b|suspicious|why is|explain|tell me about|"
                        r"money (trail|from)|where did|trace|follow the money|shell|network|linked|connected|"
                        r"connections|mule|pass.?through|structur|round.?trip|non.?filer|round.?figure|"
                        r"\bgst\b|\bitc\b|input tax|e-?way|circular trading|fake invoice|review first|"
                        r"which cases|case(s)? (should|to)|income mismatch", re.I)


def wants_risk(question: str, has_holder: bool) -> bool:
    if RISK_WORDS.search(question):
        return True
    return has_holder and bool(re.search(r"\b(who is|about|profile|details)\b", question, re.I))


class RiskAgent:
    name = "Risk agent"

    def __init__(self, case, analysis, con):
        self.case, self.a = case, analysis
        self.engine = KarDrishti(case, analysis, con, llm=None)   # code only; the Lead agent writes the answer

    def run(self, question: str) -> AgentResult:
        checked = (f"{len(self.case.holders):,} people and firms, {len(self.case.transactions):,} transactions, "
                   f"{len(self.a.findings):,} red flags")
        pattern = next((p for rx, p in PATTERN_WORDS if re.search(rx, question, re.I)), None)
        holders = resolve_holders(self.case, question)
        threshold = re.search(r"(more|less|greater|fewer|higher|lower) than|\b(above|below|over|under|at least)\b.*\d", question, re.I)
        if pattern and not holders and threshold:   # a number threshold: the Data agent answers it exactly
            return AgentResult(self.name, "none", ok=False, checked=checked,
                               steps=["The question sets its own number threshold; passing it to the Data agent."])
        if pattern and not holders:
            return self.pattern_list(pattern, checked)
        resp = self.engine.ask(question)
        if resp.tool == "sql":          # nothing risk-specific in the question: let the Data agent answer
            return AgentResult(self.name, "none", ok=False, checked=checked,
                               steps=["No risk tool fits this question."])
        tool_names = {"risk_overview": "risk ranking", "holder_profile": "holder profile", "fund_trace": "money trail",
                      "network": "link finder", "gst": "GST signals for income-tax risk"}
        text = resp.text
        if resp.tool == "gst":
            text = "GST signals for income-tax risk: " + text
        steps = [s.replace("**", "") for s in resp.steps if not s.startswith("Planner")]
        facts = steps + [text]
        if resp.tool == "holder_profile" and resp.holders:
            r = self.a.risk.set_index("holder_id").loc[resp.holders[0]]
            facts.append(f"{r['name']}: total credits {rupees(r['total_credits'])}, cash deposits "
                         f"{rupees(r['cash_deposits'])}, declared income {rupees(r['declared_income'])}.")
        return AgentResult(self.name, tool_names.get(resp.tool, resp.tool), ok=True, facts=facts, template=text,
                           checked=checked, records=resp.df, fig=resp.fig, html=resp.html,
                           steps=steps,
                           extra={"holders": resp.holders})

    def pattern_list(self, pattern: str, checked: str) -> AgentResult:
        """Complete list of holders with one red flag, highest risk first."""
        label = risk.PATTERN_LABEL.get(pattern, pattern)
        ids = set(self.a.findings.loc[self.a.findings.pattern == pattern, "holder_id"])
        rows = self.a.risk[self.a.risk.holder_id.isin(ids)]
        m = len(rows)
        if m == 0:
            text = f"No one in the loaded data has the red flag '{label}'."
            return AgentResult(self.name, f"list: {label}", ok=True, facts=[text], template=text, checked=checked)
        shown = rows.head(LIST_LIMIT)
        f = self.a.findings[self.a.findings.pattern == pattern].set_index("holder_id")["summary"]
        lines = [f"{x.name} (risk {x.risk_score:.0f}): {f.get(x.holder_id, '')}" for x in shown.itertuples()]
        head = (f"{m} people or firms have the red flag '{label}'"
                + (f" (showing {len(shown)} of {m})." if m > len(shown) else "."))
        text = head + "\n" + "\n".join(f"- {ln}" for ln in lines[:6]) + (
            f"\n- … and {m - 6} more in the records below." if m > 6 else "")
        df = pd.DataFrame({"name": rows.name, "type": rows.holder_type.str.title(), "risk score": rows.risk_score,
                           "level": rows.risk_level, "finding": rows.holder_id.map(f)})
        return AgentResult(self.name, f"list: {label}", ok=True, facts=[head] + lines, template=text, checked=checked,
                           records=df, steps=[f"Listed every holder with '{label}': {m} found."])

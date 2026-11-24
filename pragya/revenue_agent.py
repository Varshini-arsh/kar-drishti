"""Revenue agent - Task 2, Revenue & Demand Tracking. Runs code only (pragya/revenue.py does all maths).

Tools: target/collected/gap summary, year-end projection and required monthly pace, which unit is
behind (region > charge > range), month by month, demand ageing, ranked recovery list with reasons,
one taxpayer's demand ("is X worth chasing?"), refunds.
"""
from __future__ import annotations

import re

import pandas as pd

from .base import AgentResult, rupees
from .revenue import LEVELS, Revenue


def _month_name(m: str) -> str:
    return pd.Timestamp(m + "-01").strftime("%b %Y")


class RevenueAgent:
    name = "Revenue agent"

    def __init__(self, rev: Revenue):
        self.rev = rev

    def known_names(self) -> list[str]:
        return self.rev.known_names()

    # ------------------------------------------------------------------ what the question is about
    def where(self, q: str) -> dict:
        """Region / charge / range named in the question (e.g. 'Range N1-2', 'Charge S1', 'Region North')."""
        low, out = q.lower(), {}
        for lvl in LEVELS:
            if lvl not in self.rev.targets:
                continue
            for v in sorted(self.rev.targets[lvl].astype(str).unique(), key=len, reverse=True):
                short = re.sub(r"\s*\(demo\)", "", v).lower()
                if short in low or v.lower() in low:
                    out[lvl] = v
                    break
        return out

    def taxpayer(self, q: str) -> str | None:
        d = self.rev.demands
        if d is None:
            return None
        low = q.lower().replace("’", "'")
        for n in sorted(d["taxpayer_name"].astype(str).unique(), key=len, reverse=True):
            if re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low):
                return n
        return None

    def checked(self) -> str:
        r = self.rev
        bits = [f"{len(r.targets)} range targets", f"{len(r.collections):,} collection rows"]
        if r.demands is not None:
            bits.append(f"{len(r.demands)} demands")
        return ", ".join(bits) + f" ({r.label}, FY {r.fy}, as on {r.as_on})"

    # ------------------------------------------------------------------ main entry
    def run(self, q: str) -> AgentResult:
        low = q.lower()
        tp = self.taxpayer(q)
        if tp:
            return self.taxpayer_demand(tp)
        if re.search(r"ageing|aging|how old|age of", low):
            return self.ageing()
        if re.search(r"chase|recover|recovery list|undisputed|collect first|pursue", low):
            return self.recovery()
        if re.search(r"refund", low):
            return self.refunds()
        lvl = next((x for x in ("range", "charge", "region") if x in low), None)
        if re.search(r"behind|lagging|furthest|worst|weakest|best|ahead|which (range|charge|region)", low) and lvl:
            return self.ranking(lvl, best=bool(re.search(r"best|ahead|leading", low)))
        if re.search(r"pace|needed|required|need to|to reach", low):
            return self.summary(q)
        if re.search(r"month|monthly|month.?wise|trend", low):
            return self.monthly()
        return self.summary(q)

    # ------------------------------------------------------------------ tools
    def summary(self, q: str) -> AgentResult:
        w = self.where(q)
        s = self.rev.summary(w)
        scope = " / ".join(w.values()) if w else "all ranges"
        facts = [f"Scope: {scope}. Financial year {s['fy']}, collections as on {s['as_on']}.",
                 f"Target {rupees(s['target'])}; collected {rupees(s['collected'])} ({s['achieved_pct']}% of target).",
                 f"Gap {rupees(s['gap'])} with {s['months_left']} months left to 31 March: "
                 f"{rupees(s['monthly_pace_needed'])} needed per month.",
                 f"Year-end projection {rupees(s['projected'])} (last year's monthly pattern scaled by this year's "
                 f"run-rate); plain run-rate projection {rupees(s['run_rate_projected'])}.",
                 f"Projected {'shortfall' if s['projected'] < s['target'] else 'surplus'} against target: "
                 f"{rupees(abs(s['projected'] - s['target']))}."]
        if s["refunds"] is not None:
            facts.append(f"Refunds issued this year: {rupees(s['refunds'])}.")
        text = "\n".join(facts[1:5])
        table = self.rev.rollup("range", w)[["region", "charge", "range", "target", "collected", "gap",
                                             "achieved_pct", "projected"]]
        must = [rupees(s["target"]), rupees(s["collected"]), rupees(s["gap"]), rupees(s["projected"]),
                rupees(s["monthly_pace_needed"])]
        return AgentResult(self.name, "revenue summary", True, facts, text, self.checked(), records=table,
                           fig=self.chart(w), extra={"summary": s, "where": w}, must=must,
                           steps=["Summed range-level targets and collections; projection from last year's pattern."])

    def ranking(self, level: str, best: bool = False) -> AgentResult:
        g = self.rev.rollup(level).sort_values("achieved_pct", ascending=not best).reset_index(drop=True)
        top = g.iloc[0]
        word = "furthest ahead" if best else "furthest behind"
        facts = [f"{top[level]} is {word}: {top['achieved_pct']}% of its target collected "
                 f"({rupees(top['collected'])} of {rupees(top['target'])}), gap {rupees(top['gap'])}, "
                 f"projected {rupees(top['projected'])}."]
        facts += [f"{r[level]}: {r['achieved_pct']}% of target, gap {rupees(r['gap'])}." for _, r in g.iloc[1:5].iterrows()]
        text = facts[0] + "\nNext: " + "; ".join(f.rstrip(".") for f in facts[1:3]) + "." if len(facts) > 1 else facts[0]
        cols = [c for c in LEVELS[:LEVELS.index(level) + 1]] + ["target", "collected", "gap", "achieved_pct", "projected"]
        return AgentResult(self.name, f"{level} ranking", True, facts, text, self.checked(), records=g[cols],
                           must=[str(top[level]), f"{top['achieved_pct']}%"],
                           steps=[f"Ranked every {level} by share of target collected."])

    def monthly(self) -> AgentResult:
        m = self.rev.monthly()
        done = m.dropna(subset=["collected"])
        facts = [f"{_month_name(r.month)}: {rupees(r.collected)} (last year {rupees(r.last_year)})" for r in done.itertuples()]
        best = done.loc[done.collected.idxmax()]
        text = (f"Collections for {len(done)} months of FY {self.rev.fy} are loaded. Highest month: "
                f"{_month_name(best.month)} with {rupees(best.collected)}.\n" + "\n".join(f"- {f}" for f in facts[-4:]))
        facts.append(f"Highest month: {_month_name(best.month)} with {rupees(best.collected)}.")
        facts.append(f"Collections for {len(done)} months of FY {self.rev.fy} are loaded.")
        return AgentResult(self.name, "monthly collections", True, facts, text, self.checked(),
                           must=[_month_name(best.month), rupees(best.collected)],
                           extra={"always": [f"Highest month: {_month_name(best.month)} with {rupees(best.collected)}."]},
                           records=m.fillna({"collected": 0}), fig=self.chart({}),
                           steps=["Summed collections by month, this year and last year."])

    def ageing(self) -> AgentResult:
        a = self.rev.ageing()
        total = int(a["outstanding"].sum())
        facts = [f"{r.age}: {r.demands} demands, {rupees(r.outstanding)} outstanding "
                 f"({rupees(r.disputed_or_stayed)} disputed or stayed)" for r in a.itertuples()]
        facts.insert(0, f"Total outstanding demand {rupees(total)} as on {self.rev.as_on}.")
        text = "\n".join(facts[:1] + [f"- {f}" for f in facts[1:]])
        return AgentResult(self.name, "demand ageing", True, facts, text, self.checked(), records=a,
                           must=[rupees(total)], extra={"always": [facts[0]]},
                           steps=["Grouped outstanding demands by how long ago they were raised."])

    def recovery(self, n: int = 10) -> AgentResult:
        r = self.rev.recovery_list()
        if r.empty:
            t = "No outstanding demands in the loaded demand register."
            return AgentResult(self.name, "recovery list", True, [t], t, self.checked())
        clear = r[(r.disputed == "No") & (r.stay == "No")]
        facts = [f"{len(r)} demands are outstanding; {len(clear)} are undisputed with no stay "
                 f"({rupees(clear.outstanding.sum())})."]
        facts += [f"{x.rank}. {x.taxpayer_name}: {x.reason}" for x in r.head(n).itertuples()]
        text = facts[0] + " Chase first (largest undisputed, no stay):\n" + "\n".join(f"- {f}" for f in facts[1:6])
        return AgentResult(self.name, "recovery list", True, facts, text, self.checked(), records=r,
                           must=list(r.head(3).taxpayer_name),
                           steps=["Ranked outstanding demands: undisputed and no stay first, then largest first."])

    def taxpayer_demand(self, name: str) -> AgentResult:
        r = self.rev.recovery_list()
        mine = r[r.taxpayer_name == name]
        if mine.empty:
            t = f"{name} has no outstanding demand in the loaded demand register."
            return AgentResult(self.name, "taxpayer demand", True, [t], t, self.checked())
        facts = [f"{name} has {len(mine)} outstanding demand(s) totalling {rupees(mine.outstanding.sum())}."]
        for x in mine.itertuples():
            facts.append(f"Demand {x.demand_id}: rank {x.rank} of {len(r)} on the recovery list - {x.reason}.")
        clear = mine[(mine.disputed == "No") & (mine.stay == "No")]
        if len(clear):
            facts.append(f"{rupees(clear.outstanding.sum())} of it is undisputed with no stay, so it can be pursued now; "
                         "the officer decides the action.")
        else:
            facts.append("All of it is disputed or stayed, so recovery depends on the appeal / stay; the officer decides.")
        text = "\n".join(facts)
        musts = [name, rupees(mine.outstanding.sum())] + ([rupees(clear.outstanding.sum()), "undisputed"] if len(clear) else [])
        if ((mine.stay == "Yes")).any():
            musts.append("stayed")
        elif (mine.disputed == "Yes").any():
            musts.append("disputed")
        return AgentResult(self.name, "taxpayer demand", True, facts, text, self.checked(), records=mine, must=musts,
                           extra={"always": ["The officer decides the action."]},
                           steps=[f"Found {name} in the demand register and its place on the ranked recovery list."])

    def refunds(self) -> AgentResult:
        f = self.rev.refunds
        if f is None or not len(f):
            t = "No refunds file is loaded."
            return AgentResult(self.name, "refunds", False, [t], t, self.checked())
        total, n = int(f["refund_amount"].sum()), len(f)
        by = f.groupby("charge" if "charge" in f else f.columns[0], as_index=False)["refund_amount"].sum()
        facts = [f"{n} refunds issued this year, totalling {rupees(total)}."] + \
                [f"{x[by.columns[0]]}: {rupees(x['refund_amount'])}" for _, x in by.iterrows()]
        return AgentResult(self.name, "refunds", True, facts, "\n".join(facts[:4]), self.checked(), records=by,
                           must=[rupees(total)],
                           steps=["Summed the refunds file."])

    # ------------------------------------------------------------------ the one chart
    def chart(self, where: dict):
        from .charts import revenue_chart
        return revenue_chart(self.rev, where)

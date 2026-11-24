"""Lead agent: reads the question, picks one or more specialists, merges their answers.

Flow for one question (at most 2 model calls in total):
  1. Route with rules (keywords + names found in the data). Only if no rule fits and a model is
     connected, one "plan" call asks the model which specialists to use.
  2. Each chosen specialist runs code and returns facts + a template answer.
     (The Data agent's SQL-writing call counts as the plan call.)
  3. One "write" call turns the facts into 3-6 plain lines.
  4. The Checker (plain code) verifies every number, name, date and document number.
     If it fails and a call is left, the model gets one retry; otherwise the template
     answer (built by code from the same facts) is shown.
  5. If no specialist can answer: "I can't answer this from the loaded data. Here is what I can do: ..."
Every step is kept in a trace for "How I got this" and written to the audit log.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from kar_drishti.agents import resolve_holders
from kar_drishti.llm import LLMError

from . import audit, checker
from .base import CANT_ANSWER, AgentResult
from .data_agent import DataAgent
from .llm import Budget
from .risk_agent import RiskAgent, wants_risk

LAW_WORDS = re.compile(r"circular|office order|notification|\bact\b|section \d|\bsec\.? \d|leaves?\b|holiday|"
                       r"\bcl\b|\bel\b|carr(y|ied|ies)[ -]forward|"
                       r"work from home|\bwfh\b|travel|tour\b|allowance|\bda\b|advance|\bclaim\b|reimburs|"
                       r"\brule(s)?\b|instruction|guideline|what does .* say|is .* allowed|entitle", re.I)
REVENUE_WORDS = re.compile(r"target|collect|revenue|demand|recover|arrear|refund|\bgap\b|project|run.?rate|"
                           r"behind|chase|chasing|outstanding|ageing|aging|budget|pace|31 march|year.?end|"
                           r"stay\b|disputed|\btro\b", re.I)
SPLIT = re.compile(r",?\s+and\s+(?=(?:what|which|who|how|is|are|does|do|can|show|list|where|why|when)\b)", re.I)

WRITE_SYSTEM = """You are {app}, an assistant for Indian income-tax officers.
Write the answer to the officer's question using ONLY the FACTS below.
Rules:
- 3 to 6 short, plain lines. No headings, no tables.
- Copy every number, name, date and document number exactly as written in the FACTS. Never calculate.
- Never add anything that is not in the FACTS. If the FACTS do not cover part of the question, say
  "Not stated in the loaded data" for that part.
- If FACTS come from more than one agent, cover each part of the question.
- Do not decide the case; the officer decides. Use words like "indicates" or "needs verification"."""

GENERAL_LABEL = ("General answer · not from your loaded data or documents, and not checked, so verify it with "
                 "the Act, the circular or a senior officer before relying on it.")
GENERAL_SYSTEM = """You are {app}, an assistant for Indian income-tax officers, answering from general knowledge.
Rules:
- Answer in 3 to 6 short, plain lines.
- If you are not sure, say so plainly. Never invent section numbers, circular numbers, dates or amounts.
- Income-tax law: the Income-tax Act, 2025 applies from 1 April 2026; earlier years remain under the 1961 Act.
- If the question needs live or current information (weather, news, scores, prices, today's events), say that you
  cannot know it because you run offline without internet.
- Do not give a decision on any taxpayer's case; the officer decides."""
# Office-rule questions: if the loaded circulars do not cover them, a general answer would only be a guess.
OFFICE_RULE = re.compile(r"leave|tour|travel|allowance|\bda\b|\bta\b|work from home|\bwfh\b|office order|"
                         r"circular|holiday|reimburs|travel claim|\bclaims? (for|within)", re.I)

PLAN_SYSTEM = """Choose which specialists can answer an Indian income-tax officer's question.
Specialists:
- risk: fraud and risk detection on bank transactions and returns (red flags, risk scores, money trails, links)
- revenue: revenue targets, collections, demand register, recovery, refunds
- law: circulars, office orders and Acts in the library (leave, travel, rules, sections)
- data: any other question about the loaded tables
- none: the question is not about the loaded data or documents
Reply in JSON only: {"agents": ["<name>", ...]}"""


@dataclass
class Answer:
    question: str
    text: str
    results: list = field(default_factory=list)      # AgentResult per specialist used
    trace: list = field(default_factory=list)        # "How I got this" lines
    check: checker.CheckResult | None = None
    source: str = "template"                         # "model" or "template" or "cannot answer"
    llm_calls: int = 0
    seconds: float = 0.0

    @property
    def agents(self) -> list[str]:
        return [r.agent for r in self.results]


class Lead:
    name = "Lead agent"

    def __init__(self, case, analysis, con, llm=None, revenue=None, law=None, app_name: str = "Pragya",
                 audit_path=None):
        self.case, self.a, self.con, self.llm = case, analysis, con, llm
        self.app_name, self.audit_path = app_name, audit_path
        self.risk = RiskAgent(case, analysis, con) if case is not None else None
        self.data = DataAgent(con, case)
        self.revenue, self.law = revenue, law          # added in later steps (None = not loaded)

    # ------------------------------------------------------------------ names the checker knows about
    def known_names(self) -> list[str]:
        names = list(self.case.holders.name) if self.case is not None else []
        if self.revenue is not None:
            names += self.revenue.known_names()
        return names

    # ------------------------------------------------------------------ routing
    def route(self, question: str, budget: Budget, trace: list) -> list[tuple[str, str]]:
        """List of (specialist, part of the question it should answer)."""
        parts = [p.strip(" ?,") + "?" for p in SPLIT.split(question) if p.strip(" ?,")] or [question]
        plan = []
        for part in parts:
            holders = resolve_holders(self.case, part) if self.case is not None else []
            picked = []
            if REVENUE_WORDS.search(part):
                picked.append("revenue")
            if LAW_WORDS.search(part):
                picked.append("law")
            if self.risk is not None and wants_risk(part, bool(holders)) and "revenue" not in picked:
                picked.append("risk")
            for p in picked:
                if p not in [x for x, _ in plan]:
                    plan.append((p, part if len(parts) > 1 else question))
        if plan:
            trace.append("Lead agent (rules) chose: " + ", ".join(dict.fromkeys(p for p, _ in plan)) + ".")
            return plan
        if self.llm is not None and budget.take("plan"):
            try:
                d = self.llm.chat_json([{"role": "system", "content": PLAN_SYSTEM},
                                        {"role": "user", "content": question}], purpose="plan")
                chosen = [a for a in d.get("agents", []) if a in ("risk", "revenue", "law", "data")]
                trace.append("Lead agent (local model) chose: " + (", ".join(chosen) or "none") + ".")
                if chosen:
                    return [(a, question) for a in dict.fromkeys(chosen)]
                return []
            except LLMError as e:
                trace.append(f"Lead agent: model unavailable ({e}); trying the Data agent.")
        else:
            trace.append("Lead agent (rules): no specialist keyword; trying the Data agent.")
        return [("data", question)]

    # ------------------------------------------------------------------ main entry
    def ask(self, question: str) -> Answer:
        t0 = time.time()
        question = re.sub(r"\s+", " ", question or "").strip()
        budget, trace = Budget(), []
        audit.log("question", self.audit_path, question=question)
        results = []
        plan = self.route(question, budget, trace)
        planned = {a for a, _ in plan}
        for agent, part in plan:
            res = self.run_agent(agent, part, budget)
            audit.log("agent", self.audit_path, agent=res.agent, tool=res.tool, ok=res.ok, checked=res.checked)
            trace.append(f"{res.agent} · tool: {res.tool} · checked: {res.checked or '-'} · "
                         + ("answered" if res.ok else "could not answer"))
            trace += [f"   {s}" for s in res.steps]
            if not res.ok and agent in ("risk", "revenue") and "data" not in planned:
                # the specialist had nothing; give the Data agent a chance on the same part
                res2 = self.run_agent("data", part, budget)
                trace.append(f"{res2.agent} · tool: {res2.tool} · checked: {res2.checked or '-'} · "
                             + ("answered" if res2.ok else "could not answer"))
                trace += [f"   {s}" for s in res2.steps]
                res = res2
            if res.ok:
                results.append(res)
        if not results:
            general = self.general_answer(question, budget, trace)
            if general:
                audit.log("answer", self.audit_path, question=question, source="general")
                return Answer(question, general, [], trace, None, "general", budget.used, time.time() - t0)
            text = self.cannot_answer()
            trace.append("No specialist could answer from the loaded data, so nothing was made up.")
            audit.log("answer", self.audit_path, question=question, source="cannot answer")
            return Answer(question, text, [], trace, None, "cannot answer", budget.used, time.time() - t0)
        ans = self.write(question, results, budget, trace)
        not_stated = all(r.agent == "Law agent" and r.extra.get("status") == "not stated" for r in results)
        if not_stated and not OFFICE_RULE.search(question):
            general = self.general_answer(question, budget, trace)
            if general:
                ans.text = ans.text + "\n\n" + general
                ans.source = "documents + general"
        ans.seconds = time.time() - t0
        audit.log("answer", self.audit_path, question=question, agents=ans.agents, source=ans.source,
                  checker=ans.check.summary() if ans.check else None, llm_calls=ans.llm_calls)
        return ans

    def run_agent(self, agent: str, part: str, budget: Budget) -> AgentResult:
        try:
            if agent == "risk" and self.risk is not None:
                return self.risk.run(part)
            if agent == "revenue":
                if self.revenue is None:
                    return AgentResult("Revenue agent", "none", ok=False, checked="no revenue files loaded",
                                       steps=["No targets, collections or demand register are loaded."])
                return self.revenue.run(part)
            if agent == "law":
                if self.law is None or not self.law.ready():
                    return AgentResult("Law agent", "none", ok=False, checked="no documents loaded",
                                       steps=["No circulars or Acts are loaded in the Library."])
                return self.law.run(part)
            known = ""
            if self.case is not None:
                ids = resolve_holders(self.case, part)
                if ids:
                    h = self.case.holders.set_index("holder_id")
                    known = "KNOWN IDS: " + "; ".join(f"{h.loc[i, 'name']} = {i} ({h.loc[i, 'holder_type']})"
                                                      for i in ids)
            return self.data.run(part, self.llm, budget, known)
        except Exception as e:      # a broken tool must never crash the answer
            return AgentResult(agent.title() + " agent", "error", ok=False, steps=[f"Tool error: {e}"])

    # ------------------------------------------------------------------ writing + checking
    def write(self, question: str, results: list, budget: Budget, trace: list) -> Answer:
        sources = [s for r in results for s in r.sources()]
        names = self.known_names()
        template = "\n\n".join((f"**{r.agent}** · " if len(results) > 1 else "") + r.template for r in results)
        law_only = all(r.agent == "Law agent" for r in results)
        text, source = template, "template"
        if self.llm is not None and not law_only and budget.take("write"):
            facts = "\n".join(f"[{r.agent} · {r.tool}]\n" + "\n".join(f"- {f}" for f in r.facts) for r in results)
            msgs = [{"role": "system", "content": WRITE_SYSTEM.format(app=self.app_name)},
                    {"role": "user", "content": f"Question: {question}\n\nFACTS:\n{facts[:7000]}"}]
            try:
                draft = self.llm.chat(msgs, purpose="write").strip()
                res = checker.check(draft, sources, names, question)
                trace.append("Checker on the model's answer: " + res.summary())
                if res.ok:
                    text, source = draft, "model"
                elif budget.take("rewrite"):
                    msgs += [{"role": "assistant", "content": draft},
                             {"role": "user", "content": "These items are not in the FACTS: " + "; ".join(res.problems)
                                                         + ". Rewrite using only the FACTS, copying numbers exactly."}]
                    draft = self.llm.chat(msgs, purpose="write").strip()
                    res = checker.check(draft, sources, names, question)
                    trace.append("Checker on the retry: " + res.summary())
                    if res.ok:
                        text, source = draft, "model"
                if source != "model":
                    trace.append("Showing the template answer built from the tool output instead.")
            except LLMError as e:
                trace.append(f"Local model unavailable ({e}); showing the template answer.")
        elif self.llm is None:
            trace.append("No local model connected; the answer is written by code from the tool output.")
        elif law_only:
            trace.append("Law answers show the exact quoted sentences, so no model rewriting is used.")
        final = checker.check(text, sources, names, question)
        trace.append("Checker on the answer shown: " + final.summary())
        if not final.ok:                       # should not happen for templates; never show unchecked text
            text = "\n".join(f"- {f}" for r in results for f in r.facts[:6])
            source = "template"
            final = checker.check(text, sources, names, question)
        trace.append(f"Model calls used: {budget.used} of {budget.limit}"
                     + (f" ({', '.join(budget.log)})" if budget.log else "") + ".")
        return Answer(question, text, results, trace, final, source, budget.used)

    # ------------------------------------------------------------------ general knowledge (labelled, not checked)
    def general_answer(self, question: str, budget: Budget, trace: list) -> str | None:
        """A clearly labelled answer from the local model's general knowledge, when no loaded data or document
        covers the question. Not possible without a local model (then the 'I can't answer' message is shown)."""
        if self.llm is None or not budget.take("general"):
            return None
        try:
            text = self.llm.chat([{"role": "system", "content": GENERAL_SYSTEM.format(app=self.app_name)},
                                  {"role": "user", "content": question}], purpose="general").strip()
        except LLMError as e:
            trace.append(f"Local model unavailable for a general answer ({e}).")
            return None
        if not text:
            return None
        trace.append("Nothing in the loaded data or documents covers this, so the local model answered from general "
                     "knowledge. This part is labelled and was NOT checked against your data.")
        trace.append(f"Model calls used: {budget.used} of {budget.limit} ({', '.join(budget.log)}).")
        return f"{GENERAL_LABEL}\n\n{text}"

    # ------------------------------------------------------------------ out of scope
    def cannot_answer(self) -> str:
        can = []
        if self.case is not None:
            can.append(f"Rank and explain risky cases among {len(self.case.holders):,} people and firms "
                       "(e.g. 'Which cases should I investigate first?')")
        if self.revenue is not None:
            can.append("Track revenue: target, collected, gap, projection and the recovery list "
                       "(e.g. 'How far are we from the revenue target?')")
        if self.law is not None and self.law.ready():
            can.append("Quote the loaded circulars and Acts (e.g. 'How many casual leave days are allowed?')")
        can.append("Answer questions on the loaded tables: " + self.data.counts())
        can.append("Answer general questions too, clearly labelled, once a local AI model (Ollama) is connected")
        return CANT_ANSWER + "\n" + "\n".join(f"- {c}" for c in can)

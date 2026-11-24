"""Pragya self-test: 60+ questions with expected answers across all agents.

    python -m pragya.selftest                  # offline: no AI model needed (templates + rules)
    python -m pragya.selftest --model qwen3.5:4b   # the same questions with the real local model writing answers
    python -m pragya.selftest --only law       # one category

Builds a fresh synthetic world (demo case, demo revenue files, the fictitious circulars), asks every
question, and prints a pass/fail table. Includes traps ("hotel allowance", "Study Leave" -> Not stated),
cross-task questions, each conflict question asked 3 times (answers must match), out-of-scope questions,
revenue totals that must add up, checker cases and fake-model cases.
Every answer must also pass the Checker. Exit code 1 if anything fails: nothing ships if a test fails.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data_generator.generate_data import generate  # noqa: E402
from data_generator.generate_revenue import generate as gen_revenue  # noqa: E402
from kar_drishti import risk  # noqa: E402
from kar_drishti.data import build_duckdb, load_case  # noqa: E402
from pragya import checker, uploads  # noqa: E402
from pragya.base import CANT_ANSWER, rupees  # noqa: E402
from pragya.law_agent import LawAgent  # noqa: E402
from pragya.lead import Lead  # noqa: E402
from pragya.library import NOT_STATED, OFFICER_DECIDES, Library  # noqa: E402
from pragya.llm import FakeLLM, PragyaLLM  # noqa: E402
from pragya.revenue import LEVELS, Revenue  # noqa: E402
from pragya.revenue_agent import RevenueAgent  # noqa: E402


@dataclass
class Q:
    cat: str
    question: str
    agents: list | None = None          # exact list of agents expected
    contains: list = field(default_factory=list)     # strings (or callables(world) -> str) that must appear
    excludes: list = field(default_factory=list)
    cannot: bool = False                # expects "I can't answer this from the loaded data..."
    tool: str | None = None
    repeat: int = 1                     # ask N times; answers must match
    check: object = None                # callable(world, answer) -> (ok, note)


@dataclass
class World:
    case: object
    analysis: object
    rev: Revenue
    lib: Library
    lead: Lead
    tmp: Path


def build_world(llm=None, tmp: Path | None = None) -> World:
    tmp = Path(tmp or tempfile.mkdtemp(prefix="pragya_selftest_"))
    generate(tmp / "case", 42, quiet=True)
    gen_revenue(tmp / "rev", 7, quiet=True)
    case = load_case(tmp / "case")
    a = risk.analyze(case, include_gst=False)
    con = build_duckdb(case, a.risk, include_gst=False)
    rev = Revenue.from_folder(tmp / "rev")
    for kind, attr in [("revenue targets", "targets"), ("collections", "collections"), ("demand register", "demands"),
                       ("refunds", "refunds")]:
        uploads.load_into_duckdb(con, uploads.TABLE_NAME[kind], getattr(rev, attr))
    uploads.load_into_duckdb(con, "up_vehicles", pd.DataFrame({"vehicle": ["Car", "Jeep", "Car"],
                                                               "km_run": [120, 300, 80], "driver": ["A", "B", "C"]}))
    lib = Library.from_folder()
    lead = Lead(case, a, con, llm, revenue=RevenueAgent(rev), law=LawAgent(lib), audit_path=tmp / "audit.jsonl")
    return World(case, a, rev, lib, lead, tmp)


# ---------------------------------------------------------------- expected values computed from the data (never typed in)
def top_name(w):
    return w.analysis.risk.iloc[0]["name"]


def n_with(pattern):
    return lambda w: f"{int(w.analysis.findings.pattern.eq(pattern).sum())} people or firms"


def decoy(i):
    return lambda w: w.case.holder_name(w.case.ground_truth["decoys"][i]["holder_id"])


def border(i):
    return lambda w: w.case.holder_name(w.case.ground_truth["borderline"][i]["holder_id"])


def summ(key, where=None):
    return lambda w: rupees(w.rev.summary(where)[key])


def furthest(level, best=False):
    def f(w):
        g = w.rev.rollup(level).sort_values("achieved_pct", ascending=not best)
        return g.iloc[0][level]
    return f


def first_to_chase(w):
    return w.rev.recovery_list().iloc[0]["taxpayer_name"]


def records(n=None, at_least=None):
    def chk(w, ans):
        r = ans.results[0].records if ans.results else None
        k = 0 if r is None else len(r)
        ok = (n is None or k == n) and (at_least is None or k >= at_least)
        return ok, f"{k} records"
    return chk


def is_conflict(w, ans):
    ok = OFFICER_DECIDES in ans.text and any(q.get("later") for q in ans.results[0].quotes)
    return ok, "conflict shown, later marked" if ok else "conflict not shown"


QUESTIONS = [
    # ------------------------------------------------ Risk agent (Task 1)
    Q("risk", "Which cases should I investigate first?", ["Risk agent"], [top_name], tool="risk ranking"),
    Q("risk", "Why is Kuber Traders flagged?", ["Risk agent"], ["Kuber Traders", "Round-tripping"], tool="holder profile"),
    Q("risk", "Where did the money from Kuber Traders go?", ["Risk agent"], ["Kuber Traders"], tool="money trail"),
    Q("risk", "Show the shell firm network", ["Risk agent"], ["connected entities"], tool="link finder"),
    Q("risk", "Who are the non-filers with high-value transactions?", ["Risk agent"], [n_with("non_filer")]),
    Q("risk", "Who made large round-figure cash withdrawals?", ["Risk agent"], [n_with("round_cash")]),
    Q("risk", "Which accounts look like pass-through accounts?", ["Risk agent"], [n_with("pass_through")]),
    Q("risk", "Who is doing cash structuring?", ["Risk agent"], [n_with("structuring"), border(0)]),
    Q("risk", "Tell me about Siddhi Commodities", ["Risk agent"], ["Siddhi Commodities"], tool="holder profile"),
    Q("risk", "Which firms are passing fake GST input tax credit?", ["Risk agent"], ["add-on"]),
    Q("risk", lambda w: f"Why is {decoy(0)(w)} flagged?", ["Risk agent"], ["no suspicious pattern"]),
    Q("risk", lambda w: f"Why is {decoy(3)(w)} flagged?", ["Risk agent"], ["no suspicious pattern"]),
    Q("risk", lambda w: f"Why is {border(1)(w)} flagged?", ["Risk agent"], ["Income mismatch"]),
    # ------------------------------------------------ Data agent (any table)
    Q("data", "Show total cash deposits by month", ["Data agent"], check=records(12)),
    Q("data", "Who received more than 10 times their declared income?", ["Data agent"], check=records(at_least=1)),
    Q("data", "Top 5 people by total cash deposits", ["Data agent"], check=records(5)),
    Q("data", "Top 10 holders by money sent through NEFT", ["Data agent"], check=records(10)),
    Q("data", "Non-filers who received more than 20 lakh", ["Data agent"], check=records(at_least=1)),
    Q("data", "List the holders with the most transactions", ["Data agent"], check=records(at_least=1)),
    Q("risk", "List the high risk holders with their flags", ["Risk agent"], tool="risk ranking"),
    Q("data", "How many people and firms are in the holders table?", ["Data agent"],
      [lambda w: f"{len(w.case.holders)}"]),
    Q("data", "What is the total km run by vehicle?", ["Data agent"], ["Jeep", "300"]),
    Q("data", "How many rows are in up vehicles?", ["Data agent"], ["3"]),
    Q("data", "What is the average km run in up vehicles?", ["Data agent"], ["166.67"]),
    # ------------------------------------------------ Revenue agent (Task 2)
    Q("revenue", "How far are we from the revenue target?", ["Revenue agent"], [summ("gap"), summ("target")]),
    Q("revenue", "What is the projected collection by 31 March?", ["Revenue agent"], [summ("projected")]),
    Q("revenue", "How much has been collected this year?", ["Revenue agent"], [summ("collected")]),
    Q("revenue", "What monthly pace is needed to reach the target?", ["Revenue agent"], [summ("monthly_pace_needed")]),
    Q("revenue", "Which range is furthest behind its target?", ["Revenue agent"], ["Range S2-3"]),
    Q("revenue", "Which charge is furthest behind its target?", ["Revenue agent"], [furthest("charge")]),
    Q("revenue", "Which region is furthest ahead of its target?", ["Revenue agent"], [furthest("region", True)]),
    Q("revenue", "Show demand ageing", ["Revenue agent"], ["Total outstanding demand"], tool="demand ageing"),
    Q("revenue", "Which undisputed demands should we chase first?", ["Revenue agent"], [first_to_chase],
      tool="recovery list"),
    Q("revenue", "Is Ganesh Commodities' unpaid demand worth chasing?", ["Revenue agent"],
      ["₹1.69 crore", "undisputed", "officer decides"]),
    Q("revenue", "Is Kuber Traders' demand worth chasing?", ["Revenue agent"], ["stayed"]),
    Q("revenue", "What is the gap for Charge N1?", ["Revenue agent"],
      [lambda w: rupees(w.rev.summary({"charge": "Charge N1 (demo)"})["gap"])]),
    Q("revenue", "Show monthly collections", ["Revenue agent"], ["Highest month"], tool="monthly collections"),
    Q("revenue", "How much was refunded this year?", ["Revenue agent"],
      [lambda w: rupees(int(w.rev.refunds.refund_amount.sum()))]),
    # ------------------------------------------------ Law agent (Task 3) - conflicts asked 3 times
    Q("law", "How many casual leave days are allowed?", ["Law agent"], ["12 days", "15 days"], repeat=3, check=is_conflict),
    Q("law", "How many casual leave days can be carried forward?", ["Law agent"], ["cannot be carried forward",
                                                                                  "Up to 5 days"], repeat=3, check=is_conflict),
    Q("law", "What is the maximum earned leave that can be accumulated?", ["Law agent"], ["30 days", "60 days"],
      repeat=3, check=is_conflict),
    Q("law", "Is work from home allowed?", ["Law agent"], ["not permitted", "2 days in a month"], repeat=3,
      check=is_conflict),
    Q("law", "Can CL be carried forward?", ["Law agent"], ["Up to 5 days"], repeat=3, check=is_conflict),
    Q("law", "Is WFH permitted?", ["Law agent"], ["prior approval"], repeat=3, check=is_conflict),
    Q("law", "How many days in advance must leave be applied for?", ["Law agent"], ["3 working days", "not a conflict"],
      excludes=[OFFICER_DECIDES]),
    Q("law", "How much advance can be taken for a tour?", ["Law agent"], ["80%", "PRG/ADM/4/2021"]),
    Q("law", "Which class of travel is allowed?", ["Law agent"], ["AC 2-tier"]),
    Q("law", "What is the daily allowance on tour?", ["Law agent"], ["Rs 900 per day", "10 August 2021"]),
    Q("law", "Within how many days must a travel claim be filed?", ["Law agent"], ["within 15 days"]),
    Q("law", "What is the deadline to submit a travel claim?", ["Law agent"], ["within 15 days"]),
    Q("law", "What is the hotel allowance?", ["Law agent"], [NOT_STATED], repeat=3),            # trap
    Q("law", "How many days of Study Leave are allowed?", ["Law agent"], [NOT_STATED], repeat=3),  # trap
    Q("law", "Is hotel allowance paid on tour?", ["Law agent"], [NOT_STATED]),
    Q("law", "What is the maternity leave rule?", ["Law agent"], [NOT_STATED]),
    Q("law", "Which Act applies to tax year 2026-27?", ["Law agent"], ["Income-tax Act, 2025 applies"]),
    Q("law", "Which Act applies to 2025-26?", ["Law agent"], ["Income-tax Act, 1961 applies"]),
    Q("law", "What is the 2025 Act section for section 80C of the 1961 Act?", ["Law agent"], ["never guesses"]),
    # ------------------------------------------------ cross-task (two agents, merged)
    Q("cross", "Is Ganesh Commodities' unpaid demand worth chasing, and what does the recovery circular say?",
      ["Revenue agent", "Law agent"], ["₹1.69 crore", NOT_STATED]),
    Q("cross", "Which cases should I investigate first, and how far are we from the revenue target?",
      ["Risk agent", "Revenue agent"], [top_name, summ("gap")]),
    Q("cross", "What is the daily allowance on tour, and how much has been collected this year?",
      ["Law agent", "Revenue agent"], ["Rs 900 per day", summ("collected")]),
    Q("cross", "Why is Kuber Traders flagged, and which undisputed demands should we chase first?",
      ["Risk agent", "Revenue agent"], ["Kuber Traders", first_to_chase]),
    # ------------------------------------------------ out of scope: never a made-up answer
    Q("scope", "What is the weather in Delhi tomorrow?", cannot=True),
    Q("scope", "Who won the cricket match yesterday?", cannot=True),
    Q("scope", "Write me a poem about the monsoon", cannot=True),
    Q("scope", "What is the GDP of Japan?", cannot=True),
    Q("scope", "Book a train ticket to Mumbai", cannot=True),
]


# ---------------------------------------------------------------- non-question checks
def extra_checks(w: World) -> list[tuple[str, str, bool, str]]:
    out = []
    probs = w.rev.consistency()
    out.append(("totals", "Revenue totals add up: range > charge > region > all, months > year", not probs,
                "; ".join(probs) or "all levels add up"))
    total = w.rev.rollup(None).iloc[0]
    for level in LEVELS:
        g = w.rev.rollup(level)
        ok = all(int(g[c].sum()) == int(total[c]) for c in ["target", "collected", "gap", "projected"])
        out.append(("totals", f"Revenue: {level} totals = overall totals", ok, f"{len(g)} {level}s"))
    rng, ch = w.rev.rollup("range"), w.rev.rollup("charge")
    ok = all(rng[rng.charge == r.charge].collected.sum() == r.collected for r in ch.itertuples())
    out.append(("totals", "Revenue: each charge = sum of its ranges", ok, f"{len(ch)} charges checked"))
    for ans_text, src, expect, label in [
        ("Collected Rs 1.69 crore", "16900000", True, "Checker: Rs 1.69 crore = 16900000"),
        ("Collected 1,69,00,000", "Rs 1.69 crore", True, "Checker: 1,69,00,000 = Rs 1.69 crore"),
        ("Collected ₹1.75 crore", "16912345", False, "Checker rejects a wrong rounded figure"),
        ("Siddhi Commodities paid", "Kuber Traders paid", False, "Checker rejects a name not in the tool output"),
        ("See Circular No. ADM-99/2020", "Circular No. ADM-12/2019", False, "Checker rejects an unknown document number"),
    ]:
        ok = checker.check(ans_text, src, ["Kuber Traders", "Siddhi Commodities"]).ok is expect
        out.append(("checker", label, ok, "ok" if ok else "wrong"))
    # fake model: a made-up number is retried once, then replaced by the template
    fake_bad = FakeLLM({"write": "Start with Kuber Traders, who moved ₹999 crore."})
    w.lead.llm = fake_bad
    ans = w.lead.ask("Which cases should I investigate first?")
    out.append(("model", "Made-up number: retried once, then template answer", fake_bad.calls == ["write", "write"]
                and ans.source == "template" and "999" not in ans.text, f"calls {fake_bad.calls}"))
    fake_good = FakeLLM()
    w.lead.llm = fake_good
    ans = w.lead.ask("Which cases should I investigate first?")
    out.append(("model", "Checked model answer is used; at most 2 model calls", ans.source == "model"
                and ans.llm_calls <= 2, f"source {ans.source}, calls {ans.llm_calls}"))
    w.lead.llm = None
    return out


# ---------------------------------------------------------------- runner
def text_of(x, w):
    return x(w) if callable(x) else x


def run_question(w: World, q: Q) -> tuple[bool, str, float, str]:
    question = text_of(q.question, w)
    t0 = time.time()
    answers = [w.lead.ask(question) for _ in range(q.repeat)]
    secs = (time.time() - t0) / q.repeat
    ans = answers[0]
    notes = []
    ok = True
    if q.repeat > 1 and len({a.text for a in answers}) != 1:
        ok, notes = False, notes + [f"answers differ across {q.repeat} runs"]
    if q.cannot:
        if not ans.text.startswith(CANT_ANSWER):
            ok, notes = False, notes + ["expected 'I can't answer this...'"]
    else:
        if ans.text.startswith(CANT_ANSWER):
            ok, notes = False, notes + ["refused"]
        if ans.check is None or not ans.check.ok:
            ok, notes = False, notes + ["checker: " + (ans.check.summary() if ans.check else "not run")]
    if q.agents is not None and ans.agents != q.agents:
        ok, notes = False, notes + [f"agents {ans.agents}"]
    if q.tool and (not ans.results or ans.results[0].tool != q.tool):
        ok, notes = False, notes + [f"tool {ans.results[0].tool if ans.results else None}"]
    for c in q.contains:
        c = text_of(c, w)
        if c not in ans.text:
            ok, notes = False, notes + [f"missing '{c}'"]
    for c in q.excludes:
        if text_of(c, w) in ans.text:
            ok, notes = False, notes + [f"should not say '{c}'"]
    if q.check is not None:
        good, note = q.check(w, ans)
        ok = ok and good
        notes.append(note)
    if ok and not notes:
        notes.append(f"{'+'.join(a.split()[0] for a in ans.agents) or 'none'} · " + (ans.check.summary()[:60] if ans.check else "refused"))
    return ok, question, secs, "; ".join(notes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pragya self-test (60+ questions)")
    ap.add_argument("--model", help="run with a real local model, e.g. qwen3.5:4b (Ollama must be running)")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--only", help="one category: risk, data, revenue, law, cross, scope, totals, checker, model")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    llm = PragyaLLM("ollama", a.url, a.model) if a.model else None
    if llm is not None:
        installed = llm.list_models()
        if not installed:
            print(f"No model server found at {a.url}. Start Ollama first (ollama serve).")
            return 2
        if a.model not in installed and not any(m.startswith(a.model) for m in installed):
            print(f"Model '{a.model}' is not installed. Installed: {', '.join(installed)}.\n"
                  f"Install it yourself with:  ollama pull {a.model}   or run with one of the installed models.")
            return 2
    print(f"Pragya self-test · {'model ' + a.model if a.model else 'offline (no AI model)'} · building demo world "
          "(about 30-60 s)...", flush=True)
    t0 = time.time()
    w = build_world(llm)
    print(f"Demo world ready in {time.time() - t0:.0f} s. Asking the questions"
          + (" (the model writes most answers, about 10-30 s each)..." if llm else "..."), flush=True)
    todo = [q for q in QUESTIONS if not a.only or q.cat == a.only]
    rows = []
    for i, q in enumerate(todo, 1):
        ok, question, secs, note = run_question(w, q)
        rows.append((q.cat, question + (f"  (x{q.repeat})" if q.repeat > 1 else ""), ok, secs, note))
        print(f"  [{i}/{len(todo)}] {'PASS' if ok else 'FAIL'} {secs:5.1f}s  {question[:70]}", flush=True)
    if not a.only or a.only in ("totals", "checker", "model"):
        for cat, label, ok, note in extra_checks(w):
            if not a.only or a.only == cat:
                rows.append((cat, label, ok, 0.0, note))
    width = 74
    print(f"\n{'#':>3}  {'area':8} {'result':6} {'sec':>5}  {'question / check':{width}}  note")
    print("-" * (3 + 2 + 8 + 1 + 6 + 1 + 5 + 2 + width + 8))
    for i, (cat, q, ok, secs, note) in enumerate(rows, 1):
        if a.quiet and ok:
            continue
        qq = q if len(q) <= width else q[:width - 1] + "…"
        print(f"{i:>3}  {cat:8} {'PASS' if ok else 'FAIL':6} {secs:5.1f}  {qq:{width}}  {note[:70]}")
    n_ok = sum(r[2] for r in rows)
    print("-" * 40)
    by = pd.DataFrame(rows, columns=["area", "q", "ok", "sec", "note"]).groupby("area")["ok"].agg(["sum", "count"])
    print("  ".join(f"{k}: {int(v['sum'])}/{int(v['count'])}" for k, v in by.iterrows()))
    timed = [r[3] for r in rows if r[3] > 0]
    if timed:
        print(f"Average answer time: {sum(timed) / len(timed):.1f} s · slowest {max(timed):.1f} s")
    print(f"TOTAL: {n_ok}/{len(rows)} passed" + ("" if n_ok == len(rows) else "  ->  DO NOT SHIP"))
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

"""
The Kar Drishti agents.

  PlannerAgent  - reads the officer's question and decides which tool to use
  SQLAgent      - turns a plain-English question into a safe, read-only SQL query,
                  runs it on the local DuckDB database and fixes its own errors
  Explainer     - writes plain-English explanations using ONLY the computed facts
  KarDrishti    - the orchestrator that ties the tools together

Golden rule: the language model never invents numbers. Numbers always come from
code that ran on the data; the model only chooses tools, writes queries and
explains results. Every answer carries its evidence (rows / transactions).
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

import pandas as pd

from . import charts, gst, link_finder, money_trail, risk
from .data import Case, schema_text
from .llm import LLMError, LocalLLM
from .money_trail import inr

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|attach|detach|pragma|install|load|"
                       r"export|import|call|set|reset|checkpoint|vacuum|grant|revoke|truncate)\b", re.I)


# ====================================================================== SQL safety
def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    sql = m.group(1) if m else text
    m2 = re.search(r"\b(with|select)\b.*", sql, re.S | re.I)
    sql = m2.group(0) if m2 else sql
    return sql.strip().rstrip(";").strip()


def validate_sql(sql: str) -> str | None:
    """Return an error message if the query is not a single read-only SELECT."""
    s = re.sub(r"'(?:[^']|'')*'", "''", sql)          # ignore string literals
    if not re.match(r"^\s*(with|select)\b", s, re.I):
        return "Only SELECT queries are allowed."
    if ";" in s:
        return "Only one statement is allowed."
    if FORBIDDEN.search(s):
        return f"Forbidden keyword: {FORBIDDEN.search(s).group(0)}"
    if re.search(r"\b(read_csv|read_parquet|read_json|glob|read_text|read_blob)\s*\(", s, re.I):
        return "File access functions are not allowed."
    return None


def run_sql(con, sql: str, limit: int = 1000) -> pd.DataFrame:
    err = validate_sql(sql)
    if err:
        raise ValueError(err)
    return con.execute(f"SELECT * FROM ({sql}) AS kd_q LIMIT {int(limit)}").df()


# ====================================================================== SQL agent
SQL_SYSTEM = """You are the SQL analyst inside Kar Drishti, a tool used by Indian income-tax officers.
Write ONE DuckDB SQL SELECT query that answers the officer's question using ONLY these tables:

{schema}
Rules:
- Output only the query inside a ```sql code block. No explanation.
- Read-only: a single SELECT (CTEs with WITH are fine). Never modify data.
- Amounts are in rupees. 1 lakh = 100000, 1 crore = 10000000.
- Money in / received / credit = direction = 'CR'. Money out / sent / paid / transferred = direction = 'DR'.
- Cash deposit = mode = 'CASH' AND direction = 'CR'. Cash withdrawal = mode IN ('CASH','ATM') AND direction = 'DR'.
- GST paid in cash, ITC and output tax are in gst_returns (tax_paid_cash, itc_claimed, output_tax), not in transactions.
- If KNOWN IDS are given, use exactly those holder_id values. Never invent IDs.
- Do NOT add filters the officer did not ask for (no extra holder_type, filed or date filters).
- To show names, JOIN holders. Match other names with ILIKE '%name%'.
- Round money with ROUND(x); give columns readable aliases.
- Month grouping: strftime(txn_date, '%Y-%m'). NULLIF takes exactly 2 arguments.
- Add ORDER BY for rankings and LIMIT 50 unless the officer asks for everything.
"""


class SQLAgent:
    """Text-to-SQL agent with few-shot retrieval, entity grounding and self-correction."""

    def __init__(self, llm: LocalLLM, con, case: Case | None = None):
        self.llm, self.con, self.case = llm, con, case
        self.schema = schema_text(con)

    def known_ids(self, question: str) -> str:
        if self.case is None:
            return ""
        ids = resolve_holders(self.case, question)
        if not ids:
            return ""
        h = self.case.holders.set_index("holder_id")
        return "KNOWN IDS: " + "; ".join(f"{h.loc[i, 'name']} = {i} ({h.loc[i, 'holder_type']})" for i in ids)

    def messages(self, question: str) -> list[dict]:
        from .sql_examples import pick_examples
        msgs = [{"role": "system", "content": SQL_SYSTEM.format(schema=self.schema)}]
        for q, s in reversed(pick_examples(question, k=4)):      # most similar example last (closest to question)
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```sql\n{s}\n```"}]
        ids = self.known_ids(question)
        msgs.append({"role": "user", "content": question + (f"\n{ids}" if ids else "")})
        return msgs

    def answer(self, question: str, max_attempts: int = 3) -> dict:
        msgs = self.messages(question)
        attempts, empty_retry_used = [], False
        for _ in range(max_attempts + 1):
            reply = self.llm.chat(msgs)
            sql = extract_sql(reply)
            try:
                df = run_sql(self.con.cursor(), sql)
            except Exception as e:      # let the model fix its own mistake
                err = str(e).split("\n")[0][:400]
                attempts.append({"sql": sql, "error": err})
                if len(attempts) >= max_attempts:
                    break
                msgs += [{"role": "assistant", "content": f"```sql\n{sql}\n```"},
                         {"role": "user", "content": f"That query failed with this error:\n{err}\n"
                                                     "Fix it and return only the corrected query."}]
                continue
            attempts.append({"sql": sql, "error": None, "rows": len(df)})
            if df.empty and not empty_retry_used:   # reflect once on an empty result
                empty_retry_used = True
                msgs += [{"role": "assistant", "content": f"```sql\n{sql}\n```"},
                         {"role": "user", "content": "That query ran but returned 0 rows. Check the direction "
                                                     "(CR/DR), the IDs and any extra filters. If the query is right, "
                                                     "return it unchanged; otherwise return a corrected query."}]
                first_ok = {"sql": sql, "df": df}
                continue
            return {"sql": sql, "df": df, "attempts": attempts}
        ok = [a for a in attempts if a["error"] is None]
        if ok and empty_retry_used:
            return {"sql": first_ok["sql"], "df": first_ok["df"], "attempts": attempts}
        return {"sql": attempts[-1]["sql"], "df": None, "attempts": attempts, "error": attempts[-1]["error"]}


# ====================================================================== explainer
EXPLAIN_SYSTEM = """You explain data findings to an Indian income-tax officer.
Use ONLY the facts and numbers given to you. Never invent names, numbers, dates or laws.
Write 3-6 short sentences in plain English. Use Indian units (lakh, crore).
Say what is unusual and what the officer could verify next. Do not decide guilt;
use words like 'indicates', 'needs verification'."""


def llm_explain(llm: LocalLLM | None, title: str, facts: str, fallback: str) -> tuple[str, bool]:
    if llm is None:
        return fallback, False
    try:
        txt = llm.chat([{"role": "system", "content": EXPLAIN_SYSTEM},
                        {"role": "user", "content": f"{title}\n\nFACTS:\n{facts[:6000]}"}])
        return txt.strip(), True
    except LLMError:
        return fallback, False


def df_facts(df: pd.DataFrame, n: int = 25) -> str:
    if df is None or df.empty:
        return "The query returned no rows."
    return f"{len(df)} row(s). First rows:\n{df.head(n).to_csv(index=False)}"


# ====================================================================== entity resolution
ID_RE = re.compile(r"\b([PF]\d{4}|AC\d{5})\b", re.I)
PAN_RE = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")


def resolve_holders(case: Case, text: str) -> list[str]:
    found = []
    acc_owner = dict(zip(case.accounts.account_id, case.accounts.holder_id))
    for m in ID_RE.findall(text):
        m = m.upper()
        hid = acc_owner.get(m, m)
        if hid in set(case.holders.holder_id):
            found.append(hid)
    for m in PAN_RE.findall(text.upper()):
        found += case.holders.loc[case.holders.pan == m, "holder_id"].tolist()
    low = text.lower()
    names = case.holders[["holder_id", "name"]]
    hits = [(hid, nm.lower()) for hid, nm in zip(names.holder_id, names.name)
            if re.search(r"(?<![a-z0-9])" + re.escape(nm.lower()) + r"(?![a-z0-9])", low)]
    # if one matched name is part of a longer matched name ("Kuber Traders" in "Kuber Traders 2"), keep the longer
    found += [h for h, n in hits if not any(n != m and n in m for _, m in hits)]
    if not found:   # fuzzy match 2-3 word windows against names
        words = re.findall(r"[a-z’']+", low)
        grams = [" ".join(words[i:i + k]) for k in (3, 2) for i in range(len(words) - k + 1)]
        lowmap = {n.lower(): h for h, n in zip(names.holder_id, names.name)}
        for g in grams:
            hit = difflib.get_close_matches(g, lowmap.keys(), n=1, cutoff=0.85)
            if hit:
                found.append(lowmap[hit[0]])
    out = []
    for f in found:
        if f not in out:
            out.append(f)
    return out


# ====================================================================== planner
TOOLS = {
    "risk_overview": "rank / list the riskiest holders, who to investigate first, summary of the case",
    "holder_profile": "explain one specific person or firm: why risky, profile, findings",
    "fund_trace": "follow the money from one specific person/firm/account: where did it go",
    "network": "connections of a specific holder, who is linked to whom, shell firm groups",
    "gst": "GST questions: input tax credit (ITC), fake invoices, circular trading, e-way bills, GST returns",
    "sql": "any other factual question about the data: totals, lists, filters, rankings, monthly trends",
}

PLANNER_SYSTEM = """You are the planner of Kar Drishti, an AI analyst for Indian tax officers.
Choose the single best tool for the officer's question.
Tools:
{tools}
Reply in JSON only: {{"tool": "<tool name>", "reason": "<one short sentence>"}}"""


def keyword_route(q: str, has_holder: bool) -> str:
    l = q.lower()
    if re.search(r"\bgst\b|\bitc\b|input tax|e-?way|circular trading|fake invoice", l) and \
            not re.search(r"\b(sum|total|month|list|show all|how many|count)\b", l):
        return "gst"
    if has_holder and re.search(r"where .*(go|went|sent|flow)|follow|trace|trail|destination", l):
        return "fund_trace"
    if re.search(r"link|connect|related|network|associate|cluster|shell", l) and (has_holder or "cluster" in l or "shell" in l):
        return "network"
    if has_holder and re.search(r"why|explain|profile|risk|suspicious|tell me about|details|who is", l):
        return "holder_profile"
    if re.search(r"(top|most|highest|riskiest|high.risk|prioriti|investigate first|summary|overview)", l) and \
            re.search(r"risk|suspicious|investigate|case|priorit", l):
        return "risk_overview"
    if has_holder:
        return "holder_profile"
    return "sql"


@dataclass
class AgentResponse:
    question: str
    tool: str
    text: str
    holders: list = field(default_factory=list)
    df: pd.DataFrame | None = None
    fig: object = None
    html: str | None = None
    sql: str | None = None
    steps: list = field(default_factory=list)
    used_llm: bool = False


class KarDrishti:
    """Orchestrator: planner -> tool -> explanation, with a visible step trace."""

    def __init__(self, case: Case, analysis: risk.Analysis, con, llm: LocalLLM | None):
        self.case, self.a, self.con, self.llm = case, analysis, con, llm
        self.sql_agent = SQLAgent(llm, con, case) if llm else None
        self.risk_idx = analysis.risk.set_index("holder_id")

    # ------------------------------------------------------------------ planning
    def plan(self, q: str, holders: list[str], steps: list) -> str:
        if self.llm:
            try:
                tools = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())
                extra = (f"\nHolders mentioned in the question: "
                         f"{', '.join(f'{h} ({self.case.holder_name(h)})' for h in holders)}" if holders
                         else "\nNo specific holder is mentioned.")
                d = self.llm.chat_json([{"role": "system", "content": PLANNER_SYSTEM.format(tools=tools)},
                                        {"role": "user", "content": q + extra}])
                tool = d.get("tool")
                if tool in TOOLS:
                    if tool in ("holder_profile", "fund_trace") and not holders:
                        tool = "sql"
                    steps.append(f"Planner (local LLM) chose **{tool}**: {d.get('reason', '')}")
                    return tool
            except LLMError as e:
                steps.append(f"Planner LLM unavailable ({e}); using keyword routing.")
        tool = keyword_route(q, bool(holders))
        steps.append(f"Planner (rules) chose **{tool}**.")
        return tool

    # ------------------------------------------------------------------ main entry
    def ask(self, q: str) -> AgentResponse:
        steps = []
        holders = resolve_holders(self.case, q)
        if holders:
            steps.append("Identified: " + ", ".join(f"{self.case.holder_name(h)} ({h})" for h in holders))
        tool = self.plan(q, holders, steps)
        fn = {"gst": self.tool_gst, "risk_overview": self.tool_risk_overview, "holder_profile": self.tool_holder_profile,
              "fund_trace": self.tool_fund_trace, "network": self.tool_network, "sql": self.tool_sql}[tool]
        resp = fn(q, holders, steps)
        resp.steps = steps
        return resp

    # ------------------------------------------------------------------ tools
    def tool_risk_overview(self, q, holders, steps):
        r = self.a.risk
        top = r[r.risk_level != "LOW"].head(15)
        n_review = int((r.risk_level != "LOW").sum())
        steps.append(f"Ranked {len(r)} holders by risk score; {n_review} need review (showing top {len(top)}).")
        df = top.assign(flags=top["flags"].map(lambda f: ", ".join(risk.PATTERN_LABEL.get(x, x) for x in f)))[
            ["holder_id", "name", "holder_type", "risk_score", "risk_level", "flags", "total_credits",
             "declared_income"]]
        facts = "\n".join(f"{x.name} ({x.holder_id}), score {x.risk_score:.0f}: " + " | ".join(x.reasons)
                          for x in top.head(6).itertuples())
        fallback = (f"{(r.risk_level == 'HIGH').sum()} high-risk and {(r.risk_level == 'MEDIUM').sum()} "
                    f"medium-risk holders found. Start with: " +
                    "; ".join(f"{x.name} ({x.risk_score:.0f})" for x in top.head(5).itertuples()) + ".")
        text, used = llm_explain(self.llm, "Summarise which cases the officer should take up first and why.",
                                 facts, fallback)
        return AgentResponse(q, "risk_overview", text, df=df, fig=charts.risk_distribution(r), used_llm=used)

    def tool_holder_profile(self, q, holders, steps):
        hid = holders[0]
        row = self.risk_idx.loc[hid]
        f = self.a.findings[self.a.findings.holder_id == hid]
        steps.append(f"Collected {len(f)} finding(s) and evidence transactions for {row['name']}.")
        ev_ids = [t for e in f.evidence for t in e]
        tx = self.case.transactions
        ev = tx[tx.txn_id.isin(ev_ids)].sort_values("amount", ascending=False).head(40)
        ev = ev[["txn_id", "txn_date", "account_id", "mode", "direction", "amount", "counterparty_name", "narration"]]
        if ev.empty and gst.has_gst(self.case):
            gi = self.case.gst_invoices
            ev = gi[gi.invoice_id.isin(ev_ids)].sort_values("taxable_value", ascending=False).head(40)
        full = row.copy()
        full["holder_id"] = hid
        fallback = risk.explain_holder_text(full)
        facts = fallback + f"\nAnomaly score: {row['anomaly']}"
        text, used = llm_explain(self.llm, f"Explain to the officer why {row['name']} was flagged.", facts, fallback)
        return AgentResponse(q, "holder_profile", text, holders=[hid], df=ev,
                             fig=charts.monthly_flows(self.case, hid), used_llm=used)

    def tool_fund_trace(self, q, holders, steps):
        hid = holders[0]
        edges = money_trail.trace_funds(self.case, hid, depth=3, min_amount=100_000)
        steps.append(f"Traced outgoing transfers of ₹1 lakh+ from {self.case.holder_name(hid)} "
                     f"up to 3 hops forward in time: {len(edges)} flow(s).")
        if edges.empty:
            return AgentResponse(q, "fund_trace", "No outgoing transfers of ₹1 lakh or more were found.",
                                 holders=[hid])
        disp = edges.assign(source=edges.source.str.split("|").str[1],
                            target=edges.target.str.split("|").str[1]).sort_values("amount", ascending=False)
        back = disp[disp.target == self.case.holder_name(hid)]
        facts = df_facts(disp) + (f"\nMoney that came BACK to the starting holder: {inr(back.amount.sum())}"
                                  if len(back) else "")
        fallback = (f"Money from {self.case.holder_name(hid)} moved through {len(disp)} flow(s). Largest: "
                    + "; ".join(f"{r.source} → {r.target}: {inr(r.amount)}" for r in disp.head(4).itertuples())
                    + (f". {inr(back.amount.sum())} returned to the starting holder (possible round-tripping)."
                       if len(back) else "."))
        text, used = llm_explain(self.llm, "Explain where the money went.", facts, fallback)
        return AgentResponse(q, "fund_trace", text, holders=[hid], df=disp,
                             fig=charts.sankey(edges, f"Money trail from {self.case.holder_name(hid)}"),
                             used_llm=used)

    def tool_network(self, q, holders, steps):
        if not holders:
            sus = [c for c in self.a.clusters if c["suspicious"]]
            holders = sus[0]["members"] if sus else []
            steps.append(f"No holder named; showing the most suspicious linked group ({len(holders)} entities).")
        G = link_finder.ego_network(self.case, holders, hops=2)
        steps.append(f"Built network: {G.number_of_nodes()} entities, {G.number_of_edges()} links.")
        rows = []
        for a, b, d in G.edges(data=True):
            rows.append({"from": self.case.holder_name(a), "to": self.case.holder_name(b),
                         "link": d["kind"], "amount": d.get("amount")})
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("amount", ascending=False, na_position="last")
        risky = [n for n in G.nodes if self.risk_idx.loc[n, "risk_level"] != "LOW"]
        facts = (f"Entities: {G.number_of_nodes()}. Risky entities among them: "
                 + ", ".join(f"{self.case.holder_name(n)} ({self.risk_idx.loc[n, 'risk_score']:.0f})" for n in risky[:10])
                 + "\nLinks:\n" + (df.head(30).to_csv(index=False) if len(df) else "none"))
        fallback = (f"{G.number_of_nodes()} connected entities found; {len(risky)} of them are flagged as risky. "
                    "Dashed lines are shared phone/address/director links; solid arrows are money transfers.")
        text, used = llm_explain(self.llm, "Explain the important connections in this network.", facts, fallback)
        html = link_finder.to_pyvis_html(self.case, G, highlight=set(holders),
                                         risk=self.risk_idx["risk_score"].to_dict())
        return AgentResponse(q, "network", text, holders=holders, df=df, html=html, used_llm=used)

    def tool_gst(self, q, holders, steps):
        if not gst.has_gst(self.case):
            return AgentResponse(q, "gst", "This case has no GST invoice data loaded.")
        f = self.a.findings[self.a.findings.pattern.str.startswith("gst")]
        if holders:
            f = f[f.holder_id.isin(holders)] if f.holder_id.isin(holders).any() else f
        steps.append(f"Checked {len(self.case.gst_invoices)} GST invoices: {len(self.a.gst_cycles or [])} circular "
                     f"loop(s), {f.holder_id.nunique()} firm(s) with GST red flags.")
        df = f.assign(name=f.holder_id.map(self.case.holder_name),
                      pattern=f.pattern.map(lambda p: risk.PATTERN_LABEL.get(p, p)))[["holder_id", "name", "pattern", "summary"]]
        facts = "\n".join(f"{r.name}: {r.pattern}: {r.summary}" for r in df.head(15).itertuples())
        loops = "; ".join(" → ".join(self.case.holder_name(x) for x in c["route"]) for c in (self.a.gst_cycles or [])[:3])
        fallback = (f"{len(self.a.gst_cycles or [])} circular invoicing loop(s) found" + (f" ({loops})" if loops else "") +
                    f". {int((self.a.findings.pattern == 'gst_risky_itc').sum())} buyer firm(s) claimed ITC from flagged "
                    "suppliers. Red arrows in the network are invoices mostly without e-way bills.")
        text, used = llm_explain(self.llm, f"Officer's question: {q}\nExplain the GST red flags.", facts + "\nLoops: " + loops,
                                 fallback)
        firms = holders or sorted(set(f.holder_id))
        html = gst.gst_network_html(self.case, firms, self.risk_idx["risk_score"].to_dict())
        return AgentResponse(q, "gst", text, holders=holders, df=df, html=html, used_llm=used)

    def tool_sql(self, q, holders, steps):
        if not self.sql_agent:
            return AgentResponse(q, "sql", "Free-form data questions need the local LLM (Ollama). Start Ollama "
                                           "and reload, or use the ready-made analyses in the other tabs.")
        try:
            res = self.sql_agent.answer(q)
        except LLMError as e:
            return AgentResponse(q, "sql", f"Local LLM error: {e}")
        for i, at in enumerate(res["attempts"], 1):
            if at["error"]:
                steps.append(f"SQL attempt {i}: error, asked the model to fix it ({at['error'][:120]})")
            else:
                steps.append(f"SQL attempt {i}: ran, {at.get('rows', '?')} row(s)"
                             + (" - asked the model to double-check the empty result" if at.get("rows") == 0 else ""))
        if res.get("df") is None:
            return AgentResponse(q, "sql", "I could not build a working query for this question. Try rephrasing "
                                           "it more specifically.", sql=res["sql"])
        df = res["df"]
        steps.append(f"Query returned {len(df)} row(s).")
        text, used = llm_explain(self.llm, f"Officer's question: {q}\nExplain the answer.", df_facts(df),
                                 f"The query returned {len(df)} row(s).")
        return AgentResponse(q, "sql", text, df=df, fig=charts.auto_chart(df), sql=res["sql"], used_llm=used)

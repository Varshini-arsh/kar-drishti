"""Data agent - any other question about any loaded table (text-to-SQL on the locked-down DuckDB).

* With the local model: ONE model call writes a read-only SELECT (few-shot examples + the
  schema of every loaded table, including uploaded "other" tables). If the query fails and
  the question still has budget, the model gets one chance to fix it.
* Without a model: questions that closely match a worked example run that example's query.
* Before writing SQL, a simple check makes sure the question is about the loaded tables at all
  (shares a word with a table or column), so off-topic questions are not forced into SQL.
"""
from __future__ import annotations

import re

import pandas as pd

from kar_drishti import charts
from kar_drishti.agents import SQL_SYSTEM, extract_sql, run_sql
from kar_drishti.data import SCHEMA_DOC
from kar_drishti.llm import LLMError
from kar_drishti.sql_examples import EXAMPLES, _words, pick_examples

from .base import AgentResult, df_lines, indian

OFFLINE_MATCH = 0.8      # share of an example's words that must be in the question to reuse it offline
GENERIC = {"show", "list", "total", "number", "count", "many", "much", "table", "data", "value", "name", "id",
           "type", "date", "month", "year", "top", "highest", "lowest", "average", "sum"}


def schema_for(con) -> tuple[str, set[str]]:
    """Schema text for the prompt, and the set of words it contains (for the relevance check)."""
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    lines, words = [], set()
    for t in tables:
        desc, docs = SCHEMA_DOC.get(t, (con_table_comment(t), {}))
        lines.append(f"TABLE {t} -- {desc}")
        words |= _words(t.replace("_", " ") + " " + desc)
        for name, typ, *_ in con.execute(f'DESCRIBE "{t}"').fetchall():
            lines.append(f"  {name} {typ} -- {docs.get(name, '')}")
            words |= _words(name.replace("_", " ") + " " + docs.get(name, ""))
        if t not in SCHEMA_DOC:
            try:
                sample = con.execute(f'SELECT * FROM "{t}" LIMIT 2').df().to_csv(index=False).strip()
                lines.append("  sample rows:\n  " + sample.replace("\n", "\n  "))
            except Exception:
                pass
        lines.append("")
    return "\n".join(lines), words


def con_table_comment(t: str) -> str:
    return f"uploaded table '{t}' (columns below)"


def about_loaded_data(question: str, schema_words: set[str]) -> bool:
    overlap = (_words(question) & schema_words) - GENERIC
    return bool(overlap)


def offline_example(question: str, gst: bool) -> tuple[str, str] | None:
    qw = _words(question)
    best = None
    for q, s in EXAMPLES:
        if not gst and "gst_" in s:
            continue
        ew = _words(q.split("\n")[0])
        if "KNOWN IDS" in q or not ew:
            continue
        share = len(qw & ew) / len(ew)
        extra = len(qw - ew)
        if share >= OFFLINE_MATCH and extra <= 2 and (best is None or share > best[0]):
            best = (share, q, s)
    return (best[1], best[2]) if best else None


# bank-transaction words: these questions go to the worked examples or the model, not the quick table rule
BANK_WORDS = re.compile(r"\b(cash|deposit|withdraw|credit|debit|transaction|bank|income|turnover|itr|neft|rtgs|upi|"
                        r"imps|cheque|atm)", re.I)
NUMERIC_TYPES = ("INT", "DOUBLE", "DECIMAL", "FLOAT", "REAL", "NUMERIC", "HUGEINT", "BIGINT", "SMALLINT")
GROUP_WORDS = re.compile(r"\b(by|per|each|wise|for every)\b", re.I)


ABBREV = {"amt": "amount", "qty": "quantity", "bal": "balance", "dt": "date", "num": "number", "val": "value",
          "txn": "transaction", "dr": "debit", "cr": "credit", "wdl": "withdrawal", "dep": "deposit"}


def _col_words(c: str) -> set[str]:
    """Words of a column name, with common abbreviations spelt out (deposit_amt -> deposit, amount)."""
    return {ABBREV.get(w, w) for w in _words(c.replace("_", " "))}


def simple_sql(question: str, con) -> tuple[str, str] | None:
    """Rule-based SQL for simple questions on ANY table (works without the AI model):
    counts, totals, averages, highest/lowest, 'by <column>' groupings and a filter on a value
    named in the question. Returns (sql, what it does) or None."""
    qw, low = _words(question), question.lower()
    best = None
    for (t,) in con.execute("SHOW TABLES").fetchall():
        cols = [(r[0], r[1]) for r in con.execute(f'DESCRIBE "{t}"').fetchall()]
        score = len(qw & _words(t.replace("_", " "))) * 2 + sum(1 for c, _ in cols if _col_words(c) and _col_words(c) <= qw)
        if score and (best is None or score > best[0]):
            best = (score, t, cols)
    if best is None:
        return None
    _, t, cols = best
    num = [c for c, ty in cols if ty.upper().startswith(NUMERIC_TYPES) and not c.endswith("_id")]
    txt = [c for c, ty in cols if not ty.upper().startswith(NUMERIC_TYPES)]
    mentioned = sorted([c for c, _ in cols if _col_words(c) and _col_words(c) <= qw], key=len, reverse=True)
    overlap = sorted(((len(_col_words(c) & qw), c) for c in num if _col_words(c) & qw), key=lambda x: -x[0])
    measure = next((c for c in mentioned if c in num), None) or (overlap[0][1] if overlap else None) or next(
        (c for c in num if re.search(r"amount|value|collect|demand|target|refund|total", c)), num[0] if num else None)
    group = None
    m = GROUP_WORDS.search(low)
    if m:
        after = _words(low[m.end():])
        group = next((c for c in txt + num if _col_words(c) and _col_words(c) <= after and c != measure), None) or next(
            (c for c in txt if (_col_words(c) - {"name", "id", "code"}) & after and c != measure), None)
    where, found = [], []
    low_v = re.sub(r"\bpeople\b", "persons", low)            # "people" means the PERSON value
    for c in txt:
        try:
            vals = [r[0] for r in con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" LIMIT 300').fetchall()]
        except Exception:
            continue
        hits = [v for v in vals if isinstance(v, str) and len(v) >= 3 and re.search(
            r"(?<![a-z0-9])" + re.escape(v.lower()) + r"(e?s)?(?![a-z0-9])", low_v)]
        if hits:
            found.append((c, hits, len(hits) == len(vals)))
    covered = {v.lower() for _, hits, full in found if full for v in hits}   # words that already mean "everyone"
    for c, hits, full in found:
        if full or all(v.lower() in covered for v in hits):
            continue
        quoted = ", ".join("'" + v.replace("'", "''") + "'" for v in hits)
        where.append(f'"{c}" IN ({quoted})')
    w = (" WHERE " + " AND ".join(where)) if where else ""
    n = int(re.search(r"\btop (\d+)", low).group(1)) if re.search(r"\btop (\d+)", low) else 5
    if re.search(r"how many|count|number of", low):
        if group:
            return (f'SELECT "{group}", COUNT(*) AS count FROM "{t}"{w} GROUP BY 1 ORDER BY 2 DESC', f"count by {group}")
        return f'SELECT COUNT(*) AS count FROM "{t}"{w}', "count of rows"
    agg = "SUM" if re.search(r"total|sum", low) else "AVG" if re.search(r"average|mean", low) else None
    if agg and measure:
        label = ("total_" if agg == "SUM" else "average_") + measure
        if group:
            return (f'SELECT "{group}", ROUND({agg}("{measure}"), 2) AS {label} FROM "{t}"{w} GROUP BY 1 ORDER BY 2 DESC',
                    f"{agg.lower()} of {measure} by {group}")
        return f'SELECT ROUND({agg}("{measure}"), 2) AS {label} FROM "{t}"{w}', f"{agg.lower()} of {measure}"
    if measure and re.search(r"top|highest|largest|biggest|most|maximum", low):
        return f'SELECT * FROM "{t}"{w} ORDER BY "{measure}" DESC LIMIT {n}', f"highest {measure}"
    if measure and re.search(r"lowest|smallest|least|minimum", low):
        return f'SELECT * FROM "{t}"{w} ORDER BY "{measure}" ASC LIMIT {n}', f"lowest {measure}"
    if re.search(r"\b(show|list|display|give)\b", low) and (where or t.lower().replace("_", " ") in low):
        return f'SELECT * FROM "{t}"{w} LIMIT 200', "rows"
    return None


class DataAgent:
    name = "Data agent"

    def __init__(self, con, case=None):
        self.con, self.case = con, case
        self.schema, self.schema_words = schema_for(con)
        self.gst = "TABLE gst_" in self.schema

    def relevant(self, question: str) -> bool:
        """True when the question names a loaded table or column (or matches a worked example)."""
        return about_loaded_data(question, self.schema_words) or offline_example(question, self.gst) is not None \
            or simple_sql(question, self.con) is not None

    def counts(self) -> str:
        tables = [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]
        return f"{len(tables)} tables (" + ", ".join(tables[:8]) + (" …" if len(tables) > 8 else "") + ")"

    def run(self, question: str, llm=None, budget=None, known_ids: str = "") -> AgentResult:
        checked = self.counts()
        if not about_loaded_data(question, self.schema_words) and offline_example(question, self.gst) is None:
            return AgentResult(self.name, "none", ok=False, checked=checked,
                               steps=["The question does not mention anything in the loaded tables."])
        sql, how, steps, df = None, "", [], None
        ex_first = offline_example(question, self.gst)        # a close worked example: no model call needed
        rule_first = simple_sql(question, self.con) if ex_first is None and re.search(
            r"how many|count|number of|total|sum|average|mean", question, re.I) and not (
            BANK_WORDS.search(question) and llm is not None) else None
        if ex_first is not None or rule_first is not None:   # exact built-in query: no model call needed
            llm = None
        if llm is not None and budget is not None and budget.take("sql"):
            try:
                sql, df, steps = self._ask_model(question, llm, budget, known_ids)
                how = "model wrote SQL"
            except LLMError as e:
                steps.append(f"Local model error: {e}")
                sql, df = None, None
        if sql is None or df is None:            # no model, or the model's SQL failed: use the built-in queries
            if sql is not None:
                steps.append("Falling back to the built-in queries.")
            sql = None
            ex = offline_example(question, self.gst)
            if ex is None:
                rule = simple_sql(question, self.con)
                if rule:
                    ex = (f"simple rule: {rule[1]}", rule[0])
            if ex is None:
                why = "needs the local AI model" if llm is None else "the model could not write a working query"
                return AgentResult(self.name, "none", ok=False, checked=checked,
                                   steps=steps + [f"No ready-made query fits, and free-form SQL {why}."])
            sql, how = ex[1], (ex[0] if ex[0].startswith("simple rule") else f"ready-made query for '{ex[0].splitlines()[0]}'")
            df = run_sql(self.con.cursor(), sql)
            steps.append(f"Used a {how} (works without the AI model).")
        if df is None:
            return AgentResult(self.name, "database query", ok=False, checked=checked, sql=sql, steps=steps)
        text, facts = self.describe(df)
        steps.append(f"Query returned {len(df):,} row(s).")
        fig = charts.auto_chart(df) if 2 <= len(df) <= 60 else None      # a chart only when it helps
        must = [text.split(": ", 1)[1].rstrip(".")] if df.shape == (1, 1) else \
            ([f"{len(df):,} record"] if len(df) > 1 else [])
        if len(df) >= 1 and df.shape[1] <= 3:       # small table: the top row's values must appear too
            must += [v for v in (ln.split(": ", 1)[1] for ln in df_lines(df, 1)[0].split(" · ") if ": " in ln)
                     if v not in must][:3]
        return AgentResult(self.name, "database query", ok=True, facts=facts, template=text, checked=checked,
                           records=df, sql=sql, steps=steps, fig=fig, must=must)

    def _ask_model(self, question, llm, budget, known_ids):
        system = SQL_SYSTEM.format(schema=self.schema)
        if not self.gst:
            system += "\nThe GST tables are NOT loaded. Never use gst_invoices or gst_returns."
        msgs = [{"role": "system", "content": system}]
        for q, s in reversed(pick_examples(question, k=3, gst=self.gst)):
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```sql\n{s}\n```"}]
        msgs.append({"role": "user", "content": question + (f"\n{known_ids}" if known_ids else "")})
        steps = []
        sql = extract_sql(llm.chat(msgs, purpose="sql"))
        try:
            df = run_sql(self.con.cursor(), sql)
            steps.append("Model wrote a read-only SQL query; it ran.")
            return sql, df, steps
        except Exception as e:
            err = str(e).split("\n")[0][:300]
            steps.append(f"Query failed ({err[:120]}).")
            if not budget.take("sql-fix"):
                return sql, None, steps + ["No model calls left to fix it."]
            msgs += [{"role": "assistant", "content": f"```sql\n{sql}\n```"},
                     {"role": "user", "content": f"That query failed with this error:\n{err}\nReturn only the fixed query."}]
            sql = extract_sql(llm.chat(msgs, purpose="sql"))
            try:
                df = run_sql(self.con.cursor(), sql)
                steps.append("Model fixed the query; it ran.")
                return sql, df, steps
            except Exception as e2:
                steps.append(f"Fixed query also failed ({str(e2).splitlines()[0][:120]}).")
                return sql, None, steps

    @staticmethod
    def describe(df: pd.DataFrame) -> tuple[str, list[str]]:
        """Template answer + facts, written by code from the query result."""
        if df.empty:
            t = "The query ran on the loaded data and found no matching records."
            return t, [t]
        if df.shape == (1, 1):
            v = df.iat[0, 0]
            if isinstance(v, (int, float)) and pd.notna(v):
                v = indian(v, 0 if float(v).is_integer() else 2)
            t = f"{str(df.columns[0]).replace('_', ' ').capitalize()}: {v}."
            return t, [t]
        lines = df_lines(df, 6)
        head = f"{len(df):,} record{'s' if len(df) != 1 else ''} found" + (
            f" (showing the first {len(lines)})." if len(df) > len(lines) else ".")
        text = head + "\n" + "\n".join(f"- {ln}" for ln in lines)
        return text, [head] + lines


def clean_question(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()

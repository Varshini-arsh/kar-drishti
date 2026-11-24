"""
Money-trail analyzers: deterministic, explainable pattern detection on bank data.

Every detector returns "findings": one row per suspicious holder with
  holder_id, pattern, severity (0-1), summary (plain English), evidence (txn_ids), metrics
so that every statement made by the agent can be traced to actual transactions.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

import config
from .data import Case

FINDING_COLS = ["holder_id", "pattern", "severity", "summary", "evidence", "metrics"]


def inr(x: float) -> str:
    """Format rupees the Indian way: 1234567 -> ₹12.35 lakh / crore."""
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} crore"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} lakh"
    return f"₹{x:,.0f}"


def inr_short(x: float) -> str:
    """Compact rupees for cards: ₹368.2 Cr, ₹47.5 L, ₹9,500."""
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.1f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.1f} L"
    return f"₹{x:,.0f}"


def _finding(holder_id, pattern, severity, summary, evidence, **metrics):
    return {"holder_id": holder_id, "pattern": pattern, "severity": float(np.clip(severity, 0, 1)),
            "summary": summary, "evidence": list(evidence), "metrics": metrics}


def _frame(rows):
    return pd.DataFrame(rows, columns=FINDING_COLS)


def concat_findings(frames):
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else _frame([])


# ---------------------------------------------------------------- 1. structuring
def detect_structuring(case: Case) -> pd.DataFrame:
    tx = case.transactions
    hi = config.CASH_PAN_THRESHOLD
    lo = hi * (1 - config.STRUCTURING_BAND)
    cash = tx[(tx["mode"] == "CASH") & (tx["direction"] == "CR")]
    band = cash[(cash["amount"] >= lo) & (cash["amount"] < hi)]
    rows = []
    for hid, g in band.groupby("holder_id"):
        n = len(g)
        if n < config.STRUCTURING_MIN_COUNT:
            continue
        n_acc = g["account_id"].nunique()
        span = (g["txn_date"].max() - g["txn_date"].min()).days + 1
        total = g["amount"].sum()
        sev = min(1.0, 0.5 + n / 60 + (0.1 if n_acc > 1 else 0))
        rows.append(_finding(
            hid, "structuring", sev,
            f"{n} cash deposits between {inr(lo)} and {inr(hi - 1)} (just below the ₹50,000 PAN-quoting limit), "
            f"totalling {inr(total)}, spread over {n_acc} account(s) within {span} days.",
            g["txn_id"], count=n, accounts=n_acc, total=total, days=span))
    return _frame(rows)


# ---------------------------------------------------------------- 2. round cash withdrawals
def detect_round_cash(case: Case) -> pd.DataFrame:
    tx = case.transactions
    w = tx[(tx["mode"] == "CASH") & (tx["direction"] == "DR")
           & (tx["amount"] >= config.ROUND_CASH_MIN_AMOUNT)
           & (np.isclose(tx["amount"] % config.ROUND_CASH_MULTIPLE, 0))]
    rows = []
    for hid, g in w.groupby("holder_id"):
        if len(g) < config.ROUND_CASH_MIN_COUNT:
            continue
        rows.append(_finding(
            hid, "round_cash", min(1.0, 0.5 + len(g) / 15),
            f"{len(g)} large round-figure cash withdrawals totalling {inr(g['amount'].sum())} "
            f"(e.g. {', '.join(inr(a) for a in g['amount'].head(3))}).",
            g["txn_id"], count=len(g), total=g["amount"].sum()))
    return _frame(rows)


# ---------------------------------------------------------------- 3. pass-through / mule
def _pass_through_ratio(g: pd.DataFrame, window: int) -> tuple[float, float, list]:
    """FIFO-match debits against credits received in the previous `window` days."""
    daily = g.groupby([g["txn_date"].dt.normalize(), "direction"])["amount"].sum().unstack(fill_value=0)
    for c in ("CR", "DR"):
        if c not in daily:
            daily[c] = 0.0
    pending: deque = deque()   # [date, remaining]
    passed = 0.0
    for d, row in daily.sort_index().iterrows():
        if row["CR"] > 0:
            pending.append([d, row["CR"]])
        out = row["DR"]
        while out > 0 and pending:
            cd, rem = pending[0]
            if (d - cd).days > window:
                pending.popleft()
                continue
            used = min(rem, out)
            passed += used
            out -= used
            pending[0][1] -= used
            if pending[0][1] <= 1e-6:
                pending.popleft()
    total_cr = daily["CR"].sum()
    return (passed / total_cr if total_cr else 0.0), total_cr, []


def detect_pass_through(case: Case) -> pd.DataFrame:
    tx = case.transactions
    rows = []
    for acc, g in tx.groupby("account_id"):
        total_cr = g.loc[g.direction == "CR", "amount"].sum()
        if total_cr < config.PASS_THROUGH_MIN_TOTAL:
            continue
        ratio, total_cr, _ = _pass_through_ratio(g, config.PASS_THROUGH_WINDOW_DAYS)
        if ratio < config.PASS_THROUGH_MIN_RATIO:
            continue
        hid = g["holder_id"].iloc[0]
        n_src = g.loc[g.direction == "CR", "counterparty_name"].nunique()
        big_out = g[g.direction == "DR"].nlargest(10, "amount")
        rows.append(_finding(
            hid, "pass_through", min(1.0, 0.4 + ratio * 0.6),
            f"Account {acc}: {ratio:.0%} of the {inr(total_cr)} received was sent out again within "
            f"{config.PASS_THROUGH_WINDOW_DAYS} days. Money came from {n_src} different senders "
            f"(typical 'mule' / pass-through behaviour).",
            pd.concat([big_out["txn_id"], g[g.direction == "CR"].nlargest(10, "amount")["txn_id"]]),
            account_id=acc, ratio=ratio, total_in=total_cr, senders=n_src))
    out = _frame(rows)
    if len(out):   # one finding per holder (keep the strongest account)
        out = out.sort_values("severity", ascending=False).drop_duplicates("holder_id")
    return out


# ---------------------------------------------------------------- 4. round-tripping
def find_round_trips(case: Case) -> list[dict]:
    """Temporal cycle search: money leaving holder A that returns to A through
    other holders within a few days per hop, with similar amounts."""
    tx = case.transactions
    t = tx[(tx.direction == "DR") & tx.counterparty_holder_id.notna()
           & (tx.counterparty_holder_id != tx.holder_id)
           & (tx.amount >= config.ROUND_TRIP_MIN_AMOUNT)]
    t = t.sort_values("txn_date")
    by_src = defaultdict(list)
    for r in t.itertuples(index=False):
        by_src[r.holder_id].append(r)
    tol, maxd, maxlen = (config.ROUND_TRIP_AMOUNT_TOLERANCE, config.ROUND_TRIP_MAX_DAYS_PER_HOP,
                         config.ROUND_TRIP_MAX_LEN)
    chains, seen = [], set()

    def dfs(path):
        last = path[-1]
        if len(path) >= 2 and last.counterparty_holder_id == path[0].holder_id:
            key = tuple(sorted(p.txn_id for p in path))
            if key not in seen:
                seen.add(key)
                chains.append(list(path))
            return
        if len(path) >= maxlen:
            return
        visited = {p.holder_id for p in path}
        for nxt in by_src.get(last.counterparty_holder_id, []):
            dd = (nxt.txn_date - last.txn_date).days
            if dd < 0 or dd > maxd:
                continue
            if not (last.amount * (1 - tol) <= nxt.amount <= last.amount * 1.02):
                continue
            if nxt.counterparty_holder_id in visited and nxt.counterparty_holder_id != path[0].holder_id:
                continue
            dfs(path + [nxt])

    for src_list in by_src.values():
        for e in src_list:
            dfs([e])
    # canonical form: drop rotations of the same chain (keep the earliest start)
    uniq = {}
    for c in chains:
        key = frozenset(p.txn_id for p in c)
        if key not in uniq or c[0].txn_date < uniq[key][0].txn_date:
            uniq[key] = c
    out = []
    for c in uniq.values():
        if c[0].txn_date != min(p.txn_date for p in c):
            continue
        out.append({"start_holder": c[0].holder_id,
                    "path": [p.holder_id for p in c] + [c[0].holder_id],
                    "start_date": c[0].txn_date, "end_date": c[-1].txn_date,
                    "amount_out": c[0].amount, "amount_back": c[-1].amount,
                    "days": (c[-1].txn_date - c[0].txn_date).days,
                    "txn_ids": [p.txn_id for p in c]})
    return out


def detect_round_trips(case: Case, chains: list[dict] | None = None) -> pd.DataFrame:
    chains = find_round_trips(case) if chains is None else chains
    per = defaultdict(list)
    for ch in chains:
        for h in set(ch["path"]):
            per[h].append(ch)
    rows = []
    for hid, chs in per.items():
        starts = [c for c in chs if c["start_holder"] == hid]
        ex = (starts or chs)[0]
        names = " → ".join(case.holder_name(h) for h in ex["path"])
        total = sum(c["amount_out"] for c in chs)
        role = "originated" if starts else "was part of"
        rows.append(_finding(
            hid, "round_trip", min(1.0, 0.6 + 0.1 * len(chs) + (0.15 if starts else 0)),
            f"{role.capitalize()} {len(chs)} circular money flow(s) worth {inr(total)} in total. Example: "
            f"{names}; {inr(ex['amount_out'])} went out and {inr(ex['amount_back'])} came back in {ex['days']} days.",
            [t for c in chs for t in c["txn_ids"]], chains=len(chs), total=total, role=role))
    return _frame(rows)


# ---------------------------------------------------------------- 5. income mismatch / non-filer
def holder_flows(case: Case) -> pd.DataFrame:
    tx = case.transactions
    ext = tx[tx.counterparty_holder_id.isna() | (tx.counterparty_holder_id != tx.holder_id)]
    g = ext.groupby(["holder_id", "direction"])["amount"].sum().unstack(fill_value=0)
    g = g.reindex(columns=["CR", "DR"], fill_value=0).rename(columns={"CR": "total_credits", "DR": "total_debits"})
    cash = tx[(tx["mode"] == "CASH") & (tx.direction == "CR")].groupby("holder_id")["amount"].sum()
    g["cash_deposits"] = cash
    g["n_txn"] = tx.groupby("holder_id").size()
    g["n_counterparties"] = tx.groupby("holder_id")["counterparty_name"].nunique()
    return g.fillna(0).reset_index()


def detect_income_mismatch(case: Case, flows: pd.DataFrame | None = None) -> pd.DataFrame:
    flows = holder_flows(case) if flows is None else flows
    h = case.holders.merge(flows, on="holder_id", how="left").fillna({"total_credits": 0})
    rows = []
    for r in h.itertuples(index=False):
        cr = r.total_credits
        if cr < config.MISMATCH_MIN_CREDITS:
            continue
        top = case.transactions[(case.transactions.holder_id == r.holder_id)
                                & (case.transactions.direction == "CR")].nlargest(10, "amount")["txn_id"]
        if not r.filed:
            rows.append(_finding(
                r.holder_id, "non_filer", min(1.0, 0.6 + cr / 5e7),
                f"No income-tax return filed for the year, but {inr(cr)} was credited to the bank account(s).",
                top, total_credits=cr))
            continue
        ref = r.gross_receipts if r.gross_receipts > 0 else r.declared_income
        basis = "declared turnover" if r.gross_receipts > 0 else "declared income"
        if ref <= 0:
            ref = 1.0
        ratio = cr / ref
        if ratio >= config.MISMATCH_MIN_RATIO:
            rows.append(_finding(
                r.holder_id, "income_mismatch", min(1.0, 0.4 + ratio / 20),
                f"Bank credits of {inr(cr)} are {ratio:.1f}x the {basis} of {inr(ref)} in the ITR.",
                top, total_credits=cr, reference=ref, ratio=ratio, basis=basis))
    return _frame(rows)


# ---------------------------------------------------------------- fund tracer (for Sankey)
def trace_funds(case: Case, holder_id: str, depth: int = 3, min_amount: float = 100_000,
                window_days: int = 30, start_date=None) -> pd.DataFrame:
    """Follow money forward in time from a holder through the case's accounts.
    Returns edges (source, target, amount, txns) for a Sankey diagram."""
    tx = case.transactions
    out = tx[(tx.direction == "DR") & (tx.amount >= min_amount)
             & (tx.counterparty_holder_id.isna() | (tx.counterparty_holder_id != tx.holder_id))]
    edges = defaultdict(lambda: [0.0, 0])
    frontier = [(holder_id, pd.Timestamp(start_date) if start_date else tx.txn_date.min(), None)]
    visited_txn = set()
    for level in range(depth):
        nxt = []
        for h, since, until in frontier:
            q = out[(out.holder_id == h) & (out.txn_date >= since)]
            if until is not None:
                q = q[q.txn_date <= until]
            for r in q.itertuples(index=False):
                if r.txn_id in visited_txn:
                    continue
                visited_txn.add(r.txn_id)
                src = case.holder_name(h)
                if pd.notna(r.counterparty_holder_id):
                    dst = case.holder_name(r.counterparty_holder_id)
                    nxt.append((r.counterparty_holder_id, r.txn_date, r.txn_date + pd.Timedelta(days=window_days)))
                else:
                    dst = f"[outside] {r.counterparty_name}"
                e = edges[(f"L{level}|{src}", f"L{level + 1}|{dst}")]
                e[0] += r.amount
                e[1] += 1
        frontier = nxt
        if not frontier:
            break
    rows = [{"source": s, "target": t, "amount": v[0], "txns": v[1]} for (s, t), v in edges.items()]
    return pd.DataFrame(rows, columns=["source", "target", "amount", "txns"])


def run_all(case: Case) -> dict:
    flows = holder_flows(case)
    chains = find_round_trips(case)
    findings = concat_findings([
        detect_structuring(case), detect_round_cash(case), detect_pass_through(case),
        detect_round_trips(case, chains), detect_income_mismatch(case, flows),
    ])
    return {"findings": findings, "flows": flows, "round_trip_chains": chains}

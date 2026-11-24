"""
GST analytics: circular trading, ITC pass-through entities, invoices without
e-way bills, and buyers claiming ITC from risky suppliers.

Inputs (optional tables in a case):
  gst_invoices : invoice_id, seller_firm_id, buyer_firm_id, invoice_date, hsn,
                 taxable_value, gst_amount, eway_bill_no (blank = none)
  gst_returns  : firm_id, period (YYYY-MM), output_tax, itc_claimed, tax_paid_cash
"""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx
import pandas as pd

import config
from .data import Case
from .money_trail import _finding, _frame, inr


def has_gst(case: Case) -> bool:
    return case.gst_invoices is not None and len(case.gst_invoices) > 0


def invoice_graph(case: Case, min_total: float = 0) -> nx.DiGraph:
    inv = case.gst_invoices
    g = inv.groupby(["seller_firm_id", "buyer_firm_id"]).agg(
        value=("taxable_value", "sum"), gst=("gst_amount", "sum"), n=("invoice_id", "count"),
        no_eway=("eway_bill_no", lambda s: int(s.isna().sum()))).reset_index()
    G = nx.DiGraph()
    for r in g.itertuples(index=False):
        if r.value >= min_total:
            G.add_edge(r.seller_firm_id, r.buyer_firm_id, value=r.value, gst=r.gst, n=r.n, no_eway=r.no_eway)
    return G


def find_circular_trading(case: Case) -> list[dict]:
    G = invoice_graph(case, min_total=config.GST_CYCLE_MIN_EDGE)
    cycles = []
    for cyc in nx.simple_cycles(G, length_bound=config.GST_CYCLE_MAX_LEN):
        if len(cyc) < 2:
            continue
        edges = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
        vals = [G.edges[e]["value"] for e in edges]
        n = sum(G.edges[e]["n"] for e in edges)
        no_e = sum(G.edges[e]["no_eway"] for e in edges)
        cycles.append({"firms": cyc, "route": cyc + [cyc[0]], "min_edge_value": min(vals),
                       "total_value": sum(vals), "invoices": n, "no_eway_share": no_e / n if n else 0,
                       "itc_generated": sum(G.edges[e]["gst"] for e in edges)})
    # keep the largest loops; drop loops whose firms are a subset of a bigger flagged loop
    cycles.sort(key=lambda c: (-len(c["firms"]), -c["total_value"]))
    kept = []
    for c in cycles:
        if not any(set(c["firms"]) <= set(k["firms"]) for k in kept):
            kept.append(c)
    return kept


def detect(case: Case) -> tuple[pd.DataFrame, list[dict]]:
    if not has_gst(case):
        return _frame([]), []
    inv, ret = case.gst_invoices, case.gst_returns
    rows = []
    cycles = find_circular_trading(case)
    per = defaultdict(list)
    for c in cycles:
        for f in c["firms"]:
            per[f].append(c)
    inv_ids_by_firm = defaultdict(list)
    for r in inv.itertuples(index=False):
        inv_ids_by_firm[r.seller_firm_id].append(r.invoice_id)
    for f, cs in per.items():
        ex = cs[0]
        route = " → ".join(case.holder_name(x) for x in ex["route"])
        rows.append(_finding(
            f, "gst_circular", min(1.0, 0.6 + 0.4 * ex["no_eway_share"]),
            f"Part of {len(cs)} circular invoicing loop(s): {route}. Invoices worth {inr(ex['total_value'])} "
            f"went round the loop, generating ITC of {inr(ex['itc_generated'])}; {ex['no_eway_share']:.0%} "
            f"had no e-way bill (goods may not have moved).",
            inv_ids_by_firm[f][:40], loops=len(cs), total=ex["total_value"]))

    flagged_sellers = set(per)
    if ret is not None and len(ret):
        y = ret.groupby("firm_id")[["output_tax", "itc_claimed", "tax_paid_cash"]].sum()
        y["cash_ratio"] = y.tax_paid_cash / y.output_tax.where(y.output_tax > 0)
        for fid, r in y.iterrows():
            if r.output_tax >= config.GST_PASSTHROUGH_MIN_TAX and r.cash_ratio < config.GST_PASSTHROUGH_MAX_CASH_RATIO:
                flagged_sellers.add(fid)
                rows.append(_finding(
                    fid, "gst_itc_passthrough", min(1.0, 0.7 + (config.GST_PASSTHROUGH_MAX_CASH_RATIO - r.cash_ratio) * 10),
                    f"Output GST of {inr(r.output_tax)} was almost fully set off with ITC of {inr(r.itc_claimed)}; "
                    f"only {inr(r.tax_paid_cash)} ({r.cash_ratio:.1%}) was paid in cash. Typical of an entity that "
                    f"only passes on credit.", inv_ids_by_firm[fid][:40], output_tax=r.output_tax,
                    cash_ratio=r.cash_ratio))

    big = inv[inv.taxable_value > config.EWAY_BILL_THRESHOLD]
    for fid, g in big.groupby("seller_firm_id"):
        miss = g[g.eway_bill_no.isna()]
        share = len(miss) / len(g)
        if len(miss) >= config.GST_NO_EWAY_MIN_COUNT and share >= config.GST_NO_EWAY_MIN_SHARE:
            rows.append(_finding(
                fid, "gst_no_eway", min(1.0, 0.4 + share * 0.6),
                f"{len(miss)} of {len(g)} invoices above {inr(config.EWAY_BILL_THRESHOLD)} ({share:.0%}) have no "
                f"e-way bill, worth {inr(miss.taxable_value.sum())}. Movement of goods needs verification.",
                miss.invoice_id.tolist()[:40], count=len(miss), share=share))

    # buyers of credit from flagged sellers (firms that are themselves inside a loop are excluded)
    risky_in = inv[inv.seller_firm_id.isin(flagged_sellers) & ~inv.buyer_firm_id.isin(set(per))]
    for fid, g in risky_in.groupby("buyer_firm_id"):
        itc = g.gst_amount.sum()
        if itc < config.GST_RISKY_ITC_MIN:
            continue
        rows.append(_finding(
            fid, "gst_risky_itc", min(1.0, 0.5 + itc / 2e7),
            f"Claimed ITC of {inr(itc)} on {len(g)} invoices from {g.seller_firm_id.nunique()} supplier(s) that are "
            f"flagged (circular trading / no cash tax paid): " +
            ", ".join(case.holder_name(s) for s in g.seller_firm_id.unique()[:4]) +
            f". {g.eway_bill_no.isna().mean():.0%} of these invoices have no e-way bill.",
            g.invoice_id.tolist()[:40], itc=itc, invoices=len(g)))
    return _frame(rows), cycles


def gst_network_html(case: Case, firms: list[str] | None = None, risk: dict | None = None,
                     height: str = "620px") -> str:
    """Invoice-flow network (seller -> buyer). Red edges: mostly without e-way bills."""
    from pyvis.network import Network
    G = invoice_graph(case)
    if firms:
        keep = set(firms)
        for f in firms:
            if f in G:
                keep |= set(G.predecessors(f)) | set(G.successors(f))
        G = G.subgraph(keep).copy()
    risk = risk or {}
    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line")
    net.barnes_hut(gravity=-7000, spring_length=160)
    for n in G.nodes:
        sc = risk.get(n, 0)
        net.add_node(n, label=case.holder_name(n)[:22], shape="square", size=14 + sc / 6,
                     color="#DC2626" if sc >= config.RISK_HIGH else "#D97706" if sc >= config.RISK_MEDIUM else "#78716C",
                     font={"size": 22, "face": "Segoe UI, Arial", "color": "#1C1917"},
                     title=f"{case.holder_name(n)} ({n})\nRisk score {sc:.0f}")
    mx = max([d["value"] for _, _, d in G.edges(data=True)] or [1])
    for a, b, d in G.edges(data=True):
        bad = d["no_eway"] / d["n"] >= 0.5
        net.add_edge(a, b, width=1 + 7 * d["value"] / mx, color="#DC2626" if bad else "#D6D3D1", arrows="to",
                     title=f"{d['n']} invoices, {inr(d['value'])} taxable, ITC {inr(d['gst'])}; "
                           f"{d['no_eway']} without e-way bill")
    html_doc = net.generate_html(notebook=False)
    html_doc = re.sub(r"<script[^>]+src=\"https?://[^\"]+\"[^>]*>\s*</script>", "", html_doc)
    return re.sub(r"<link[^>]+href=\"https?://[^\"]+\"[^>]*/?>", "", html_doc)

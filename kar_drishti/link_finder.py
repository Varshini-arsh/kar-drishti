"""
Link finder: discovers hidden connections between persons and firms.

Identity links  : shared phone, shared address, director-of
Money links     : transfers between accounts inside the case
Suspicious shell cluster: a group (>= CLUSTER_MIN_SIZE) joined by identity links
that contains 2+ firms which are recently registered or whose bank flows are far
above their declared turnover.
"""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx
import pandas as pd

import config
from .data import Case
from .money_trail import _finding, _frame, inr

TYPE_COLOR = {"PERSON": "#0F766E", "FIRM": "#78716C"}
EDGE_COLOR = {"phone": "#A78BFA", "address": "#F59E0B", "director": "#10B981", "money": "#94A3B8"}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""


def identity_edges(case: Case) -> list[tuple[str, str, str]]:
    h = case.holders
    edges = []
    for col, label in (("phone", "phone"), ("address", "address")):
        groups = defaultdict(list)
        for hid, v in zip(h.holder_id, h[col]):
            k = _norm(v)
            if k:
                groups[k].append(hid)
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.append((members[i], members[j], label))
    firms = h[(h.holder_type == "FIRM") & h.director_person_id.notna()]
    for fid, pid in zip(firms.holder_id, firms.director_person_id):
        edges.append((pid, fid, "director"))
    return edges


def money_edges(case: Case, min_total: float = 100_000) -> pd.DataFrame:
    tx = case.transactions
    t = tx[(tx.direction == "DR") & tx.counterparty_holder_id.notna() & (tx.counterparty_holder_id != tx.holder_id)]
    g = t.groupby(["holder_id", "counterparty_holder_id"]).agg(amount=("amount", "sum"), txns=("txn_id", "count"))
    g = g[g.amount >= min_total].reset_index()
    return g.rename(columns={"holder_id": "src", "counterparty_holder_id": "dst"})


def build_graph(case: Case) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for r in case.holders.itertuples(index=False):
        G.add_node(r.holder_id, name=r.name, holder_type=r.holder_type)
    for a, b, lbl in identity_edges(case):
        G.add_edge(a, b, kind=lbl)
    for r in money_edges(case).itertuples(index=False):
        G.add_edge(r.src, r.dst, kind="money", amount=r.amount, txns=r.txns)
    return G


def find_clusters(case: Case, flows: pd.DataFrame | None = None) -> list[dict]:
    U = nx.Graph()
    U.add_nodes_from(case.holders.holder_id)
    reasons = defaultdict(set)
    for a, b, lbl in identity_edges(case):
        U.add_edge(a, b)
        reasons[frozenset((a, b))].add(lbl)
    h = case.holders.set_index("holder_id")
    fy_end = pd.Timestamp(config.FY_END)
    if flows is None:
        from .money_trail import holder_flows
        flows = holder_flows(case)
    credits = flows.set_index("holder_id")["total_credits"] if len(flows) else pd.Series(dtype=float)
    me = money_edges(case)
    clusters = []
    for comp in nx.connected_components(U):
        if len(comp) < config.CLUSTER_MIN_SIZE:
            continue
        members = sorted(comp)
        firms = [m for m in members if h.loc[m, "holder_type"] == "FIRM"]
        recent = [f for f in firms if pd.notna(h.loc[f, "registration_date"])
                  and (fy_end - h.loc[f, "registration_date"]).days <= config.SHELL_RECENT_REG_DAYS]
        inflated = [f for f in firms if h.loc[f, "gross_receipts"] > 0
                    and credits.get(f, 0) >= config.MISMATCH_MIN_RATIO * h.loc[f, "gross_receipts"]
                    and credits.get(f, 0) >= config.MISMATCH_MIN_CREDITS]
        shared = defaultdict(int)
        for i in members:
            for j in members:
                if i < j:
                    for lbl in reasons.get(frozenset((i, j)), ()):
                        shared[lbl] += 1
        internal = me[me.src.isin(members) & me.dst.isin(members)]
        suspicious_firms = sorted(set(recent) | set(inflated))
        clusters.append({
            "members": members, "firms": firms, "recent_firms": recent, "inflated_firms": inflated,
            "shared_links": dict(shared), "internal_flow": float(internal.amount.sum()),
            "suspicious": len(firms) >= 2 and len(suspicious_firms) >= 2,
            "suspicious_firms": suspicious_firms})
    clusters.sort(key=lambda c: (not c["suspicious"], -c["internal_flow"]))
    for i, c in enumerate(clusters, 1):
        c["cluster_id"] = f"C{i:02d}"
    return clusters


def detect_shell_clusters(case: Case, clusters: list[dict]) -> pd.DataFrame:
    rows = []
    for c in clusters:
        if not c["suspicious"]:
            continue
        links = ", ".join(f"{v} shared {k}" for k, v in c["shared_links"].items())
        for f in c["suspicious_firms"]:
            why = []
            if f in c["recent_firms"]:
                why.append("registered recently")
            if f in c["inflated_firms"]:
                why.append("bank flows far above declared turnover")
            rows.append(_finding(
                f, "shell_cluster", 0.6 + 0.1 * len(why),
                f"Part of linked group {c['cluster_id']} ({len(c['members'])} entities, {len(c['firms'])} firms; "
                f"{links}). This firm is {' and '.join(why)}. Money moved inside the group: {inr(c['internal_flow'])}.",
                [], cluster_id=c["cluster_id"], members=len(c["members"])))
    return _frame(rows)


def ego_network(case: Case, holder_ids: list[str], hops: int = 2, max_nodes: int = 60) -> nx.MultiDiGraph:
    G = build_graph(case)
    U = G.to_undirected(as_view=True)
    keep = set()
    for h in holder_ids:
        if h in U:
            keep |= set(nx.single_source_shortest_path_length(U, h, cutoff=hops))
    if len(keep) > max_nodes:   # keep the closest nodes
        dist = {}
        for h in holder_ids:
            if h in U:
                for n, d in nx.single_source_shortest_path_length(U, h, cutoff=hops).items():
                    dist[n] = min(d, dist.get(n, 99))
        keep = set(sorted(dist, key=dist.get)[:max_nodes])
    return G.subgraph(keep).copy()


def to_pyvis_html(case: Case, G: nx.MultiDiGraph, highlight: set[str] | None = None,
                  risk: dict | None = None, height: str = "620px") -> str:
    """Interactive network as a self-contained HTML string (works offline)."""
    from pyvis.network import Network
    highlight = highlight or set()
    risk = risk or {}
    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line",
                  bgcolor="#ffffff", font_color="#1C1917")
    net.barnes_hut(gravity=-6000, spring_length=140)
    for n, d in G.nodes(data=True):
        score = risk.get(n, 0)
        color = ("#DC2626" if score >= config.RISK_HIGH else "#D97706" if score >= config.RISK_MEDIUM
                 else TYPE_COLOR.get(d.get("holder_type"), "#94A3B8"))
        shape = "dot" if d.get("holder_type") == "PERSON" else "square"
        title = f"{d.get('name')} ({n})\n{d.get('holder_type')}\nRisk score: {score:.0f}"
        net.add_node(n, label=d.get("name", n)[:22], title=title, color=color, shape=shape,
                     size=12 + score / 7, borderWidth=3 if n in highlight else 1,
                     font={"size": 22, "face": "Segoe UI, Arial", "color": "#1C1917"})
    agg = defaultdict(lambda: {"amount": 0.0, "txns": 0})
    for a, b, d in G.edges(data=True):
        k = d.get("kind")
        if k == "money":
            agg[(a, b)]["amount"] += d.get("amount", 0)
            agg[(a, b)]["txns"] += d.get("txns", 0)
        else:
            net.add_edge(a, b, color=EDGE_COLOR[k], title=f"shared {k}" if k != "director" else "director of",
                         dashes=True, arrows="")
    mx = max([v["amount"] for v in agg.values()] or [1])
    for (a, b), v in agg.items():
        net.add_edge(a, b, color="#64748B", width=1 + 6 * v["amount"] / mx,
                     title=f"{inr(v['amount'])} in {v['txns']} transfer(s)", arrows="to")
    html_doc = net.generate_html(notebook=False)
    # pyvis links Bootstrap from a CDN; remove it so the page makes no internet requests
    html_doc = re.sub(r"<script[^>]+src=\"https?://[^\"]+\"[^>]*>\s*</script>", "", html_doc)
    html_doc = re.sub(r"<link[^>]+href=\"https?://[^\"]+\"[^>]*/?>", "", html_doc)
    return html_doc

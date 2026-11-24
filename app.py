"""
Kar Drishti - Offline AI case analyst for Income Tax field officers.
Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _reload_changed_modules():
    """If project files were updated while the app is running, reload them, so a browser refresh
    is enough after an update (no restart needed)."""
    import importlib
    import os
    order = ["config", "kar_drishti.data", "kar_drishti.money_trail", "kar_drishti.link_finder", "kar_drishti.gst",
             "kar_drishti.risk", "kar_drishti.charts", "kar_drishti.llm", "kar_drishti.sql_examples",
             "kar_drishti.agents", "kar_drishti.report", "kar_drishti.statements", "kar_drishti.ui"]
    loaded = [sys.modules[n] for n in order if n in sys.modules]
    changed = any(os.path.getmtime(m.__file__) > getattr(m, "_kd_loaded", 0) for m in loaded
                  if getattr(m, "__file__", None))
    if changed:
        for m in loaded:
            importlib.reload(m)
        st.cache_resource.clear()


def _stamp_modules():
    import os
    for n, m in list(sys.modules.items()):
        if (n == "config" or n.startswith("kar_drishti.")) and getattr(m, "__file__", None):
            m._kd_loaded = os.path.getmtime(m.__file__)


_reload_changed_modules()
import config  # noqa: E402
from kar_drishti import charts, gst, link_finder, money_trail, report, risk, ui  # noqa: E402
from kar_drishti.agents import KarDrishti, llm_explain  # noqa: E402
from kar_drishti.data import build_duckdb, load_case  # noqa: E402
from kar_drishti.llm import LLMError, LocalLLM  # noqa: E402
from kar_drishti.money_trail import inr, inr_short  # noqa: E402

_stamp_modules()

st.set_page_config(page_title="Kar Drishti", page_icon=str(ROOT / "assets" / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.logo(str(ROOT / "assets" / "logo.svg"), size="large", icon_image=str(ROOT / "assets" / "icon.svg"))
PC = ui.PLOTLY_CONFIG
BOT = str(ROOT / "assets" / "icon.svg")
LABEL = risk.PATTERN_LABEL


# ====================================================================== data
@st.cache_resource(show_spinner="Analysing case data on this computer...")
def analyse_folder(path: str, stamp: float):
    case = load_case(path)
    return case, risk.analyze(case)


def ensure_demo():
    if not (config.DEMO_DIR / "transactions.csv").exists():
        from data_generator.generate_data import generate
        with st.spinner("Creating the synthetic demo case (first run only)..."):
            generate(config.DEMO_DIR, config.DEMO_SEED, quiet=True)


def current_case():
    if "uploaded_case" in st.session_state:
        return st.session_state["uploaded_case"]
    folder = Path(st.session_state.get("case_folder", str(config.DEMO_DIR)))
    if folder == config.DEMO_DIR:
        ensure_demo()
    return analyse_folder(str(folder), (folder / "transactions.csv").stat().st_mtime)


case, A = current_case()
case_key = f"{case.name}-{len(case.transactions)}-{id(A)}"
if st.session_state.get("con_key") != case_key:
    st.session_state["con"] = build_duckdb(case, A.risk)
    st.session_state["con_key"] = case_key
    st.session_state["chat"] = []
con = st.session_state["con"]
RISK = A.risk.set_index("holder_id")
S = case.summary()


def holder_label(hid):
    r = RISK.loc[hid]
    return f"{r['name']}  ·  {hid}  ·  {r['risk_score']:.0f}"


def embed_html(html_doc: str, height: int):
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=height)
    else:
        components.html(html_doc, height=height, scrolling=True)


TX_COLS = ["txn_id", "txn_date", "account_id", "mode", "direction", "amount", "counterparty_name", "narration"]
TX_CONFIG = {"amount": st.column_config.NumberColumn("amount (₹)", format="localized"),
             "txn_date": st.column_config.DateColumn("date", format="DD MMM YYYY"),
             "txn_id": "txn", "account_id": "account", "counterparty_name": "counterparty"}


def tx_view(df: pd.DataFrame) -> pd.DataFrame:
    d = df[[c for c in TX_COLS if c in df.columns]].copy()
    if "txn_date" in d:
        d["txn_date"] = pd.to_datetime(d["txn_date"]).dt.date
    return d


def evidence_tables(ids: list[str]):
    tx = case.transactions[case.transactions.txn_id.isin(ids)].sort_values("txn_date")
    if len(tx):
        st.dataframe(tx_view(tx), hide_index=True, width="stretch", height=min(260, 40 + 35 * len(tx)),
                     column_config=TX_CONFIG)
    if gst.has_gst(case):
        gi = case.gst_invoices[case.gst_invoices.invoice_id.isin(ids)].sort_values("invoice_date")
        if len(gi):
            gi = gi[["invoice_id", "invoice_date", "seller_firm_id", "buyer_firm_id", "hsn", "taxable_value",
                     "gst_amount", "eway_bill_no"]].assign(invoice_date=gi.invoice_date.dt.date,
                                                           eway_bill_no=gi.eway_bill_no.fillna("— none —"))
            st.dataframe(gi, hide_index=True, width="stretch", height=min(260, 40 + 35 * len(gi)),
                         column_config={"taxable_value": st.column_config.NumberColumn("taxable (₹)", format="localized"),
                                        "gst_amount": st.column_config.NumberColumn("GST (₹)", format="localized")})


def queue_table(df: pd.DataFrame, key: str, height: int | str = "auto"):
    q = df.copy()
    q["patterns"] = q["flags"].map(lambda f: ", ".join(LABEL.get(x, x) for x in f if x != "anomaly")
                                   + (" + ML outlier" if "anomaly" in f else ""))
    q["declared"] = [inr_short(v) if f else "Not filed" for v, f in zip(q.declared_income, q.filed)]
    q["credits"] = q.total_credits.map(inr_short)
    q["type"] = q.holder_type.str.title()
    view = q[["name", "type", "risk_score", "risk_level", "patterns"]]
    styled = view.style.map(lambda v: f"color:{ui.LEVEL_COLOR.get(v, '#334155')};font-weight:700",
                            subset=["risk_level"])
    return st.dataframe(styled, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                        key=key, height=height,
                        column_config={"name": st.column_config.TextColumn("Name", width="medium"),
                                       "type": st.column_config.TextColumn("Type", width="small"),
                                       "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                                                     format="%.0f", width="small"),
                                       "risk_level": st.column_config.TextColumn("Level", width="small"),
                                       "patterns": st.column_config.TextColumn("What was found", width="large")})


# ====================================================================== sidebar: AI model
if config.PUBLIC_DEMO:
    st.markdown("<div style='background:#FFFBEB;border:1px solid #FDE68A;color:#92400E;border-radius:10px;"
                "padding:8px 12px;font-size:.85rem;margin-bottom:10px'><b>Online demo</b> · all names, PANs and "
                "transactions are synthetic (fake). Uploading real data is disabled. The real tool runs fully offline "
                "on the officer's own computer.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.caption(f"**{case.name}** · {S['holders']} holders · {S['transactions']:,} transactions")
    probe_cfg = st.session_state.setdefault("llm_cfg", {"provider": config.LLM_PROVIDER,
                                                         "url": config.LLM_BASE_URL, "model": None})
    LLM, models, status_msg = None, [], ""
    try:
        probe = LocalLLM(probe_cfg["provider"], probe_cfg["url"], config.LLM_MODEL)
        models = probe.list_models()
    except LLMError as e:
        status_msg = str(e)
    if models:
        if probe_cfg["model"] not in models:
            probe_cfg["model"] = config.LLM_MODEL if config.LLM_MODEL in models else next(
                (m for m in models if m.startswith("qwen2.5")), models[0])
        LLM = LocalLLM(probe_cfg["provider"], probe_cfg["url"], probe_cfg["model"])
        ok, status_msg = LLM.status()
        if not ok:
            LLM = None
    else:
        if config.PUBLIC_DEMO and not config.LLM_EXPECTED:
            status_msg = "Not available in this online demo. All analysis pages work."
        elif config.PUBLIC_DEMO:
            status_msg = "The AI model is starting up. Refresh in a minute."
        else:
            status_msg = status_msg or "Start Ollama: open PowerShell and run  ollama serve"
    ui.ai_status(LLM is not None, probe_cfg["model"] or config.LLM_MODEL, status_msg)
    with st.expander("Model settings"):
        prov = st.selectbox("Server", ["ollama", "openai"], index=0 if probe_cfg["provider"] == "ollama" else 1,
                            format_func=lambda p: "Ollama" if p == "ollama" else "LM Studio / llama.cpp")
        url = st.text_input("Address", probe_cfg["url"])
        mdl = st.selectbox("Model", models, index=models.index(probe_cfg["model"])) if models else None
        if (prov, url, mdl) != (probe_cfg["provider"], probe_cfg["url"], probe_cfg["model"]):
            probe_cfg.update(provider=prov, url=url, model=mdl)
            st.rerun()
        st.caption("Only local addresses are allowed. No data leaves this computer.")

AGENT = KarDrishti(case, A, con, LLM)


# ====================================================================== pages
def page_dashboard():
    r = A.risk
    n_high, n_med = int((r.risk_level == "HIGH").sum()), int((r.risk_level == "MEDIUM").sum())
    ui.header("Overview", f"{S['date_from']:%d %b %Y} to {S['date_to']:%d %b %Y} · analysed on this computer")
    ui.kpis([("People and firms", f"{S['holders']}", f"{S['persons']} people · {S['firms']} firms"),
             ("Transactions", f"{S['transactions']:,}", f"{S['accounts']} bank accounts"),
             ("Money in", inr_short(S["total_credits"]), "total credits"),
             ("Need review", f"{n_high + n_med}", f"{n_high} high · {n_med} medium", "red")])
    a, b = st.columns(2, gap="medium")
    with a:
        ui.section("Risk split")
        st.plotly_chart(charts.risk_donut(r), width="stretch", config=PC)
    with b:
        ui.section("What was found")
        if len(A.findings):
            st.plotly_chart(charts.pattern_counts(A.findings, LABEL), width="stretch", config=PC)
    ui.section("Review queue")
    st.caption("Sorted by risk. Click a row to open the case.")
    qv = st.session_state.setdefault("queue_v", 0)
    ev = queue_table(r[r.risk_level != "LOW"], f"queue-{qv}")
    if ev and ev.selection.rows:
        hid = r[r.risk_level != "LOW"].iloc[ev.selection.rows[0]]["holder_id"]
        st.session_state["selected_holder"] = hid
        st.session_state["queue_v"] = qv + 1          # fresh table (no selection) when coming back
        st.switch_page(PAGES["case"])


def page_case():
    ids = A.risk.holder_id.tolist()
    sel = st.session_state.get("selected_holder", ids[0])
    hid = st.selectbox("Choose a person or firm", ids, index=ids.index(sel) if sel in ids else 0,
                       format_func=holder_label)
    st.session_state["selected_holder"] = hid
    r = RISK.loc[hid]
    h = case.holders.set_index("holder_id").loc[hid]
    is_firm = h["holder_type"] == "FIRM"
    bits = ["Firm" if is_firm else f"Person · {h['occupation']}", h["city"] if pd.notna(h["city"]) else "",
            f"PAN {h['pan']}" if pd.notna(h["pan"]) and h["pan"] else ""]
    st.markdown(f"<div class='kd-profile'><div class='n'>{ui.esc(r['name'])}{ui.badge(r['risk_level'])}</div>"
                f"<div class='sub'>{ui.esc(' · '.join(b for b in bits if b))}</div></div>", unsafe_allow_html=True)
    if not r["filed"]:
        declared = ("Declared income", "Not filed", "no ITR for the year", "red")
    elif is_firm or h["gross_receipts"] > 0:
        declared = ("Declared turnover", inr_short(h["gross_receipts"]), "as per ITR")
    else:
        declared = ("Declared income", inr_short(r["declared_income"]), "as per ITR")
    ratio = r["total_credits"] / max(h["gross_receipts"] if h["gross_receipts"] > 0 else r["declared_income"], 1)
    tone = {"HIGH": "red", "MEDIUM": "amber"}.get(r["risk_level"], "")
    ui.kpis([("Risk score", f"{r['risk_score']:.0f} / 100", f"ML anomaly {r['anomaly']:.2f}", tone),
             declared,
             ("Money in", inr_short(r["total_credits"]), f"{ratio:,.1f}× declared" if r["filed"] else "total credits"),
             ("Cash deposits", inr_short(r["cash_deposits"]), "total in the year")])

    tab1, tab_c, tab2, tab3 = st.tabs(["Findings", "Charts", "Money trail", "Transactions"])
    with tab1:
        f = A.findings[A.findings.holder_id == hid]
        if f.empty and "anomaly" not in r["flags"]:
            st.success("No suspicious pattern detected for this holder.")
        for fr in f.itertuples(index=False):
            ui.finding_card(LABEL.get(fr.pattern, fr.pattern), fr.summary, fr.severity,
                            f"severity {fr.severity:.2f}")
            if fr.evidence:
                with st.expander(f"Evidence · {len(fr.evidence)} record(s)"):
                    evidence_tables(fr.evidence)
        if "anomaly" in r["flags"]:
            ui.finding_card(LABEL["anomaly"], "Behaviour is in the top 10% most unusual compared with other holders "
                                              "(Isolation Forest).", r["anomaly"], f"score {r['anomaly']:.2f}")
        key = f"explain-{case_key}-{hid}"
        if st.button("Explain with AI", icon=":material/auto_awesome:", disabled=LLM is None,
                     help=None if LLM else "Start Ollama to enable"):
            row = r.copy()
            row["holder_id"] = hid
            fb = risk.explain_holder_text(row)
            with st.spinner("Local model is writing the explanation..."):
                st.session_state[key] = llm_explain(LLM, f"Explain to the officer why {r['name']} was flagged.", fb, fb)[0]
        if key in st.session_state:
            ui.note(st.session_state[key])
    with tab_c:
        a, b = st.columns(2, gap="medium")
        with a:
            ui.section("Monthly money in / out")
            st.plotly_chart(charts.monthly_flows(case, hid), width="stretch", config=PC)
        with b:
            ui.section("Cash deposits")
            sc = charts.cash_deposits_scatter(case, hid)
            if sc:
                st.plotly_chart(sc, width="stretch", config=PC)
            else:
                st.caption("No cash deposits.")
    with tab2:
        c1, c2 = st.columns(2)
        min_amt = c1.select_slider("Minimum transfer", [50_000, 100_000, 200_000, 500_000, 1_000_000], 100_000,
                                   format_func=inr_short)
        depth = c2.slider("Hops to follow", 1, 4, 3)
        edges = money_trail.trace_funds(case, hid, depth=depth, min_amount=min_amt)
        fig = charts.sankey(edges)
        if fig:
            back = edges[edges.target.str.split("|").str[1] == r["name"]]
            if len(back):
                st.warning(f"{inr(back.amount.sum())} came back to {r['name']} through other entities — "
                           "possible round-tripping.", icon=":material/sync_problem:")
            st.plotly_chart(fig, width="stretch", config=PC)
            st.caption("Blue = inside this case · red = left the case (outside party) · red links = money returning "
                       "to the start.")
        else:
            st.info("No outgoing transfers above the minimum.")
    with tab3:
        tx = case.transactions[case.transactions.holder_id == hid]
        modes = sorted(tx["mode"].unique())
        pick = st.pills("Mode", modes, selection_mode="multi", default=modes, label_visibility="collapsed")
        st.dataframe(tx_view(tx[tx["mode"].isin(pick or modes)]), hide_index=True, width="stretch",
                     column_config=TX_CONFIG, height=480)


EXAMPLES = [
    "Which cases should I investigate first?",
    "Show total cash deposits by month",
    "Who received more than 10 times their declared income?",
    "Which firms are passing fake GST input tax credit?",
    "List firms that paid less than 1% of their GST in cash",
    "Show the shell firm network",
]
TOOL_NAME = {"risk_overview": "Risk ranking", "holder_profile": "Holder profile", "fund_trace": "Money trail",
             "network": "Link finder", "sql": "Database query", "gst": "GST analysis"}


def render_response(resp):
    st.markdown(f"<span class='kd-chip'>{TOOL_NAME.get(resp.tool, resp.tool)}</span>"
                + ("<span class='kd-chip' style='background:#ECFDF5;color:#047857'>AI explanation</span>"
                   if resp.used_llm else ""), unsafe_allow_html=True)
    st.markdown(resp.text)
    if resp.fig is not None:
        st.plotly_chart(resp.fig, width="stretch", config=PC)
    if resp.html:
        embed_html(resp.html, 560)
    if resp.df is not None and len(resp.df):
        if "txn_id" in resp.df.columns:
            d, cfg = tx_view(resp.df), TX_CONFIG
        else:
            d = resp.df
            cfg = {c: st.column_config.NumberColumn(c.replace("_", " "), format="localized")
                   for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not c.endswith("_id")}
        st.dataframe(d, hide_index=True, width="stretch", height=min(380, 40 + 35 * len(d)), column_config=cfg)
    if resp.steps or resp.sql:
        with st.expander("How I got this answer"):
            for stp in resp.steps:
                st.markdown(f"- {stp}")
            if resp.sql:
                st.code(resp.sql, language="sql")
            st.caption("Numbers come from code run on the data, never from the AI's memory.")


def page_chat():
    ui.header("Ask Kar Drishti", "Ask in plain English. The agent picks a tool, runs it on the case data and explains "
                                 "the result.")
    if LLM is None:
        st.info(("This online demo has no AI model (the free host is too small to run one). " if config.PUBLIC_DEMO
                 else "The local AI model is not connected. ")
                + "Questions about risk ranking, a named person or firm, money trails, links and GST still work; "
                  "free-form data questions need the local model on the officer's computer.", icon=":material/info:")
    top = A.risk.iloc[0]
    ex = (EXAMPLES[:4] if LLM else ["Which cases should I investigate first?",
                                     "Which firms are passing fake GST input tax credit?",
                                     "Show the shell firm network"]) + [f"Why is {top['name']} flagged?"]
    if A.chains:
        ex.append(f"Where did the money from {case.holder_name(A.chains[0]['start_holder'])} go?")
    clicked = st.pills("Try", ex, selection_mode="single", key=f"pill-{len(st.session_state['chat'])}",
                       label_visibility="collapsed")
    for item in st.session_state["chat"]:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(item.question)
        with st.chat_message("assistant", avatar=BOT):
            render_response(item)
    q = st.chat_input("Ask about this case, e.g. Which accounts received money from both Kuber Traders and Vardhan "
                      "Enterprises?") or clicked
    if q:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(q)
        with st.chat_message("assistant", avatar=BOT):
            with st.spinner("Thinking — the local model can take 20–40 seconds..."):
                resp = AGENT.ask(q)
            render_response(resp)
        st.session_state["chat"].append(resp)
        st.rerun()


def page_links():
    ui.header("Link finder", "People and firms connected by shared phones, addresses, directorships and money.")
    cl = A.clusters
    sus = [c for c in cl if c["suspicious"]]
    ui.kpis([("Linked groups", len(cl), "3 or more people or firms"),
             ("Suspicious groups", len(sus), "possible shell networks", "red"),
             ("Money inside them", inr_short(sum(c["internal_flow"] for c in sus)), "transfers within the groups")])
    mode = st.segmented_control("Show", ["Linked group", "Person or firm"], default="Linked group",
                                label_visibility="collapsed")
    if mode == "Person or firm":
        c1, c2 = st.columns([3, 1])
        members = [c1.selectbox("Person or firm", A.risk.holder_id.tolist(), format_func=holder_label)]
        hops = c2.slider("Degrees", 1, 3, 2)
    elif cl:
        g = st.selectbox("Group", [c["cluster_id"] for c in cl],
                         format_func=lambda x: next(f"{c['cluster_id']} · {len(c['members'])} entities"
                                                    + ("  ·  suspicious" if c["suspicious"] else "")
                                                    for c in cl if c["cluster_id"] == x))
        members, hops = next(c for c in cl if c["cluster_id"] == g)["members"], 1
    else:
        st.info("No linked groups found.")
        return
    G = link_finder.ego_network(case, members, hops=hops)
    embed_html(link_finder.to_pyvis_html(case, G, set(members), RISK["risk_score"].to_dict(), height="560px"), 580)
    st.caption("● person  ■ firm  ·  red = high risk, amber = medium  ·  dashed = shared phone / address / director  "
               "·  arrows = money (thicker = more). Drag nodes to rearrange, hover for details.")
    if cl:
        with st.expander("All linked groups"):
            st.dataframe(pd.DataFrame([{
                "group": c["cluster_id"], "entities": len(c["members"]), "firms": len(c["firms"]),
                "suspicious": "Yes" if c["suspicious"] else "", "money inside": inr_short(c["internal_flow"]),
                "shared links": ", ".join(f"{v} {k}" for k, v in c["shared_links"].items()),
                "members": ", ".join(case.holder_name(m) for m in c["members"][:6]) + (" …" if len(c["members"]) > 6 else "")}
                for c in cl]), hide_index=True, width="stretch")


def page_gst():
    ui.header("GST network", "Circular trading, credit pass-through firms, missing e-way bills and risky ITC.")
    if not gst.has_gst(case):
        st.info("This case has no GST invoice data. Add gst_invoices.csv and gst_returns.csv to the case folder.")
        return
    inv = case.gst_invoices
    f = A.findings[A.findings.pattern.str.startswith("gst")]
    risky_sellers = set(f[f.pattern.isin(["gst_circular", "gst_itc_passthrough"])].holder_id)
    ui.kpis([("B2B invoices", f"{len(inv):,}", f"{inr_short(inv.taxable_value.sum())} taxable value"),
             ("ITC on invoices", inr_short(inv.gst_amount.sum()), "credit available to buyers"),
             ("ITC from flagged sellers", inr_short(inv[inv.seller_firm_id.isin(risky_sellers)].gst_amount.sum()),
              "likely fake credit", "red"),
             ("Circular loops", len(A.gst_cycles or []), "invoice rings", "red")])
    if A.gst_cycles:
        ui.section("Circular invoicing loops")
        for cy in A.gst_cycles:
            route = " → ".join(case.holder_name(x) for x in cy["route"])
            ui.finding_card(route, f"{cy['invoices']} invoices · {inr(cy['total_value'])} went round the loop · "
                                   f"ITC generated {inr(cy['itc_generated'])} · {cy['no_eway_share']:.0%} without "
                                   f"e-way bill", 0.9, f"{len(cy['firms'])} firms")
    t1, t2, t3 = st.tabs(["Invoice network", "Firms with red flags", "Monthly returns"])
    with t1:
        firms_all = sorted(set(inv.seller_firm_id) | set(inv.buyer_firm_id), key=lambda x: -RISK.loc[x, "risk_score"])
        opt = st.selectbox("Show", ["Flagged firms and their trading partners"] + firms_all,
                           format_func=lambda x: x if x.startswith("Flagged") else holder_label(x),
                           label_visibility="collapsed")
        focus = sorted(set(f.holder_id)) if opt.startswith("Flagged") else [opt]
        embed_html(gst.gst_network_html(case, focus, RISK["risk_score"].to_dict(), height="540px"), 560)
        st.caption("■ firm (red = high risk) · arrow = invoices seller → buyer, thicker = more value · "
                   "red arrow = most invoices have no e-way bill.")
    with t2:
        if len(f):
            t = f.assign(name=f.holder_id.map(case.holder_name), red_flag=f.pattern.map(lambda p: LABEL.get(p, p)))
            st.dataframe(t[["holder_id", "name", "red_flag", "summary"]], hide_index=True, width="stretch",
                         column_config={"summary": st.column_config.TextColumn("details", width="large")})
        else:
            st.success("No GST red flags.")
    with t3:
        if case.gst_returns is not None and len(case.gst_returns):
            fid = st.selectbox("Firm", firms_all, format_func=holder_label, key="gst_firm")
            rr = case.gst_returns[case.gst_returns.firm_id == fid].sort_values("period")
            st.plotly_chart(charts.gst_monthly(rr), width="stretch", config=PC)
            tot = rr[["output_tax", "itc_claimed", "tax_paid_cash"]].sum()
            st.caption(f"Year: output tax {inr(tot.output_tax)} · ITC {inr(tot.itc_claimed)} · paid in cash "
                       f"{inr(tot.tax_paid_cash)} ({tot.tax_paid_cash / max(tot.output_tax, 1):.1%} of liability)")


def page_report():
    ui.header("Case report", "A printable summary with findings and evidence. Open the downloaded file and print to PDF.")
    flagged = A.risk[A.risk.risk_level != "LOW"].holder_id.tolist()
    sel = st.multiselect("Holders to include", A.risk.holder_id.tolist(), default=flagged[:15], format_func=holder_label)
    notes = st.text_area("Officer's notes (optional)", height=90)
    use_ai = st.toggle("Add an AI-written summary", value=LLM is not None, disabled=LLM is None)
    if st.button("Generate report", type="primary", icon=":material/description:"):
        summary = ""
        if use_ai and LLM:
            facts = "\n".join(f"{x.name} ({x.holder_id}) score {x.risk_score:.0f}: " + " | ".join(x.reasons)
                              for x in A.risk[A.risk.holder_id.isin(sel)].head(10).itertuples())
            with st.spinner("Local model is writing the summary..."):
                summary = llm_explain(LLM, "Write a short executive summary of this case for a senior officer.",
                                      facts, "")[0]
        st.session_state["report_html"] = report.build_report(case, A, sel or None, notes, summary)
    if "report_html" in st.session_state:
        st.download_button("Download report", st.session_state["report_html"], file_name="kar_drishti_case_report.html",
                           mime="text/html", icon=":material/download:")
        embed_html(st.session_state["report_html"], 760)


def page_accuracy():
    ui.header("Accuracy check", "How many of the fraud cases planted in the synthetic demo data were found.")
    if not case.ground_truth:
        st.info("Accuracy can only be measured on the synthetic demo case, where the fraud cases are known.")
        return
    e = risk.evaluate(A, case.ground_truth)
    o = e["overall"]
    ui.kpis([("Planted suspicious entities", o["planted_entities"], "known fraud cases"),
             ("Flagged for review", o["flagged_medium_or_high"], "by Kar Drishti"),
             ("Recall", f"{o['recall']:.0%}", "planted cases caught", "teal"),
             ("Precision", f"{o['precision']:.0%}" if o["precision"] is not None else "–", "flags that were right", "teal")])
    rows = [{"pattern": LABEL.get(k, k), "planted": v["planted"], "detected": v["detected"],
             "recall": (v["recall"] or 0) * 100, "missed": ", ".join(v["missed"])} for k, v in e["per_pattern"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"recall": st.column_config.ProgressColumn("recall", min_value=0, max_value=100,
                                                                         format="%.0f%%")})
    st.caption("Planted patterns in synthetic data are cleaner than real cases; expect more borderline cases on real "
               "data. Thresholds can be tuned in config.py.")
    with st.expander("What was planted"):
        for n in case.ground_truth.get("notes", []):
            st.markdown(f"- {n}")


def page_load():
    ui.header("Load data", "Synthetic demo case only in the online demo." if config.PUBLIC_DEMO else
              "Use the synthetic demo case, upload bank statements, or point to a case folder.")
    if config.PUBLIC_DEMO:
        st.info("This is the public online demo, so uploading real bank statements or case folders is disabled. "
                "Use the synthetic demo case below; the offline version on an officer's computer supports uploads.",
                icon=":material/lock:")
        tabs = st.tabs(["Demo case"])
        t1, t2, t3 = tabs[0], None, None
    else:
        t1, t2, t3 = st.tabs(["Demo case", "Upload bank statements", "Case folder"])
    with t1:
        st.write("A synthetic jurisdiction (100% fake data) with planted fraud patterns, for demonstration and testing.")
        seed = st.number_input("Random seed", value=config.DEMO_SEED, step=1)
        a, b = st.columns(2)
        if a.button("Use demo case", width="stretch"):
            st.session_state.pop("uploaded_case", None)
            st.session_state["case_folder"] = str(config.DEMO_DIR)
            st.rerun()
        if b.button("Generate a new demo case", width="stretch"):
            from data_generator.generate_data import generate
            generate(config.DEMO_DIR, int(seed), quiet=True)
            st.session_state.pop("uploaded_case", None)
            st.session_state["case_folder"] = str(config.DEMO_DIR)
            st.cache_resource.clear()
            st.rerun()
    if t2 is None:
        return
    with t2:
        st.write("One file per account (CSV or Excel from net banking). Date, narration, debit and credit columns are "
                 "detected automatically. Sample files are in the `sample_statements` folder.")
        files = st.file_uploader("Statements", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
                                 label_visibility="collapsed")
        items = []
        for i, fl in enumerate(files or []):
            guess = " ".join(Path(fl.name).stem.replace("statement_", "").split("_")[:-1]) or Path(fl.name).stem
            with st.container(border=True):
                st.caption(fl.name)
                c = st.columns([3, 1.3, 1.8, 1])
                name = c[0].text_input("Account holder", guess, key=f"n{i}")
                htype = c[1].selectbox("Type", ["PERSON", "FIRM"], key=f"t{i}")
                inc = c[2].number_input("Declared income / turnover (₹)", value=0, step=50000, key=f"i{i}")
                filed = c[3].checkbox("ITR filed", value=True, key=f"f{i}")
            items.append({"file_name": fl.name, "data": fl.getvalue(), "holder_name": name, "holder_type": htype,
                          "declared_income": inc, "gross_receipts": inc if htype == "FIRM" else 0, "filed": filed})
        if items and st.button("Analyse these statements", type="primary"):
            from kar_drishti.statements import build_case_from_statements
            try:
                with st.spinner("Reading statements and analysing..."):
                    up = build_case_from_statements(items, "Uploaded statements")
                    st.session_state["uploaded_case"] = (up, risk.analyze(up))
                st.switch_page(PAGES["dashboard"])
            except Exception as ex:
                st.error(f"Could not read the statements: {ex}")
    with t3:
        st.write("A folder with persons.csv, firms.csv, accounts.csv, transactions.csv and itr_filings.csv "
                 "(optional: gst_invoices.csv, gst_returns.csv) in the same format as `data/demo_case`.")
        folder = st.text_input("Folder path", str(config.DEMO_DIR))
        if st.button("Load folder"):
            if (Path(folder) / "transactions.csv").exists():
                st.session_state.pop("uploaded_case", None)
                st.session_state["case_folder"] = folder
                st.rerun()
            else:
                st.error("transactions.csv not found in that folder.")


PAGES = {
    "dashboard": st.Page(page_dashboard, title="Overview", icon=":material/space_dashboard:", url_path="overview",
                         default=True),
    "case": st.Page(page_case, title="Case analysis", icon=":material/person_search:", url_path="case"),
    "chat": st.Page(page_chat, title="Ask Kar Drishti", icon=":material/forum:", url_path="ask"),
    "links": st.Page(page_links, title="Link finder", icon=":material/hub:", url_path="links"),
    "gst": st.Page(page_gst, title="GST network", icon=":material/receipt_long:", url_path="gst"),
    "report": st.Page(page_report, title="Case report", icon=":material/description:", url_path="report"),
    "accuracy": st.Page(page_accuracy, title="Accuracy check", icon=":material/verified:", url_path="accuracy"),
    "load": st.Page(page_load, title="Load data", icon=":material/upload_file:", url_path="data"),
}
nav = st.navigation({"Investigate": [PAGES["dashboard"], PAGES["case"], PAGES["chat"], PAGES["links"], PAGES["gst"]],
                     "Output": [PAGES["report"], PAGES["accuracy"]],
                     "Data": [PAGES["load"]]})
nav.run()

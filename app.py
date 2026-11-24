"""
Pragya - Insight for every tax desk.
An offline, multi-agent assistant for Indian Income Tax officers, built on the
Kar Drishti risk engine (kar_drishti/). Name, tagline, logo and colour live in branding.py.

Run:  streamlit run app.py
Screens: Ask (home) · Cases · Revenue · Library, plus a small Settings icon.
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
    order = ["config", "branding", "pragya.profile", "pragya.system", "pragya.audit", "pragya.checker", "pragya.base",
             "pragya.llm", "pragya.uploads", "pragya.revenue", "pragya.charts", "kar_drishti.data", "kar_drishti.money_trail", "kar_drishti.link_finder", "kar_drishti.gst",
             "kar_drishti.risk", "kar_drishti.charts", "kar_drishti.llm", "kar_drishti.sql_examples",
             "kar_drishti.agents", "kar_drishti.report", "kar_drishti.statements", "kar_drishti.ui",
             "pragya.library", "pragya.risk_agent", "pragya.data_agent", "pragya.revenue_agent",
             "pragya.law_agent", "pragya.selftest_risk", "pragya.lead"]
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
        if (n in ("config", "branding") or n.startswith(("kar_drishti.", "pragya."))) and getattr(m, "__file__", None):
            m._kd_loaded = os.path.getmtime(m.__file__)


_reload_changed_modules()
import config  # noqa: E402
from kar_drishti import charts, gst, link_finder, money_trail, report, risk, ui  # noqa: E402
from kar_drishti.agents import llm_explain  # noqa: E402
from kar_drishti.data import build_duckdb, load_case  # noqa: E402
from kar_drishti.llm import LLMError  # noqa: E402
from kar_drishti.money_trail import inr, inr_short  # noqa: E402

_stamp_modules()


_reload_changed_modules()
import branding  # noqa: E402
import config  # noqa: E402
from kar_drishti import charts, gst, link_finder, money_trail, report, risk, ui  # noqa: E402
from kar_drishti.agents import llm_explain  # noqa: E402
from kar_drishti.data import build_duckdb, load_case  # noqa: E402
from kar_drishti.llm import LLMError  # noqa: E402
from kar_drishti.money_trail import inr, inr_short  # noqa: E402
from pragya import audit, uploads  # noqa: E402
from pragya import profile as prof  # noqa: E402
from pragya.lead import GENERAL_LABEL, Lead  # noqa: E402
from pragya.revenue import load_demo as load_demo_revenue, LEVELS, Revenue  # noqa: E402
from pragya.revenue_agent import RevenueAgent  # noqa: E402
from pragya.selftest_risk import risk_self_test  # noqa: E402
from pragya.law_agent import LawAgent  # noqa: E402
from pragya.library import Library, parse_document  # noqa: E402
from pragya.llm import PragyaLLM, choose_model  # noqa: E402
from pragya.system import memory_gb  # noqa: E402

_stamp_modules()

ASSETS = branding.asset_paths(ROOT / "assets")
st.set_page_config(page_title=branding.page_title(), page_icon=ASSETS["icon"], layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.logo(ASSETS["logo"], size="large", icon_image=ASSETS["icon"])
PC = ui.PLOTLY_CONFIG
BOT = ASSETS["icon"]
LABEL = risk.PATTERN_LABEL
PROFILE = st.session_state.setdefault("officer_profile",
                                      prof.Profile() if config.PUBLIC_DEMO else prof.load_profile())


# ====================================================================== data
@st.cache_resource(show_spinner="Checking the case data on this computer...")
def analyse_folder(path: str, stamp: float, gst_on: bool):
    case = load_case(path)
    return case, risk.analyze(case, include_gst=gst_on)


def ensure_demo():
    if not (config.DEMO_DIR / "transactions.csv").exists():
        from data_generator.generate_data import generate
        with st.spinner("Creating the synthetic demo case (first run only)..."):
            generate(config.DEMO_DIR, config.DEMO_SEED, quiet=True)


GST_ON = bool(st.session_state.setdefault("gst_on", config.GST_ENABLED))


def current_case():
    if "uploaded_case" in st.session_state:
        return st.session_state["uploaded_case"]
    folder = Path(st.session_state.get("case_folder", str(config.DEMO_DIR)))
    if folder == config.DEMO_DIR:
        ensure_demo()
    return analyse_folder(str(folder), (folder / "transactions.csv").stat().st_mtime, GST_ON)


case, A = current_case()
case_key = f"{case.name}-{len(case.transactions)}-{id(A)}-{A.gst_on}"
if st.session_state.get("con_key") != case_key:
    st.session_state["con"] = build_duckdb(case, A.risk, include_gst=A.gst_on)
    # synthetic revenue demo files first, so uploaded files with the same type replace them
    demo_rev = load_demo_revenue()
    if demo_rev is not None:
        for kind, attr in [("revenue targets", "targets"), ("collections", "collections"),
                           ("last year's collections", "last_year"), ("demand register", "demands"),
                           ("recovery actions", "actions"), ("refunds", "refunds")]:
            if getattr(demo_rev, attr) is not None:
                uploads.load_into_duckdb(st.session_state["con"], uploads.TABLE_NAME[kind], getattr(demo_rev, attr))
    # every file added with "Add data" is loaded again (tables into DuckDB, documents for the Library)
    st.session_state["up_tables"], st.session_state["up_docs"] = uploads.load_saved(st.session_state["con"])
    st.session_state["revenue"] = Revenue.from_tables(
        {t["kind"]: t["df"] for t in st.session_state["up_tables"]}, base=demo_rev)
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
    q["patterns"] = q["breakdown"].map(lambda b: "  ·  ".join(f"{x['label']} +{x['points']:.0f}" for x in b))
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
                                       "patterns": st.column_config.TextColumn("What was found (points)",
                                                                               width="large")})




# ====================================================================== local AI model
def connect_model():
    """Find the local model server and pick an INSTALLED model (never downloads one).
    Returns (model or None, installed models, plain-English status)."""
    cfg = st.session_state.setdefault("llm_cfg", {"provider": config.LLM_PROVIDER, "url": config.LLM_BASE_URL,
                                                   "model": None, "wanted": None})
    llm, models, msg = None, [], ""
    try:
        models = PragyaLLM(cfg["provider"], cfg["url"]).list_models()
    except LLMError as e:
        msg = str(e)
    if models:
        free, _ = memory_gb()
        cfg["model"], note = choose_model(models, free, cfg.get("wanted"))
        llm = PragyaLLM(cfg["provider"], cfg["url"], cfg["model"])
        msg = note + " Temperature 0, fixed seed, thinking off."
    elif config.PUBLIC_DEMO and not config.LLM_EXPECTED:
        msg = "Not available in this online demo. All screens still work."
    elif config.PUBLIC_DEMO:
        msg = "The AI model is starting up. Refresh in a minute."
    else:
        msg = msg or "Start Ollama: open PowerShell and run  ollama serve"
    return llm, models, msg


LLM, MODELS, MODEL_MSG = connect_model()
def build_library() -> Library:
    """Demo circulars + every document added with Add data (+ an uploaded section mapping table)."""
    lib = Library.from_folder(embed_url=st.session_state["llm_cfg"]["url"]
                              if st.session_state["llm_cfg"]["provider"] == "ollama" else None)
    for d in st.session_state.get("up_docs", []):
        lib.add(parse_document(d["name"], d["text"], d["kind"], source="added"))
    for t in st.session_state.get("up_tables", []):
        if {"section_1961", "section_2025"} <= set(t["df"].columns):
            lib.section_map = t["df"]
    if LLM is not None:
        lib.try_embeddings()
    return lib


lib_key = (st.session_state.get("con_key"), LLM is not None)
if st.session_state.get("lib_key") != lib_key:
    st.session_state["library"], st.session_state["lib_key"] = build_library(), lib_key
LIB = st.session_state["library"]
REV = st.session_state.get("revenue")
AGENT = Lead(case, A, con, LLM, revenue=RevenueAgent(REV) if REV is not None else None, law=LawAgent(LIB),
             app_name=branding.APP_NAME)


# ====================================================================== Add data (one button, any file)
@st.dialog("Add data", width="large")
def add_data_dialog():
    st.caption("CSV, Excel, JSON, TXT, Word or PDF. Each file is recognised from its columns or text; correct the "
               "type if needed, then load. Files stay on this computer.")
    files = st.file_uploader("Files", accept_multiple_files=True, label_visibility="collapsed",
                             type=["csv", "xlsx", "xls", "xlsm", "json", "txt", "md", "docx", "pdf"])
    picks = []
    for i, f in enumerate(files or []):
        data = f.getvalue()
        p = uploads.read_file(f.name, data)
        with st.container(border=True):
            size = f"{p.size / 1024:,.0f} KB" if p.size >= 1024 else f"{p.size} bytes"
            st.markdown(f"**{ui.esc(f.name)}** · {size} · SHA-256 `{p.sha256[:12]}…`")
            if p.note:
                st.warning(p.note, icon=":material/warning:")
            kinds = uploads.TABLE_KINDS if p.is_table else uploads.TEXT_KINDS
            kind = st.selectbox(f"Recognised as: {p.kind}", kinds, index=kinds.index(p.kind) if p.kind in kinds else 0,
                                key=f"kind-{p.sha256}", help=p.reason)
            if p.is_table:
                rows = sum(len(d) for _, d in p.tables)
                with st.expander(f"Show records · {rows:,} row{'s' if rows != 1 else ''}"
                                 + (f" in {len(p.tables)} sheets" if len(p.tables) > 1 else "")):
                    st.dataframe(p.tables[0][1].head(20), hide_index=True, width="stretch")
            elif p.text:
                with st.expander(f"Show text · {len(p.text):,} characters"):
                    st.text(p.text[:2000])
        if p.is_table or p.text:
            picks.append((p, data, kind))
    if picks and st.button(f"Load {len(picks)} file{'s' if len(picks) != 1 else ''}", type="primary",
                           icon=":material/check:"):
        for p, data, kind in picks:
            uploads.save_upload(p, data, kind, tables=[uploads.TABLE_NAME.get(kind, "uploaded table")] if p.is_table else [])
        st.session_state["con_key"] = None            # rebuild the database with the new files
        st.toast(f"Loaded {len(picks)} file(s).", icon=":material/check_circle:")
        st.rerun()


# ====================================================================== officer profile (sidebar)
def profile_box():
    """Name, designation, office and domain. Decides the starter questions and the signature on every draft."""
    label = (f"{PROFILE.name} · {PROFILE.rank}" if PROFILE.is_set else "Officer profile")
    with st.expander(label, icon=":material/badge:", expanded=not PROFILE.is_set):
        with st.form("profile_form", border=False):
            ranks = list(prof.RANKS)
            name = st.text_input("Name", PROFILE.name, placeholder="e.g. A. Sharma")
            rank = st.selectbox("Designation", ranks, index=ranks.index(PROFILE.rank),
                                help="The full title (e.g. Income Tax Officer) is printed on drafts")
            office = st.text_input("Office", PROFILE.office, placeholder="e.g. O/o the ITO, Ward 2(1), Pune")
            posting = st.selectbox("Domain", prof.POSTINGS, index=prof.POSTINGS.index(PROFILE.posting),
                                   help="Your posting: decides the 4 starter questions on Ask")
            if st.form_submit_button("Save", width="stretch"):
                new = prof.Profile(name=name.strip(), rank=rank, posting=posting, office=office.strip())
                if not config.PUBLIC_DEMO:          # online demo: keep the profile for this visitor's session only
                    prof.save_profile(new)
                audit.log("approval", what="officer profile saved", rank=rank, posting=posting)
                st.session_state["officer_profile"] = new
                st.rerun()


# ====================================================================== sidebar
def sidebar():
    with st.sidebar:
        st.page_link(PAGES["ask"], label="Ask", icon=":material/forum:")
        st.page_link(PAGES["cases"], label="Cases", icon=":material/person_search:")
        st.page_link(PAGES["revenue"], label="Revenue", icon=":material/payments:")
        st.page_link(PAGES["library"], label="Library", icon=":material/menu_book:")
        st.page_link(PAGES["settings"], label="Settings", icon=":material/settings:")
        if not config.PUBLIC_DEMO and st.button("Add data", icon=":material/add:", width="stretch"):
            add_data_dialog()
        st.divider()
        profile_box()
        demo = "Demo data · " if case.name == config.DEMO_DIR.name else ""
        st.markdown(f"<div class='kd-side'>{demo}{S['holders']} people and firms</div>", unsafe_allow_html=True)
        ui.ai_status(LLM is not None, st.session_state["llm_cfg"]["model"] or config.LLM_MODEL, MODEL_MSG)


if config.PUBLIC_DEMO:
    st.markdown("<div style='font-size:.8rem;color:#8A8580;margin:0 0 1rem'>Online demo · synthetic data "
                "only, uploads off, no AI model. The real tool runs fully offline on the officer's computer.</div>",
                unsafe_allow_html=True)


# ====================================================================== Ask (home)
def md_lines(text: str) -> str:
    """Keep the answer's line breaks in Markdown: '- ' lines become a list, other lines stay on their own line."""
    import re
    text = re.sub(r"\n(?=- )", "\n\n", text)
    return re.sub(r"(?<!\n)\n(?!\n|- )", "  \n", text)


def show_quotes(r):
    """Exact quotes; a conflict is shown side by side with the later document marked."""
    if not r.quotes:
        return
    if r.extra.get("status") == "conflict":
        cols = st.columns(len(r.quotes))
        for c, q in zip(cols, r.quotes):
            with c:
                ui.quote_card(q)
        st.warning("These documents give different answers on the same point. An officer must decide which applies.",
                   icon=":material/balance:")
    else:
        for q in r.quotes:
            ui.quote_card(q)


def render_answer(ans):
    """3-6 plain lines, then a card / chart / table only if useful, then 'How I got this'."""
    chips = "".join(f"<span class='kd-chip'>{ui.esc(r.agent)}</span>" for r in ans.results)
    if chips:
        st.markdown(chips, unsafe_allow_html=True)
    text = ans.text
    if len(ans.results) == 1 and ans.results[0].quotes:          # law answer: one line, then the quote cards
        st_ = ans.results[0].extra.get("status")
        text = text.split("\n")[0] if st_ in ("conflict", "same") else f"{ans.results[0].quotes[0]['doc']} says:"
    if GENERAL_LABEL in text:                     # model's general knowledge: clearly labelled, not checked
        before, _, after = text.partition(GENERAL_LABEL)
        if before.strip():
            st.markdown(md_lines(before.strip()))
        st.caption(":material/info: " + GENERAL_LABEL)
        st.markdown(md_lines(after.strip()))
    else:
        st.markdown(md_lines(text))
    for r in ans.results:
        show_quotes(r)
        if r.fig is not None:
            st.plotly_chart(r.fig, width="stretch", config=PC)
        if r.html:
            embed_html(r.html, 520)
        if r.records is not None and len(r.records):
            with st.expander(f"Show records · {r.agent} · {len(r.records):,}"):
                d = tx_view(r.records) if "txn_id" in r.records.columns else r.records
                cfg = TX_CONFIG if "txn_id" in r.records.columns else {
                    c: st.column_config.NumberColumn(str(c).replace("_", " "), format="localized")
                    for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not str(c).endswith("_id")}
                st.dataframe(d, hide_index=True, width="stretch", height=min(380, 40 + 35 * len(d)), column_config=cfg)
    with st.expander("How I got this"):
        if ans.results:
            st.dataframe(pd.DataFrame([{"agent": r.agent, "tool": r.tool, "checked": r.checked}
                                       for r in ans.results]), hide_index=True, width="stretch")
        lines = [ln for ln in ans.trace if not ln.startswith("Checker on the answer shown")]
        st.markdown("\n".join(("    - " if ln.startswith("   ") else "- ") + ln.strip().replace("*", "\\*")
                               for ln in lines))
        for r in ans.results:
            if r.sql:
                st.code(r.sql, language="sql")
        if ans.check is not None:
            (st.success if ans.check.ok else st.error)("Checker: " + ans.check.summary(), icon=":material/verified:")
        if "general" in ans.source:
            st.caption("The general-knowledge part was written by the local model and is not checked against your data.")
        st.caption(f"Written by {'the local model' if ans.source in ('model', 'general') else 'code from the tool output'} · "
                   f"{ans.seconds:.1f} s. Numbers come from code run on the data, never from the AI's memory.")


def page_ask():
    first = PROFILE.name.split()[0] if PROFILE.is_set else ""
    shown = PROFILE.name.strip() if first.endswith(".") or len(first) <= 2 else first      # "A. Sharma", not "A."
    hello = f"Good to see you, {ui.esc(shown)}. " if PROFILE.is_set else ""
    ui.header(f"Ask {branding.APP_NAME}", f"{hello}Cases, revenue, circulars or any loaded table, in plain English.")
    top = A.risk.iloc[0]["name"] if len(A.risk) else ""
    trail = case.holder_name(A.chains[0]["start_holder"]) if A.chains else ""
    starters = prof.starter_questions(PROFILE, top=top, trail=trail)
    clicked = st.pills("Try", starters, selection_mode="single", key=f"pill-{len(st.session_state['chat'])}",
                       label_visibility="collapsed")
    for item in st.session_state["chat"]:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(item.question)
        with st.chat_message("assistant", avatar=BOT):
            render_answer(item)
    q = st.chat_input("Ask about cases, revenue, circulars or any loaded table") or clicked
    if q:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(q)
        with st.chat_message("assistant", avatar=BOT):
            with st.spinner("Checking the data and the numbers..."):
                ans = AGENT.ask(q)
            render_answer(ans)
        st.session_state["chat"].append(ans)
        st.rerun()


# ====================================================================== Cases
def page_cases():
    if st.session_state.get("case_open"):
        return case_detail(st.session_state["case_open"])
    r = A.risk
    n_high, n_med = int((r.risk_level == "HIGH").sum()), int((r.risk_level == "MEDIUM").sum())
    ui.header("Cases", f"{S['date_from']:%d %b %Y} to {S['date_to']:%d %b %Y} · checked on this computer · "
                       + ("income-tax checks + GST signals" if A.gst_on else "income-tax checks (GST add-on off)"))
    view = st.segmented_control("View", ["Review queue", "Self-test on demo data"], default="Review queue",
                                label_visibility="collapsed") if case.ground_truth else "Review queue"
    if view == "Self-test on demo data":
        return self_test_view()
    ui.kpis([("People and firms", f"{S['holders']}", f"{S['persons']} people · {S['firms']} firms"),
             ("Transactions", f"{S['transactions']:,}", f"{S['accounts']} bank accounts"),
             ("Money in", inr_short(S["total_credits"]), "total credits"),
             ("Need review", f"{n_high + n_med}", f"{n_high} high · {n_med} medium", "red")])
    ui.section("Review queue")
    qv = st.session_state.setdefault("queue_v", 0)
    flagged = r[r.risk_level != "LOW"]
    st.caption(f"All {len(flagged)} of {len(r)} that need review, highest risk first. Click a row to open the case.")
    ev = queue_table(flagged, f"queue-{qv}")
    if ev and ev.selection.rows:
        st.session_state["case_open"] = flagged.iloc[ev.selection.rows[0]]["holder_id"]
        st.session_state["queue_v"] = qv + 1
        st.rerun()
    a, b = st.columns(2, gap="medium")
    with a:
        ui.section("Risk split")
        st.plotly_chart(charts.risk_donut(r), width="stretch", config=PC)
    with b:
        ui.section("What was found")
        if len(A.findings):
            st.plotly_chart(charts.pattern_counts(A.findings, LABEL), width="stretch", config=PC)
    how_it_works()


def self_test_view():
    """Honest results of the risk rules on the synthetic demo data (planted, borderline, decoys, false alarms)."""
    t = risk_self_test(case, A)
    ui.kpis([("Planted cases caught", f"{t['caught']} of {t['planted']}", "every planted case should be caught",
              "" if t["caught"] == t["planted"] else "red"),
             ("Borderline caught", f"{t['borderline_caught']} of {t['borderline_total']}", "just over a threshold",
              "" if t["borderline_caught"] == t["borderline_total"] else "red"),
             ("Decoys wrongly flagged", f"{t['decoys_flagged']} of {t['decoys_total']}", "look suspicious, under every threshold",
              "red" if t["decoys_flagged"] else ""),
             ("Other false alarms", f"{t['false_alarms']}", "flagged but not planted", "amber" if t["false_alarms"] else "")])
    st.caption("The demo data is synthetic and these cases were planted on purpose, so this shows the rules work as "
               "designed. It is not proof of real-world accuracy: real cases are messier and every flag needs an "
               "officer's review." + ("" if t["gst_on"] else " GST signals are not judged (add-on off)."))
    ui.section("Borderline cases")
    st.dataframe(t["borderline"], hide_index=True, width="stretch")
    ui.section("Decoys")
    st.dataframe(t["decoys"], hide_index=True, width="stretch")
    if t["false_alarms"]:
        ui.section("Other false alarms")
        st.dataframe(t["false_alarm_table"], hide_index=True, width="stretch")
    with st.expander("Show records · result for every red-flag rule"):
        st.dataframe(t["per_pattern"], hide_index=True, width="stretch")


def case_detail(hid):
    ids = A.risk.holder_id.tolist()
    if hid not in ids:
        st.session_state.pop("case_open", None)
        st.rerun()
    c1, c2 = st.columns([1, 3])
    if c1.button("Back to queue", icon=":material/arrow_back:"):
        st.session_state.pop("case_open", None)
        st.rerun()
    pick = c2.selectbox("Case", ids, index=ids.index(hid), format_func=holder_label, label_visibility="collapsed")
    if pick != hid:
        st.session_state["case_open"] = pick
        st.rerun()
    r = RISK.loc[hid]
    h = case.holders.set_index("holder_id").loc[hid]
    is_firm = h["holder_type"] == "FIRM"
    bits = ["Firm" if is_firm else f"Person · {h['occupation']}", h["city"] if pd.notna(h["city"]) else "",
            f"PAN {h['pan']}" if pd.notna(h["pan"]) and h["pan"] else ""]
    st.markdown(f"<div class='kd-profile'><div class='n'>{ui.esc(r['name'])}{ui.badge(r['risk_level'])}</div>"
                f"<div class='sub'>{ui.esc(' · '.join(b for b in bits if b))}</div></div>", unsafe_allow_html=True)

    names = ["Summary", "Evidence", "Money trail", "Links"]
    show_gst = A.gst_on and gst.has_gst(case) and (
        hid in set(case.gst_invoices.seller_firm_id) | set(case.gst_invoices.buyer_firm_id))
    if show_gst:
        names.append("GST")
    names.append("Report")
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Summary"]:
        summary_tab(hid, r, h, is_firm)
    with tabs["Evidence"]:
        evidence_tab(hid)
    with tabs["Money trail"]:
        money_trail_tab(hid, r)
    with tabs["Links"]:
        links_tab(hid)
    if show_gst:
        with tabs["GST"]:
            st.caption("GST signals for income-tax risk (optional add-on).")
            gst_tab(hid)
    with tabs["Report"]:
        report_tab(hid)


def summary_tab(hid, r, h, is_firm):
    if not r["filed"]:
        declared = ("Declared income", "Not filed", "no ITR for the year", "red")
    elif is_firm or h["gross_receipts"] > 0:
        declared = ("Declared turnover", inr_short(h["gross_receipts"]), "as per ITR")
    else:
        declared = ("Declared income", inr_short(r["declared_income"]), "as per ITR")
    ratio = r["total_credits"] / max(h["gross_receipts"] if h["gross_receipts"] > 0 else r["declared_income"], 1)
    tone = {"HIGH": "red", "MEDIUM": "amber"}.get(r["risk_level"], "")
    n_flags = len(r["breakdown"])
    ui.kpis([("Risk score", f"{r['risk_score']:.0f} / 100",
              f"{n_flags} red flag{'s' if n_flags != 1 else ''} · {r['points']:.0f} points", tone),
             declared,
             ("Money in", inr_short(r["total_credits"]), f"{ratio:,.1f}× declared" if r["filed"] else "total credits"),
             ("Cash deposits", inr_short(r["cash_deposits"]), "total in the year")])
    ui.score_breakdown(r["breakdown"], r["points"], r["risk_score"], r["floored"], config.RISK_HIGH,
                       config.RISK_MEDIUM)
    f = A.findings[A.findings.holder_id == hid]
    if f.empty:
        st.success("No suspicious pattern found for this person or firm.")
    for fr in f.itertuples(index=False):
        ui.finding_card(LABEL.get(fr.pattern, fr.pattern), fr.summary, fr.severity,
                        f"+{config.RISK_WEIGHTS.get(fr.pattern, 10) * fr.severity:.0f} points")
    if r["outlier"]:
        st.caption(f"Hint (not counted in the score): a machine-learning outlier model finds this behaviour unusual "
                   f"compared with others (score {r['anomaly']:.2f}, top 10%). Only the red flags above count.")
    key = f"explain-{case_key}-{hid}"
    if st.button("Explain in plain English", icon=":material/auto_awesome:", type="primary", disabled=LLM is None,
                 help=None if LLM else "Needs the local AI model (see Settings)"):
        row = r.copy()
        row["holder_id"] = hid
        fb = risk.explain_holder_text(row)
        with st.spinner("Writing the explanation..."):
            st.session_state[key] = llm_explain(LLM, f"Explain to the officer why {r['name']} was flagged.", fb, fb)[0]
    if key in st.session_state:
        ui.note(st.session_state[key])
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


def evidence_tab(hid):
    f = A.findings[A.findings.holder_id == hid]
    if f.empty:
        st.caption("No red flags, so there is no evidence to show.")
    for fr in f.itertuples(index=False):
        n = len(fr.evidence)
        with st.expander(f"{LABEL.get(fr.pattern, fr.pattern)} · {n} record{'s' if n != 1 else ''}"):
            st.write(fr.summary)
            if n:
                evidence_tables(fr.evidence)
    tx = case.transactions[case.transactions.holder_id == hid]
    with st.expander(f"Show records · all {len(tx):,} transactions"):
        modes = sorted(tx["mode"].unique())
        pick = st.pills("Mode", modes, selection_mode="multi", default=modes, label_visibility="collapsed")
        st.dataframe(tx_view(tx[tx["mode"].isin(pick or modes)]), hide_index=True, width="stretch",
                     column_config=TX_CONFIG, height=420)


def money_trail_tab(hid, r):
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
        st.caption("Teal = inside this case · red = left the case (outside party) · red links = money returning "
                   "to the start.")
    else:
        st.info("No outgoing transfers above the minimum.")


def report_tab(hid):
    st.write("A printable DRAFT report for this case, with findings, evidence and your signature block.")
    notes = st.text_area("Your notes (optional)", height=90, key=f"notes-{hid}")
    if st.button("Make draft report", icon=":material/description:", key=f"rep-{hid}"):
        st.session_state[f"report-{hid}"] = report.build_report(case, A, [hid], notes,
                                                                signature=prof.signature_block(PROFILE))
        audit.log("approval", what="draft report made", holder=hid, officer=PROFILE.name)
    html_doc = st.session_state.get(f"report-{hid}")
    if html_doc:
        st.download_button("Download draft report", html_doc, file_name=f"DRAFT_report_{hid}.html",
                           mime="text/html", icon=":material/download:")
        embed_html(html_doc, 700)


def links_tab(hid):
    """People and firms connected to this holder by shared phone / address / director and by money."""
    group = next((c for c in A.clusters if hid in c["members"]), None)
    if group:
        txt = (f"Part of linked group {group['cluster_id']}: {len(group['members'])} entities "
               f"({len(group['firms'])} firms), sharing "
               + ", ".join(f"{v} {k}" for k, v in group["shared_links"].items())
               + f". Money moved inside the group: {inr(group['internal_flow'])}.")
        (st.warning if group["suspicious"] else st.info)(txt, icon=":material/hub:")
    c1, c2 = st.columns([3, 1])
    view = c1.segmented_control("Show", ["Direct connections", "Whole linked group"] if group else
                                ["Direct connections"], default="Direct connections", key=f"lv-{hid}",
                                label_visibility="collapsed")
    if view == "Whole linked group" and group:
        members, hops = group["members"], 1
    else:
        members, hops = [hid], c2.slider("Degrees", 1, 3, 1, key=f"hops-{hid}")
    G = link_finder.ego_network(case, members, hops=hops)
    if G.number_of_edges() == 0:
        st.caption("No shared phone, address, director or large transfers with other holders.")
    else:
        embed_html(link_finder.to_pyvis_html(case, G, set(members), RISK["risk_score"].to_dict(), height="520px"), 540)
        st.caption("● person  ■ firm  ·  red = high risk, amber = medium  ·  dashed = shared phone / address / "
                   "director  ·  arrows = money (thicker = more). Drag to rearrange, hover for details.")
    if A.clusters:
        with st.expander(f"All linked groups in the case ({len(A.clusters)})"):
            st.dataframe(pd.DataFrame([{
                "group": c["cluster_id"], "entities": len(c["members"]), "firms": len(c["firms"]),
                "suspicious": "Yes" if c["suspicious"] else "", "money inside": inr_short(c["internal_flow"]),
                "shared links": ", ".join(f"{v} {k}" for k, v in c["shared_links"].items()),
                "members": ", ".join(case.holder_name(m) for m in c["members"][:6]) + (" …" if len(c["members"]) > 6 else "")}
                for c in A.clusters]), hide_index=True, width="stretch")


def gst_tab(hid):
    """GST add-on: this firm's invoices, loops it is part of, and monthly returns."""
    loops = [c for c in (A.gst_cycles or []) if hid in c["firms"]]
    for cy in loops:
        route = " → ".join(case.holder_name(x) for x in cy["route"])
        ui.finding_card("Circular invoicing loop: " + route,
                        f"{cy['invoices']} invoices · {inr(cy['total_value'])} went round the loop · ITC generated "
                        f"{inr(cy['itc_generated'])} · {cy['no_eway_share']:.0%} without e-way bill", 0.9,
                        f"{len(cy['firms'])} firms")
    inv = case.gst_invoices
    sold, bought = inv[inv.seller_firm_id == hid], inv[inv.buyer_firm_id == hid]
    ui.kpis([("Sales invoices", f"{len(sold):,}", inr_short(sold.taxable_value.sum()) + " taxable"),
             ("Purchase invoices", f"{len(bought):,}", inr_short(bought.taxable_value.sum()) + " taxable"),
             ("ITC on purchases", inr_short(bought.gst_amount.sum()), "credit claimed from suppliers"),
             ("No e-way bill", f"{sold.eway_bill_no.isna().mean():.0%}" if len(sold) else "–", "of sales invoices")])
    a, b = st.columns(2, gap="medium")
    with a:
        ui.section("Who it trades with")
        embed_html(gst.gst_network_html(case, [hid], RISK["risk_score"].to_dict(), height="420px"), 440)
        st.caption("Arrow = invoices seller → buyer, thicker = more value · red arrow = mostly no e-way bill.")
    with b:
        ui.section("Monthly GST returns")
        rr = case.gst_returns[case.gst_returns.firm_id == hid].sort_values("period") \
            if case.gst_returns is not None else pd.DataFrame()
        if len(rr):
            st.plotly_chart(charts.gst_monthly(rr), width="stretch", config=PC)
            tot = rr[["output_tax", "itc_claimed", "tax_paid_cash"]].sum()
            st.caption(f"Year: output tax {inr(tot.output_tax)} · ITC {inr(tot.itc_claimed)} · paid in cash "
                       f"{inr(tot.tax_paid_cash)} ({tot.tax_paid_cash / max(tot.output_tax, 1):.1%} of liability)")
        else:
            st.caption("No GST returns for this firm.")


def how_it_works():
    """Short method note plus the accuracy check on the synthetic demo case."""
    ui.section("How it works")
    w = config.RISK_WEIGHTS
    checks = [k for k in w if A.gst_on or k not in risk.GST_PATTERNS]
    st.markdown(
        "1. **Load** bank transactions and income-tax returns (demo data, uploaded statements or a case folder).\n"
        f"2. **Check** every person and firm against {len(checks)} red-flag rules: cash structuring, round-tripping, "
        "pass-through accounts, income mismatch, non-filers, large round-figure cash and linked shell firms"
        + (", plus 4 GST signals for income-tax risk (add-on on)." if A.gst_on else ". GST checks are an optional add-on (Settings > Data).") + "\n"
        "3. **Score**: each red flag adds up to its maximum points, scaled by how strong the evidence is; the total is "
        f"capped at 100. High ≥ {config.RISK_HIGH}, Medium ≥ {config.RISK_MEDIUM}. Any red flag means at least Medium.\n"
        "4. **Ask**: the agents pick a tool, run it on the data and explain the result. It never makes up "
        "numbers, and everything runs on this computer.")
    with st.expander("Maximum points per red flag"):
        st.dataframe(pd.DataFrame([{"red flag": LABEL.get(k, k), "max points": w[k]} for k in checks]),
                     hide_index=True, width="stretch")
    st.caption("See **Self-test on demo data** above for how the rules do on the planted demo cases.")


def data_settings():
    """Settings > Data: files added with "Add data", the audit log, the demo case, bank statements as a new
    case, a case folder and the GST add-on."""
    ui.section("Files added")
    saved = uploads.read_manifest()
    if not saved:
        st.caption("No files added yet. Use **Add data** in the sidebar.")
    loaded = {t["file"]: t for t in st.session_state.get("up_tables", [])}
    for e in saved:
        c1, c2 = st.columns([5, 1])
        t = loaded.get(e["name"])
        c1.markdown(f"**{ui.esc(e['name'])}** · {ui.esc(e['kind'])}"
                    + (f" · table `{t['table']}` · {t['rows']:,} rows" if t else "")
                    + f"  \nSHA-256 `{e['sha256']}`")
        if c2.button("Remove", key=f"rm-{e['sha256']}", icon=":material/delete:"):
            uploads.remove_upload(e["sha256"])
            st.session_state["con_key"] = None
            st.rerun()
    with st.expander("Audit log · last 100 events"):
        ev = audit.read(last=100)
        if ev:
            st.dataframe(pd.DataFrame(ev[::-1]).astype(str), hide_index=True, width="stretch", height=320)
        else:
            st.caption("Nothing logged yet.")
    ui.section("Case data")
    st.caption("Synthetic demo case only in the online demo." if config.PUBLIC_DEMO else
               "Use the synthetic demo case, upload bank statements, or point to a case folder.")
    with st.container(border=True):
        on = st.toggle("GST checks (add-on)", value=GST_ON,
                       help="Circular invoicing, ITC set-off without cash tax, missing e-way bills and ITC from "
                            "flagged suppliers. Needs gst_invoices.csv and gst_returns.csv.")
        st.caption("GST is handled by the GST department (CBIC), so these checks are optional. Leave off to focus "
                   "on income-tax investigation." + ("" if gst.has_gst(case) else " This case has no GST data."))
        if on != GST_ON:
            st.session_state["gst_on"] = on
            if "uploaded_case" in st.session_state:
                up = st.session_state["uploaded_case"][0]
                st.session_state["uploaded_case"] = (up, risk.analyze(up, include_gst=on))
            st.rerun()
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
        if items and st.button("Check these statements"):
            from kar_drishti.statements import build_case_from_statements
            try:
                with st.spinner("Reading the statements..."):
                    up = build_case_from_statements(items, "Uploaded statements")
                    st.session_state["uploaded_case"] = (up, risk.analyze(up, include_gst=GST_ON))
                st.switch_page(PAGES["cases"])
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




# ====================================================================== Revenue and Library
def page_revenue():
    if REV is None:
        ui.header("Revenue", "Target, collections, gap and year-end projection, with a ranked recovery list.")
        st.info("No targets and collections are loaded. Add them with **Add data** in the sidebar.",
                icon=":material/payments:")
        return
    ui.header("Revenue", f"FY {REV.fy} · collections as on {pd.Timestamp(REV.as_on):%d %b %Y} · {REV.label}")
    for n in REV.notes:
        st.caption(n)
    where, c = {}, st.columns(3)
    t = REV.targets
    for i, lvl in enumerate(LEVELS):
        opts = sorted(t[(t[list(where)] == pd.Series(where)).all(axis=1)][lvl].unique()) if where else sorted(t[lvl].unique())
        pick = c[i].selectbox(lvl.title(), ["All"] + list(opts), key=f"rev-{lvl}")
        if pick == "All":
            break
        where[lvl] = pick
    s = REV.summary(where)
    behind = s["projected"] < s["target"]
    ui.kpis([("Target", inr_short(s["target"]), f"FY {s['fy']}"),
             ("Collected", inr_short(s["collected"]), f"{s['achieved_pct']}% of target"),
             ("Gap", inr_short(s["gap"]), f"needs {inr_short(s['monthly_pace_needed'])} a month to 31 March",
              "amber" if s["gap"] > 0 else ""),
             ("Projected", inr_short(s["projected"]),
              f"{inr_short(abs(s['projected'] - s['target']))} {'short of' if behind else 'above'} target",
              "red" if behind else "teal")])
    st.plotly_chart(RevenueAgent(REV).chart(where), width="stretch", config=PC)
    problems = REV.consistency()
    if problems:
        st.error("Totals do not add up: " + "; ".join(problems), icon=":material/error:")
    else:
        st.caption("✓ Totals add up at every level: range > charge > region > all, and months > year. "
                   "Projection = last year's monthly pattern scaled by this year's run-rate "
                   f"(plain run-rate: {inr_short(s['run_rate_projected'])}).")
    nxt = next((l for l in LEVELS if l not in where), None)
    with st.expander(f"Show records · by {nxt or 'range'} and by month"):
        g = REV.rollup(nxt or "range", where)
        st.dataframe(g.drop(columns=["ly_same_period", "ly_full_year", "run_rate_projected", "projected_vs_target"]),
                     hide_index=True, width="stretch", column_config={
                         c: st.column_config.NumberColumn(c.replace("_", " "), format="localized")
                         for c in ["target", "collected", "gap", "projected", "monthly_pace_needed"]})
        st.dataframe(REV.monthly(where), hide_index=True, width="stretch")
    ui.section("Recovery list")
    st.caption("Largest undisputed demands with no stay first, then the rest by size. The officer decides the action.")
    rl = REV.recovery_list(where)
    if rl.empty:
        st.caption("No outstanding demands.")
    else:
        top = rl.head(10).assign(outstanding=rl.head(10).outstanding.map(inr_short))
        st.dataframe(top[["rank", "taxpayer_name", "range", "outstanding", "reason"]], hide_index=True,
                     width="stretch", column_config={"rank": st.column_config.NumberColumn("#", width="small"),
                                                     "taxpayer_name": "Taxpayer", "reason": st.column_config.TextColumn(
                                                         "Why this rank", width="large")})
        with st.expander(f"Show records · all {len(rl)} outstanding demands · ageing"):
            st.dataframe(rl, hide_index=True, width="stretch")
            ag = REV.ageing(where)
            st.dataframe(ag.assign(outstanding=ag.outstanding.map(inr_short),
                                   disputed_or_stayed=ag.disputed_or_stayed.map(inr_short)),
                         hide_index=True, width="stretch")


def page_library():
    ui.header("Library", "Circulars, office orders and Acts. Answers quote the exact sentences, with document "
                         "number and date.")
    st.caption(LIB.embed_status + " Add your own documents with **Add data**.")
    rows = [{"document": d.title, "number": d.number, "date": d.date, "type": d.kind, "passages": len(d.passages),
             "source": "fictitious demo" if d.source == "demo" else "added"} for d in LIB.docs.values()]
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("date"), hide_index=True, width="stretch")
    else:
        st.info("No documents loaded yet.", icon=":material/menu_book:")
    smap = LIB.section_map
    st.caption("Section mapping 1961 ↔ 2025 Act: " + (f"{len(smap)} rows loaded." if smap is not None and len(smap)
               else "no rows loaded (template only). Pragya never guesses a section."))
    ui.section("Test search")
    q = st.text_input("Search the library", placeholder="e.g. How many casual leave days can be carried forward?",
                      label_visibility="collapsed")
    if q:
        res = LawAgent(LIB).run(q)
        st.markdown(res.template.split("\n")[0] if res.extra.get("status") in ("conflict", "same") else res.template)
        show_quotes(res)
        hits = LIB.search(q, 6)
        with st.expander(f"Show matching passages · {len(hits)}"):
            for score, p in hits:
                d = LIB.docs[p.doc_id]
                st.markdown(f"`{score:.2f}` · No. {d.number} · {ui.esc(p.text)}")


# ====================================================================== Settings
def page_settings():
    ui.header("Settings", "The local AI model, the data in use and your signature. Everything stays on this computer.")
    t_model, t_data, t_sig = st.tabs(["Model", "Data", "Signature"])
    with t_sig:
        st.caption("Set your name, designation, office and domain in the sidebar. This block ends every draft.")
        st.code(prof.signature_block(PROFILE), language=None)
    with t_model:
        model_settings()
    with t_data:
        data_settings()


def model_settings():
    cfg = st.session_state["llm_cfg"]
    free, total = memory_gb()
    ui.kpis([("Model in use", cfg["model"] or "none", "connected" if LLM else "not connected", "" if LLM else "amber"),
             ("Free memory", f"{free:.1f} GB" if free is not None else "–",
              f"of {total:.1f} GB" if total is not None else "could not read"),
             ("Server", "Ollama" if cfg["provider"] == "ollama" else "OpenAI-compatible", cfg["url"])])
    st.caption(MODEL_MSG)
    prov = st.selectbox("Runtime", ["ollama", "openai"], index=0 if cfg["provider"] == "ollama" else 1,
                        format_func=lambda p: "Ollama" if p == "ollama" else "LM Studio / llama.cpp (OpenAI-compatible)")
    url = st.text_input("Address", cfg["url"])
    mdl = st.selectbox("Model", MODELS, index=MODELS.index(cfg["model"]) if cfg["model"] in MODELS else 0) \
        if MODELS else None
    if (prov, url) != (cfg["provider"], cfg["url"]) or (mdl and mdl != cfg["model"]):
        cfg.update(provider=prov, url=url, wanted=mdl)
        st.rerun()
    st.caption("Default qwen3.5:4b (thinking off), fallback qwen2.5:3b; qwen3.5:9b only with 8 GB or more free. "
               "Only addresses on this computer are allowed (a Department server needs the admin setting), and no "
               "model is ever downloaded automatically. "
               "Installed models: " + (", ".join(MODELS) if MODELS else "none found") + ".")




# ====================================================================== navigation
PAGES = {
    "ask": st.Page(page_ask, title="Ask", icon=":material/forum:", url_path="ask", default=True),
    "cases": st.Page(page_cases, title="Cases", icon=":material/person_search:", url_path="cases"),
    "revenue": st.Page(page_revenue, title="Revenue", icon=":material/payments:", url_path="revenue"),
    "library": st.Page(page_library, title="Library", icon=":material/menu_book:", url_path="library"),
    "settings": st.Page(page_settings, title="Settings", icon=":material/settings:", url_path="settings"),
}
nav = st.navigation(list(PAGES.values()), position="hidden")
sidebar()
nav.run()

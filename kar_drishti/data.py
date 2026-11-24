"""
Case data loading and the local DuckDB database used by the agents.

A "case" is a folder of CSV files:
    persons.csv, firms.csv, accounts.csv, transactions.csv, itr_filings.csv
(the demo generator writes exactly these). Officers can also upload raw bank
statements, which are normalised into the same structure (see statements.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

TABLES = ["persons", "firms", "accounts", "transactions", "itr_filings"]
OPTIONAL_TABLES = ["gst_invoices", "gst_returns"]

# Plain-language description of every table/column. Used in the LLM prompt for
# the SQL agent so the model knows exactly what it can query.
SCHEMA_DOC = {
    "holders": ("One row per person or firm in the case (combined view).", {
        "holder_id": "ID like P0001 (person) or F0001 (firm)",
        "holder_type": "'PERSON' or 'FIRM'",
        "name": "name of person or firm",
        "pan": "PAN (fake in demo data)",
        "city": "city",
        "occupation": "Salaried / Business / Professional / Retired / Self-employed / Firm",
        "phone": "phone number", "address": "address",
        "registration_date": "firm registration date (NULL for persons)",
        "director_person_id": "for firms: holder_id of the director",
        "filed": "TRUE if income-tax return filed for the tax year",
        "declared_income": "income declared in ITR (rupees)",
        "gross_receipts": "turnover / gross receipts declared in ITR (rupees)",
        "tax_paid": "tax paid (rupees)"}),
    "accounts": ("Bank accounts.", {
        "account_id": "ID like AC00001", "holder_id": "owner (joins holders.holder_id)",
        "holder_type": "'PERSON' or 'FIRM'", "bank": "bank name", "branch": "branch",
        "account_type": "'Savings' or 'Current'", "open_date": "account opening date"}),
    "transactions": ("Bank transactions for FY 2025-26. One row per entry in an account.", {
        "txn_id": "transaction ID", "account_id": "account of this entry (joins accounts)",
        "holder_id": "owner of account_id (joins holders)",
        "txn_date": "DATE of transaction",
        "mode": "CASH, ATM, UPI, NEFT, RTGS, IMPS, CHEQUE",
        "direction": "'CR' = money in (credit), 'DR' = money out (debit)",
        "amount": "amount in rupees (always positive)",
        "counterparty_account_id": "other account if it is inside this case, else NULL",
        "counterparty_holder_id": "owner of the counterparty account if inside the case, else NULL",
        "counterparty_name": "name of the other party as shown in the statement",
        "narration": "statement narration text", "ref_no": "bank reference"}),
    "gst_invoices": ("GST B2B invoices between firms in the case (GSTR-1 style).", {
        "invoice_id": "invoice ID", "seller_firm_id": "seller (joins holders.holder_id)",
        "seller_gstin": "seller GSTIN", "buyer_firm_id": "buyer (joins holders.holder_id)",
        "buyer_gstin": "buyer GSTIN", "invoice_date": "DATE of invoice", "hsn": "HSN code and goods",
        "taxable_value": "taxable value (rupees)", "gst_amount": "GST charged = ITC available to buyer (rupees)",
        "eway_bill_no": "e-way bill number; NULL means no e-way bill"}),
    "gst_returns": ("Monthly GST return summary per firm (GSTR-3B style).", {
        "firm_id": "joins holders.holder_id", "gstin": "GSTIN", "period": "month as 'YYYY-MM'",
        "outward_taxable_value": "outward supplies (rupees)", "output_tax": "output GST liability",
        "itc_claimed": "input tax credit claimed", "tax_paid_cash": "GST paid in cash"}),
    "risk_scores": ("Risk score per holder computed by Kar Drishti (0-100).", {
        "holder_id": "joins holders", "name": "name", "holder_type": "PERSON/FIRM",
        "risk_score": "0-100, higher = riskier", "risk_level": "HIGH / MEDIUM / LOW",
        "flags": "comma-separated list of detected patterns",
        "total_credits": "total money in (rupees)", "total_debits": "total money out",
        "cash_deposits": "total cash deposited", "declared_income": "declared income"}),
}


@dataclass
class Case:
    name: str
    persons: pd.DataFrame
    firms: pd.DataFrame
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    itr_filings: pd.DataFrame
    holders: pd.DataFrame = field(default=None)
    gst_invoices: pd.DataFrame | None = None
    gst_returns: pd.DataFrame | None = None
    ground_truth: dict | None = None

    def __post_init__(self):
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self):
        tx = self.transactions.copy()
        tx["txn_date"] = pd.to_datetime(tx["txn_date"])
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        for c in ["counterparty_account_id", "counterparty_name", "narration", "ref_no", "mode"]:
            if c not in tx:
                tx[c] = None
        acc_owner = self.accounts.set_index("account_id")["holder_id"]
        tx["holder_id"] = tx["account_id"].map(acc_owner)
        tx["counterparty_holder_id"] = tx["counterparty_account_id"].map(acc_owner)
        if "counterparty_holder_id_hint" in tx:   # from uploaded statements
            tx["counterparty_holder_id"] = tx["counterparty_holder_id"].fillna(tx["counterparty_holder_id_hint"])
            tx = tx.drop(columns=["counterparty_holder_id_hint"])
        tx["mode"] = tx["mode"].fillna("OTHER").str.upper()
        self.transactions = tx

        p = self.persons.copy()
        p["holder_type"] = "PERSON"
        p = p.rename(columns={"person_id": "holder_id"})
        f = self.firms.copy()
        if len(f):
            f["holder_type"] = "FIRM"
            f["occupation"] = "Firm"
            f = f.rename(columns={"firm_id": "holder_id"})
        cols = ["holder_id", "holder_type", "name", "pan", "city", "occupation", "phone",
                "address", "registration_date", "director_person_id"]
        h = pd.concat([p.reindex(columns=cols), f.reindex(columns=cols)], ignore_index=True)
        itr = self.itr_filings.reindex(columns=["holder_id", "filed", "declared_income",
                                                "gross_receipts", "tax_paid"])
        h = h.merge(itr, on="holder_id", how="left")
        h["filed"] = h["filed"].fillna(False).astype(bool)
        for c in ["declared_income", "gross_receipts", "tax_paid"]:
            h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0)
        h["registration_date"] = pd.to_datetime(h["registration_date"], errors="coerce")
        self.holders = h
        if self.gst_invoices is not None and len(self.gst_invoices):
            gi = self.gst_invoices.copy()
            gi["invoice_date"] = pd.to_datetime(gi["invoice_date"])
            gi["eway_bill_no"] = gi["eway_bill_no"].replace("", None) if "eway_bill_no" in gi else None
            self.gst_invoices = gi

    # ------------------------------------------------------------------
    def holder_name(self, holder_id) -> str:
        m = self.holders.loc[self.holders.holder_id == holder_id, "name"]
        return m.iloc[0] if len(m) else str(holder_id)

    def summary(self) -> dict:
        tx = self.transactions
        return {"holders": len(self.holders),
                "persons": int((self.holders.holder_type == "PERSON").sum()),
                "firms": int((self.holders.holder_type == "FIRM").sum()),
                "accounts": len(self.accounts), "transactions": len(tx),
                "total_credits": float(tx.loc[tx.direction == "CR", "amount"].sum()),
                "total_debits": float(tx.loc[tx.direction == "DR", "amount"].sum()),
                "date_from": tx.txn_date.min(), "date_to": tx.txn_date.max()}


def load_case(folder: Path | str, name: str | None = None) -> Case:
    folder = Path(folder)
    dfs = {}
    for t in TABLES:
        fp = folder / f"{t}.csv"
        dfs[t] = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    for t in OPTIONAL_TABLES:
        fp = folder / f"{t}.csv"
        dfs[t] = pd.read_csv(fp) if fp.exists() else None
    truth = None
    gt = folder / "ground_truth.json"
    if gt.exists():
        import json
        truth = json.loads(gt.read_text())
    return Case(name=name or folder.name, ground_truth=truth, **dfs)


def build_duckdb(case: Case, risk_scores: pd.DataFrame | None = None,
                 include_gst: bool = True) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the case tables. The SQL agent gets a locked-down
    connection: no file or network access and no configuration changes."""
    con = duckdb.connect(database=":memory:")
    holders = case.holders.copy()
    tx = case.transactions.copy()
    tx["txn_date"] = tx["txn_date"].dt.date
    con.register("_holders", holders)
    con.register("_accounts", case.accounts)
    con.register("_tx", tx)
    con.execute("CREATE TABLE holders AS SELECT * FROM _holders")
    con.execute("CREATE TABLE accounts AS SELECT * FROM _accounts")
    con.execute("CREATE TABLE transactions AS SELECT * FROM _tx")
    if include_gst and case.gst_invoices is not None and len(case.gst_invoices):
        gi = case.gst_invoices.copy()
        gi["invoice_date"] = pd.to_datetime(gi["invoice_date"]).dt.date
        con.register("_gi", gi)
        con.execute("CREATE TABLE gst_invoices AS SELECT * FROM _gi")
    if include_gst and case.gst_returns is not None and len(case.gst_returns):
        con.register("_gr", case.gst_returns)
        con.execute("CREATE TABLE gst_returns AS SELECT * FROM _gr")
    if risk_scores is not None and len(risk_scores):
        rs = risk_scores.copy()
        rs["flags"] = rs["flags"].apply(lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else x)
        keep = [c for c in SCHEMA_DOC["risk_scores"][1] if c in rs.columns]
        con.register("_rs", rs[keep])
        con.execute("CREATE TABLE risk_scores AS SELECT * FROM _rs")
    for t in ["_holders", "_accounts", "_tx", "_rs", "_gi", "_gr"]:
        try:
            con.unregister(t)
        except Exception:
            pass
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def schema_text(con: duckdb.DuckDBPyConnection) -> str:
    """Schema description for the LLM prompt."""
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    lines = []
    for t, (desc, cols) in SCHEMA_DOC.items():
        if t not in tables:
            continue
        lines.append(f"TABLE {t} -- {desc}")
        real = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]
        types = {r[0]: r[1] for r in con.execute(f"DESCRIBE {t}").fetchall()}
        for c in real:
            lines.append(f"  {c} {types[c]} -- {cols.get(c, '')}")
        lines.append("")
    return "\n".join(lines)

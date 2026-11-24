"""
Import real bank statements (CSV / Excel) exported from net banking or received
from banks, and convert them into a Kar Drishti case.

Works with the usual Indian statement layout:
  Date | Narration / Description / Particulars | Chq/Ref No | Withdrawal (Dr) | Deposit (Cr) | Balance
or a single Amount column with a Dr/Cr indicator.
"""
from __future__ import annotations

import io
import re
from difflib import get_close_matches

import pandas as pd

from .data import Case

COLS = {
    "date": ["txn date", "transaction date", "date", "value date", "value dt", "tran date", "posting date"],
    "narration": ["narration", "description", "particulars", "remarks", "details", "transaction details"],
    "ref": ["chq/ref no", "chq./ref.no.", "ref no", "reference", "cheque no", "chq no", "ref no./cheque no."],
    "debit": ["withdrawal amt", "withdrawal", "withdrawals", "debit", "debit amount", "dr", "withdrawal amount (inr )"],
    "credit": ["deposit amt", "deposit", "deposits", "credit", "credit amount", "cr", "deposit amount (inr )"],
    "amount": ["amount", "transaction amount", "amount (inr)"],
    "drcr": ["dr/cr", "cr/dr", "type", "dr / cr"],
}


def _find(cols, keys):
    low = {c.lower().strip(): c for c in cols}
    for k in keys:
        if k in low:
            return low[k]
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return None


def infer_mode(narr: str) -> str:
    n = str(narr).upper()
    for key, mode in [("ATM", "ATM"), ("CASH", "CASH"), ("CSH", "CASH"), ("BY CASH", "CASH"),
                      ("RTGS", "RTGS"), ("NEFT", "NEFT"), ("IMPS", "IMPS"), ("UPI", "UPI"),
                      ("CHQ", "CHEQUE"), ("CHEQUE", "CHEQUE"), ("CLG", "CHEQUE")]:
        if key in n:
            return mode
    return "OTHER"


def counterparty_from_narration(narr: str) -> str:
    """Best-effort party name from narrations like 'NEFT/HDFC0001/RAMESH KUMAR/...'."""
    n = str(narr)
    parts = [p.strip() for p in re.split(r"[/\-:]", n) if p.strip()]
    words = [p for p in parts if re.search(r"[A-Za-z]{3,}", p) and not re.match(
        r"^(NEFT|RTGS|IMPS|UPI|CHQ|ATM|CASH|BY|TO|TRF|TRANSFER|INB|MB|P2A|P2M|DEP|WDL|CR|DR)$", p, re.I)
             and not re.search(r"\d{6,}", p) and not re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", p)]
    return words[0][:60] if words else n[:40]


def normalise_statement(df: pd.DataFrame, account_id: str) -> pd.DataFrame:
    # drop junk rows above the real header if present
    if _find(df.columns, COLS["date"]) is None:
        for i in range(min(30, len(df))):
            row = [str(x).lower() for x in df.iloc[i].tolist()]
            if any(any(k == v.strip() for k in COLS["date"]) for v in row):
                df = df.iloc[i + 1:].set_axis(df.iloc[i].astype(str).tolist(), axis=1)
                break
    c_date, c_narr = _find(df.columns, COLS["date"]), _find(df.columns, COLS["narration"])
    c_dr, c_cr = _find(df.columns, COLS["debit"]), _find(df.columns, COLS["credit"])
    c_amt, c_type, c_ref = _find(df.columns, COLS["amount"]), _find(df.columns, COLS["drcr"]), _find(df.columns, COLS["ref"])
    if c_date is None or (c_dr is None and c_cr is None and c_amt is None):
        raise ValueError(f"Could not recognise date/amount columns. Found columns: {list(df.columns)}")

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(r"[,\s₹]|Rs\.?", "", regex=True)
                             .str.replace(r"\(?([\d.]+)\)?", r"\1", regex=True), errors="coerce").fillna(0)

    out = pd.DataFrame({"txn_date": pd.to_datetime(df[c_date], errors="coerce", dayfirst=True, format="mixed"),
                        "narration": df[c_narr].astype(str) if c_narr else "",
                        "ref_no": df[c_ref].astype(str) if c_ref else None})
    if c_dr is not None or c_cr is not None:
        dr = num(df[c_dr]) if c_dr is not None else 0
        cr = num(df[c_cr]) if c_cr is not None else 0
        out["direction"] = (pd.Series(cr, index=df.index) > 0).map({True: "CR", False: "DR"})
        out["amount"] = pd.Series(cr, index=df.index).where(out["direction"] == "CR", pd.Series(dr, index=df.index))
    else:
        amt = num(df[c_amt])
        t = df[c_type].astype(str).str.upper() if c_type else pd.Series("", index=df.index)
        out["direction"] = t.str.contains("CR").map({True: "CR", False: "DR"})
        out["amount"] = amt.abs()
    out = out[out.txn_date.notna() & (out.amount > 0)].copy()
    out["account_id"] = account_id
    out["mode"] = out.narration.map(infer_mode)
    out["counterparty_name"] = out.narration.map(counterparty_from_narration)
    out["counterparty_account_id"] = None
    return out


def read_any(file_name: str, data: bytes) -> pd.DataFrame:
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), dtype=str)
    return pd.read_csv(io.BytesIO(data), dtype=str, encoding_errors="ignore")


def build_case_from_statements(items: list[dict], case_name: str = "Uploaded case") -> Case:
    """items: [{file_name, data(bytes), holder_name, holder_type, declared_income, filed}]"""
    persons, firms, accounts, txs, itr = [], [], [], [], []
    holder_ids = {}
    for i, it in enumerate(items, 1):
        name = it["holder_name"].strip() or f"Holder {i}"
        if name not in holder_ids:
            is_firm = it.get("holder_type", "PERSON") == "FIRM"
            hid = f"{'F' if is_firm else 'P'}{len(holder_ids) + 1:04d}"
            holder_ids[name] = hid
            rec = {"name": name, "pan": it.get("pan") or "", "city": "", "phone": "", "address": ""}
            if is_firm:
                firms.append({"firm_id": hid, **rec, "gstin": "", "registration_date": None, "director_person_id": None})
            else:
                persons.append({"person_id": hid, **rec, "occupation": it.get("occupation", "Unknown"), "age": None})
            inc = float(it.get("declared_income") or 0)
            itr.append({"holder_id": hid, "pan": rec["pan"], "tax_year": "", "filed": bool(it.get("filed", True)),
                        "declared_income": inc, "gross_receipts": float(it.get("gross_receipts") or 0), "tax_paid": 0})
        hid = holder_ids[name]
        acc = f"AC{i:05d}"
        accounts.append({"account_id": acc, "holder_id": hid, "holder_type": "FIRM" if hid.startswith("F") else "PERSON",
                         "bank": it.get("bank", ""), "branch": "", "account_type": "", "open_date": None})
        txs.append(normalise_statement(read_any(it["file_name"], it["data"]), acc))
    tx = pd.concat(txs, ignore_index=True).sort_values("txn_date").reset_index(drop=True)
    tx["txn_id"] = [f"T{i + 1:07d}" for i in range(len(tx))]
    # link counterparties that are themselves holders in this case (fuzzy name match)
    names = {n.upper(): h for n, h in holder_ids.items()}
    def link(cp):
        hit = get_close_matches(str(cp).upper(), names.keys(), n=1, cutoff=0.8)
        return names[hit[0]] if hit else None
    tx["counterparty_holder_id_hint"] = tx.counterparty_name.map(link)
    return Case(name=case_name, persons=pd.DataFrame(persons), firms=pd.DataFrame(firms),
                accounts=pd.DataFrame(accounts), transactions=tx, itr_filings=pd.DataFrame(itr))

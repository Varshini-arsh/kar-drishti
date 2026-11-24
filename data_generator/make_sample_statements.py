"""
Export a few accounts from the demo case as realistic bank-statement files
(HDFC-style CSV and SBI-style Excel) so the 'Upload statements' feature can be tried.
Usage: python data_generator/make_sample_statements.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from kar_drishti.data import load_case  # noqa: E402

OUT = config.BASE_DIR / "sample_statements"


def export(case, holder_id, fmt, out_dir=OUT):
    tx = case.transactions[case.transactions.holder_id == holder_id].sort_values(["txn_date", "txn_id"])
    acc = tx.account_id.iloc[0]
    tx = tx[tx.account_id == acc]
    bal = 250000 + (tx.amount.where(tx.direction == "CR", -tx.amount)).cumsum()
    name = case.holder_name(holder_id)
    narr = tx.apply(lambda r: r.narration if "/" in str(r.narration) else
                    f"{r['mode']}/{r.counterparty_name}/{r.narration}", axis=1)
    if fmt == "csv":
        df = pd.DataFrame({"Date": tx.txn_date.dt.strftime("%d/%m/%y"), "Narration": narr,
                           "Chq./Ref.No.": tx.ref_no, "Value Dt": tx.txn_date.dt.strftime("%d/%m/%y"),
                           "Withdrawal Amt.": tx.amount.where(tx.direction == "DR", ""),
                           "Deposit Amt.": tx.amount.where(tx.direction == "CR", ""), "Closing Balance": bal.round(2)})
        fp = out_dir / f"statement_{name.replace(' ', '_')}_{acc}.csv"
        df.to_csv(fp, index=False)
    else:
        df = pd.DataFrame({"Txn Date": tx.txn_date.dt.strftime("%d %b %Y"), "Description": narr,
                           "Ref No./Cheque No.": tx.ref_no,
                           "Debit": tx.amount.where(tx.direction == "DR", ""),
                           "Credit": tx.amount.where(tx.direction == "CR", ""), "Balance": bal.round(2)})
        fp = out_dir / f"statement_{name.replace(' ', '_')}_{acc}.xlsx"
        df.to_excel(fp, index=False)
    return fp, name


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    case = load_case(config.DEMO_DIR)
    gt = case.ground_truth["planted"]
    picks = [(gt["structuring"][0], "csv"), (gt["pass_through"][0], "xlsx"), (gt["non_filer"][0], "csv")]
    for hid, fmt in picks:
        fp, name = export(case, hid, fmt)
        print(f"wrote {fp.name}  ({name})")

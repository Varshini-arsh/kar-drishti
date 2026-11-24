"""
Synthetic revenue demo files for Pragya (FAKE data - no real offices or taxpayers).

Writes to data/demo_revenue/:
  revenue_targets.csv        annual target per range (region > charge > range)
  collections.csv            this year's collections per range per month (April to the 'as on' month)
  collections_last_year.csv  last year's collections per range per month (all 12 months)
  demand_register.csv        outstanding demands with disputed / stay status
  recovery_actions.csv       actions taken on demands
  refunds.csv                refunds issued this year
All amounts are whole rupees, so totals add up exactly at every level.

Usage:  python data_generator/generate_revenue.py [--seed 7]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "demo_revenue"
FY, LAST_FY = "2026-27", "2025-26"
AS_ON = "2026-10-31"                       # collections are loaded up to this date
MONTHS = [f"2026-{m:02d}" for m in range(4, 13)] + [f"2027-{m:02d}" for m in range(1, 4)]
LAST_MONTHS = [f"2025-{m:02d}" for m in range(4, 13)] + [f"2026-{m:02d}" for m in range(1, 4)]
# Indian collections bunch up in advance-tax months (Jun, Sep, Dec, Mar) and at year end.
PATTERN = np.array([5, 6, 11, 6, 6, 12, 6, 6, 12, 7, 8, 15], dtype=float)
PATTERN = PATTERN / PATTERN.sum()

HIERARCHY = {"Region North (demo)": {"Charge N1 (demo)": ["Range N1-1", "Range N1-2", "Range N1-3"],
                                     "Charge N2 (demo)": ["Range N2-1", "Range N2-2", "Range N2-3"]},
             "Region South (demo)": {"Charge S1 (demo)": ["Range S1-1", "Range S1-2", "Range S1-3"],
                                     "Charge S2 (demo)": ["Range S2-1", "Range S2-2", "Range S2-3"]}}

TAXPAYERS = ["Ganesh Commodities", "Kuber Traders", "Siddhi Commodities", "Navkar Agencies", "Om Sai Exports",
             "Shubh Infra Pvt Ltd", "Ananta Solutions", "Riddhi Industries", "Mahalaxmi Trading Co",
             "Shree Balaji Enterprises", "Lakshmi Commodities", "Vardhan Enterprises", "Arjun Mehta",
             "Kavya Nair", "Rohit Bansal", "Priya Iyer", "Sanjay Kulkarni", "Meera Joshi", "Imran Shaikh",
             "Deepa Reddy", "Vikram Sethi", "Neha Kapoor", "Alok Verma", "Farah Khan"]
ACTIONS = ["Reminder letter", "Show-cause notice", "Personal hearing", "Bank account attachment",
           "Instalment plan agreed"]


def ranges():
    for reg, charges in HIERARCHY.items():
        for ch, rgs in charges.items():
            for rg in rgs:
                yield reg, ch, rg


def fake_pan(rng, kind="P"):
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return "".join(rng.choice(list(letters), 3)) + kind + rng.choice(list(letters)) + \
        f"{rng.integers(1000, 9999)}" + rng.choice(list(letters))


def generate(out: Path = OUT, seed: int = 7, quiet: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    n_done = MONTHS.index(AS_ON[:7]) + 1                       # months with collections (Apr..Oct = 7)
    targets, coll, last = [], [], []
    for i, (reg, ch, rg) in enumerate(ranges()):
        last_total = int(rng.integers(40, 160)) * 10_000_000        # ₹40-160 crore last year
        for m, share in zip(LAST_MONTHS, PATTERN):
            amt = int(round(last_total * share * rng.uniform(0.9, 1.1), -3))
            last.append({"fy": LAST_FY, "region": reg, "charge": ch, "range": rg, "month": m, "amount_collected": amt})
        ly = sum(r["amount_collected"] for r in last if r["range"] == rg)
        target = int(round(ly * rng.uniform(1.08, 1.22), -5))
        targets.append({"fy": FY, "region": reg, "charge": ch, "range": rg, "target": target})
        pace = rng.uniform(0.82, 1.12)                                # some ranges run ahead, some behind
        if rg == "Range S2-3":
            pace = 0.62                                               # one clearly lagging range
        for m, share in list(zip(MONTHS, PATTERN))[:n_done]:
            amt = int(round(target * share * pace * rng.uniform(0.92, 1.08), -3))
            coll.append({"fy": FY, "region": reg, "charge": ch, "range": rg, "month": m, "amount_collected": amt})

    demands, actions, refunds = [], [], []
    rlist = list(ranges())
    as_on = date.fromisoformat(AS_ON)
    firm_words = ("Commodities", "Traders", "Agencies", "Exports", "Ltd", "Solutions", "Industries", "Co", "Enterprises")
    pans = {n: fake_pan(rng, "F" if n.split()[-1] in firm_words else "P") for n in TAXPAYERS}   # one PAN each
    for k, name in enumerate(TAXPAYERS):
        reg, ch, rg = rlist[k % len(rlist)]
        for j in range(1 if k % 3 else 2):
            years_old = float(rng.choice([0.4, 0.9, 1.5, 2.5, 3.5, 6.0]))
            raised = date.fromordinal(as_on.toordinal() - int(years_old * 365))
            amount = int(rng.integers(5, 400)) * 100_000
            paid = int(round(amount * float(rng.choice([0, 0, 0.1, 0.25, 0.5])), -3))
            disputed = bool(rng.random() < 0.35)
            stay = disputed and bool(rng.random() < 0.5)
            if name == "Ganesh Commodities" and j == 0:              # the cross-task demo case
                amount, paid, disputed, stay = 16_900_000, 0, False, False
                raised = date(2025, 2, 14)
            if name == "Kuber Traders" and j == 0:                   # large but stayed: must rank lower
                amount, paid, disputed, stay = 42_500_000, 0, True, True
            ay = f"{raised.year - (0 if raised.month >= 4 else 1)}-{(raised.year + (1 if raised.month >= 4 else 0)) % 100:02d}"
            did = f"DM{len(demands) + 1:04d}"
            demands.append({"demand_id": did, "pan": pans[name],
                            "taxpayer_name": name, "region": reg, "charge": ch, "range": rg,
                            "assessment_year": ay, "date_raised": raised.isoformat(), "demand_amount": amount,
                            "collected_so_far": paid, "outstanding": amount - paid,
                            "disputed": "Yes" if disputed else "No", "stay": "Yes" if stay else "No",
                            "appeal_status": "Appeal pending" if disputed else "No appeal"})
            for a in range(int(rng.integers(0, 3))):
                actions.append({"demand_id": did, "date": date.fromordinal(raised.toordinal() + 60 * (a + 1)).isoformat(),
                                "action": ACTIONS[a % len(ACTIONS)], "amount_recovered": 0})
    for (reg, ch, rg) in rlist:
        for m in MONTHS[:n_done]:
            if rng.random() < 0.6:
                refunds.append({"refund_id": f"RF{len(refunds) + 1:04d}", "region": reg, "charge": ch, "range": rg,
                                "month": m, "refund_amount": int(rng.integers(2, 60)) * 100_000, "status": "Issued"})
    frames = {"revenue_targets": pd.DataFrame(targets), "collections": pd.DataFrame(coll),
              "collections_last_year": pd.DataFrame(last), "demand_register": pd.DataFrame(demands),
              "recovery_actions": pd.DataFrame(actions), "refunds": pd.DataFrame(refunds)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "README.txt").write_text("SYNTHETIC DEMO DATA - not real offices, taxpayers or collections.\n"
                                    f"Financial year {FY}, collections as on {AS_ON}. Seed {seed}.\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"fy": FY, "last_fy": LAST_FY, "as_on": AS_ON, "seed": seed}),
                                   encoding="utf-8")
    if not quiet:
        print(f"Synthetic revenue files written to {out}")
        for name, df in frames.items():
            print(f"  {name:22s} {len(df):>5} rows")
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic revenue demo files for Pragya")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    generate(a.out, a.seed)

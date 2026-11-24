"""
Synthetic case-data generator for Kar Drishti.

Creates a realistic but 100% FAKE "jurisdiction" for one financial year:
persons, firms, bank accounts, bank transactions and ITR filings.
Known fraud patterns are PLANTED inside the data and recorded in
ground_truth.json so we can measure how many the agent finds.

No real taxpayer data is used. Names, PANs, GSTINs and account numbers are random.

Usage:
    python data_generator/generate_data.py            # writes to data/demo_case
    python data_generator/generate_data.py --seed 7   # a different random case
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CITIES = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune"]
BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Canara Bank", "Bank of Baroda"]
STATE_CODES = {"Bengaluru": "29", "Mumbai": "27", "Delhi": "07",
               "Chennai": "33", "Hyderabad": "36", "Pune": "27"}
FIRM_SUFFIX = ["Traders", "Enterprises", "Pvt Ltd", "Industries", "Exports",
               "Solutions", "Agencies", "Infra Pvt Ltd", "Trading Co"]
SHELL_WORDS = ["Vardhan", "Siddhi", "Shree Balaji", "Om Sai", "Navkar", "Kuber",
               "Lakshmi", "Riddhi", "Ganesh", "Mahalaxmi", "Shubh", "Ananta"]


class Gen:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.fake = Faker("en_IN")
        Faker.seed(seed)
        self.start = date.fromisoformat(config.FY_START)
        self.end = date.fromisoformat(config.FY_END)
        self.days = (self.end - self.start).days + 1
        self.persons, self.firms, self.accounts, self.txns, self.itr = [], [], [], [], []
        self._txn_seq = 0
        self._ref_seq = 0
        self.used_pans: set[str] = set()
        self.truth = {k: [] for k in ["structuring", "round_trip", "pass_through",
                                       "income_mismatch", "non_filer", "round_cash",
                                       "shell_cluster", "gst_circular", "gst_itc_passthrough",
                                       "gst_no_eway", "gst_risky_itc"]}
        self.gst_invoices, self.gst_returns = [], []
        self.truth_notes: list[str] = []

    # ------------------------------------------------------------ id helpers
    def pan(self, kind: str) -> str:
        while True:
            p = ("".join(random.choices(string.ascii_uppercase, k=3)) + kind
                 + random.choice(string.ascii_uppercase)
                 + f"{random.randint(0, 9999):04d}" + random.choice(string.ascii_uppercase))
            if p not in self.used_pans:
                self.used_pans.add(p)
                return p

    def phone(self) -> str:
        return f"9{random.randint(100000000, 999999999)}"

    def rand_date(self, lo: date | None = None, hi: date | None = None) -> date:
        lo, hi = lo or self.start, hi or self.end
        return lo + timedelta(days=int(self.rng.integers(0, (hi - lo).days + 1)))

    def clamp(self, d: date) -> date:
        return min(max(d, self.start), self.end)

    # ------------------------------------------------------------ entities
    def add_person(self, occupation: str, income: float, city: str | None = None,
                   address: str | None = None, filed: bool = True,
                   receipts: float = 0.0) -> dict:
        city = city or random.choice(CITIES)
        pid = f"P{len(self.persons) + 1:04d}"
        p = {"person_id": pid, "name": self.fake.name(), "pan": self.pan("P"),
             "city": city, "occupation": occupation,
             "age": int(self.rng.integers(24, 70)), "phone": self.phone(),
             "address": address or f"{self.fake.building_number()}, {self.fake.street_name()}, {city}"}
        self.persons.append(p)
        self.itr.append({"holder_id": pid, "pan": p["pan"], "tax_year": config.TAX_YEAR,
                         "filed": filed, "declared_income": round(income if filed else 0, 0),
                         "gross_receipts": round(receipts if filed else 0, 0),
                         "tax_paid": round(self.tax(income) if filed else 0, 0)})
        return p

    def add_firm(self, name: str, city: str, reg_date: date, director: str,
                 turnover: float, address: str | None = None, phone: str | None = None,
                 shell: bool = False) -> dict:
        fid = f"F{len(self.firms) + 1:04d}"
        base, k = name, 2
        while any(f["name"] == name for f in self.firms):   # keep firm names unique
            name = f"{base} {k}"
            k += 1
        pan = self.pan("C" if "Ltd" in name else "F")
        f = {"firm_id": fid, "name": name, "pan": pan,
             "gstin": f"{STATE_CODES[city]}{pan}1Z{random.choice(string.ascii_uppercase + string.digits)}",
             "city": city, "registration_date": reg_date, "director_person_id": director,
             "phone": phone or self.phone(),
             "address": address or f"{self.fake.building_number()}, {self.fake.street_name()}, {city}",
             "_shell": shell}
        self.firms.append(f)
        self.itr.append({"holder_id": fid, "pan": pan, "tax_year": config.TAX_YEAR,
                         "filed": True, "declared_income": round(turnover * 0.08, 0),
                         "gross_receipts": round(turnover, 0),
                         "tax_paid": round(turnover * 0.08 * 0.25, 0)})
        return f

    def add_account(self, holder_id: str, holder_type: str, city: str,
                    acc_type: str, bank: str | None = None) -> str:
        aid = f"AC{len(self.accounts) + 1:05d}"
        self.accounts.append({"account_id": aid, "holder_id": holder_id,
                              "holder_type": holder_type, "bank": bank or random.choice(BANKS),
                              "branch": f"{city} - {self.fake.street_name()}",
                              "account_type": acc_type,
                              "open_date": self.rand_date(date(2010, 1, 1), date(2024, 12, 31))})
        return aid

    @staticmethod
    def tax(income: float) -> float:
        # rough new-regime style slab estimate (illustrative only)
        slabs = [(400000, 0), (800000, .05), (1200000, .10), (1600000, .15),
                 (2000000, .20), (2400000, .25), (float("inf"), .30)]
        t, prev = 0.0, 0
        for lim, r in slabs:
            if income > prev:
                t += (min(income, lim) - prev) * r
            prev = lim
        return 0 if income <= 1200000 else t

    # ------------------------------------------------------------ transactions
    def _row(self, acc, d, mode, direction, amount, cp_acc, cp_name, narration, ref):
        self._txn_seq += 1
        self.txns.append({"txn_id": f"T{self._txn_seq:07d}", "account_id": acc,
                          "txn_date": d, "mode": mode, "direction": direction,
                          "amount": round(float(amount), 2),
                          "counterparty_account_id": cp_acc, "counterparty_name": cp_name,
                          "narration": narration, "ref_no": ref})

    def ref(self) -> str:
        self._ref_seq += 1
        return f"REF{self._ref_seq:08d}"

    def external(self, acc, d, mode, direction, amount, cp_name, narration):
        self._row(acc, self.clamp(d), mode, direction, amount, None, cp_name, narration, self.ref())

    def transfer(self, from_acc, to_acc, d, amount, mode, narration):
        d = self.clamp(d)
        r = self.ref()
        fn, tn = self.holder_name(from_acc), self.holder_name(to_acc)
        self._row(from_acc, d, mode, "DR", amount, to_acc, tn, f"{mode}/{tn}/{narration}", r)
        self._row(to_acc, d, mode, "CR", amount, from_acc, fn, f"{mode}/{fn}/{narration}", r)

    def holder_name(self, acc_id: str) -> str:
        acc = self._acc_index[acc_id]
        hid = acc["holder_id"]
        src = self._person_index if acc["holder_type"] == "PERSON" else self._firm_index
        return src[hid]["name"]

    def index(self):
        self._acc_index = {a["account_id"]: a for a in self.accounts}
        self._person_index = {p["person_id"]: p for p in self.persons}
        self._firm_index = {f["firm_id"]: f for f in self.firms}

    def accounts_of(self, holder_id):
        return [a["account_id"] for a in self.accounts if a["holder_id"] == holder_id]

    def month_starts(self):
        d = self.start
        while d <= self.end:
            yield d
            d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)

    # ------------------------------------------------------------ normal behaviour
    def normal_person_activity(self, p, employer_acc: str | None):
        accs = self.accounts_of(p["person_id"])
        main = accs[0]
        itr = next(i for i in self.itr if i["holder_id"] == p["person_id"])
        income = max(itr["declared_income"], 300000)
        occ = p["occupation"]
        for m in self.month_starts():
            if occ == "Salaried" and employer_acc:
                self.transfer(employer_acc, main, m + timedelta(days=int(self.rng.integers(0, 3))),
                              round(income / 12 * self.rng.uniform(0.92, 1.0), -2), "NEFT",
                              f"SALARY {m.strftime('%b%Y').upper()}")
            elif occ in ("Business", "Professional", "Self-employed"):
                n = int(self.rng.integers(6, 16))
                monthly = itr["gross_receipts"] / 12 if itr["gross_receipts"] else income / 12 * 3
                for amt in self.rng.dirichlet(np.ones(n)) * monthly:
                    self.external(main, m + timedelta(days=int(self.rng.integers(0, 28))),
                                  random.choice(["UPI", "NEFT", "IMPS"]), "CR", round(amt, 0),
                                  self.fake.name(), "CUSTOMER PAYMENT")
                for _ in range(int(self.rng.integers(0, 3))):
                    self.external(main, m + timedelta(days=int(self.rng.integers(0, 28))), "CASH",
                                  "CR", round(self.rng.uniform(5000, 49000), -2), "SELF",
                                  "CASH DEPOSIT")
            elif occ == "Retired":
                self.external(main, m + timedelta(days=1), "NEFT", "CR",
                              round(income / 12, -2), "PENSION DISBURSING OFFICE", "PENSION")
            # spending
            for _ in range(int(self.rng.integers(8, 20))):
                self.external(main, m + timedelta(days=int(self.rng.integers(0, 28))), "UPI", "DR",
                              round(self.rng.lognormal(6.6, 0.9), 0),
                              random.choice(["SWIGGY", "AMAZON", "DMART", "ZOMATO", "IRCTC",
                                             "BESCOM", "JIO", "FLIPKART", "LOCAL STORE"]),
                              "UPI PAYMENT")
            for _ in range(int(self.rng.integers(1, 3))):
                self.external(main, m + timedelta(days=int(self.rng.integers(0, 28))), "ATM", "DR",
                              float(self.rng.integers(4, 40) * 500), "ATM", "ATM CASH WITHDRAWAL")
            if self.rng.random() < 0.7:
                self.external(main, m + timedelta(days=5), "NEFT", "DR",
                              round(income / 12 * self.rng.uniform(0.15, 0.3), -2),
                              random.choice(["HOME LOAN EMI", "RENT - LANDLORD", "SIP - MUTUAL FUND"]),
                              "STANDING INSTRUCTION")
            if len(accs) > 1 and self.rng.random() < 0.3:
                self.transfer(main, accs[1], m + timedelta(days=10),
                              round(income / 12 * 0.2, -2), "IMPS", "SELF TRANSFER")

    def normal_firm_activity(self, f, suppliers: list[str], employees: list[str]):
        acc = self.accounts_of(f["firm_id"])[0]
        itr = next(i for i in self.itr if i["holder_id"] == f["firm_id"])
        monthly = itr["gross_receipts"] / 12
        for m in self.month_starts():
            n = int(self.rng.integers(10, 25))
            for amt in self.rng.dirichlet(np.ones(n)) * monthly:
                self.external(acc, m + timedelta(days=int(self.rng.integers(0, 28))),
                              random.choice(["NEFT", "RTGS"]) if amt > 200000 else "NEFT", "CR",
                              round(amt, 0), self.fake.company(), "SALES RECEIPT")
            for s in random.sample(suppliers, k=min(len(suppliers), int(self.rng.integers(0, 3)))):
                self.transfer(acc, s, m + timedelta(days=int(self.rng.integers(3, 25))),
                              round(monthly * self.rng.uniform(0.05, 0.2), 0), "RTGS",
                              "PURCHASE INVOICE")
            for _ in range(int(self.rng.integers(3, 8))):
                self.external(acc, m + timedelta(days=int(self.rng.integers(0, 28))), "NEFT", "DR",
                              round(monthly * self.rng.uniform(0.03, 0.1), 0), self.fake.company(),
                              "VENDOR PAYMENT")
            self.external(acc, m + timedelta(days=20), "NEFT", "DR", round(monthly * 0.03, 0),
                          "GST / TAX PAYMENT", "TAX PAYMENT")

    # ------------------------------------------------------------ build everything
    def build(self):
        # ---- normal persons (with some families sharing an address)
        occ_choices = ["Salaried"] * 6 + ["Business"] * 2 + ["Professional", "Retired", "Self-employed"]
        normal_persons = []
        for i in range(120):
            occ = random.choice(occ_choices)
            inc = float(np.clip(self.rng.lognormal(13.7, 0.5), 350000, 6000000))
            receipts = inc * self.rng.uniform(3, 5) if occ in ("Business", "Professional", "Self-employed") else 0
            addr = None
            if i % 4 == 1 and normal_persons:      # every 4th person lives with a family member
                addr = normal_persons[-1]["address"]
            p = self.add_person(occ, inc, address=addr, receipts=receipts,
                                city=normal_persons[-1]["city"] if addr else None)
            normal_persons.append(p)

        # ---- normal firms (directors are business persons)
        directors = [p for p in normal_persons if p["occupation"] == "Business"] or normal_persons
        normal_firms = []
        for i in range(28):
            d = random.choice(directors)
            name = f"{self.fake.last_name()} {random.choice(FIRM_SUFFIX)}"
            normal_firms.append(self.add_firm(name, d["city"], self.rand_date(date(2004, 1, 1), date(2021, 12, 31)),
                                              d["person_id"], float(self.rng.uniform(2e7, 2e8))))

        # ---- planted: controllers of shell networks / round-trippers
        controllers = [self.add_person("Business", float(self.rng.uniform(6e5, 1.2e6)),
                                       receipts=float(self.rng.uniform(2e6, 4e6))) for _ in range(3)]
        # ---- planted: two shell clusters (shared address + phone, recently registered)
        shell_clusters = []
        for c in range(2):
            city = random.choice(CITIES)
            addr = f"Office No. {random.randint(101, 499)}, {self.fake.street_name()}, {city}"
            ph = self.phone()
            cluster = []
            for k in range(4):
                name = f"{random.choice(SHELL_WORDS)} {random.choice(['Traders', 'Enterprises', 'Impex', 'Commodities'])}"
                cluster.append(self.add_firm(
                    name, city, self.rand_date(date(2024, 6, 1), date(2025, 8, 31)),
                    controllers[c]["person_id"] if k < 2 else controllers[2]["person_id"],
                    float(self.rng.uniform(5e5, 2e6)), address=addr, phone=ph if k != 3 else self.phone(),
                    shell=True))
            shell_clusters.append(cluster)
            self.truth["shell_cluster"] += [f["firm_id"] for f in cluster]

        # ---- planted: structuring persons
        structurers = [self.add_person(random.choice(["Salaried", "Self-employed"]),
                                       float(self.rng.uniform(4e5, 8e5)),
                                       receipts=0) for _ in range(5)]
        # ---- planted: pass-through (mule) holders
        mules = [self.add_person("Self-employed", float(self.rng.uniform(2.5e5, 3.8e5)),
                                 receipts=float(self.rng.uniform(3e5, 5e5))) for _ in range(4)]
        # ---- planted: income mismatch (3 low-income filers + 3 non-filers)
        mismatch_filers = [self.add_person("Salaried", float(self.rng.uniform(5e5, 9e5))) for _ in range(3)]
        non_filers = [self.add_person("Self-employed", 0, filed=False) for _ in range(3)]
        # ---- planted: large round-figure cash withdrawals
        round_cash = [self.add_person("Business", float(self.rng.uniform(1e6, 2e6)),
                                      receipts=float(self.rng.uniform(5e6, 9e6))) for _ in range(3)]

        # ---- accounts
        for p in self.persons:
            n = 3 if p in structurers else (2 if self.rng.random() < 0.35 else 1)
            banks = random.sample(BANKS, n)
            for b in banks:
                self.add_account(p["person_id"], "PERSON", p["city"], "Savings", b)
        for f in self.firms:
            self.add_account(f["firm_id"], "FIRM", f["city"], "Current")
        self.index()

        # ---- normal activity
        firm_accs = [self.accounts_of(f["firm_id"])[0] for f in normal_firms]
        salaried = [p for p in self.persons if p["occupation"] == "Salaried"]
        for p in self.persons:
            employer = random.choice(firm_accs) if p["occupation"] == "Salaried" else None
            self.normal_person_activity(p, employer)
        for i, f in enumerate(normal_firms):
            # suppliers only among later firms -> no accidental cycles
            self.normal_firm_activity(f, firm_accs[i + 1:i + 4], [])
        for f in [f for c in shell_clusters for f in c]:
            acc = self.accounts_of(f["firm_id"])[0]
            for m in self.month_starts():
                self.external(acc, m + timedelta(days=int(self.rng.integers(0, 28))), "NEFT", "CR",
                              round(self.rng.uniform(2e4, 1.5e5), 0), self.fake.company(), "SALES RECEIPT")

        # ---- planted pattern 1: structuring (cash deposits just under Rs 50,000)
        lo = config.CASH_PAN_THRESHOLD * (1 - config.STRUCTURING_BAND)
        for p in structurers:
            accs = self.accounts_of(p["person_id"])
            n = int(self.rng.integers(18, 40))
            burst_start = self.rand_date(self.start, self.end - timedelta(days=120))
            total = 0
            for _ in range(n):
                d = burst_start + timedelta(days=int(self.rng.integers(0, 110)))
                amt = float(self.rng.integers(int(lo / 100) + 1, 499) * 100)
                total += amt
                self.external(random.choice(accs), d, "CASH", "CR", amt, "SELF", "CASH DEPOSIT")
            self.external(accs[0], burst_start + timedelta(days=115), "RTGS", "DR", round(total * 0.9, -3),
                          "SRI SAI BUILDERS & DEVELOPERS", "RTGS/PROPERTY BOOKING ADVANCE")
            self.truth["structuring"].append(p["person_id"])
        self.truth_notes.append("structuring: repeated cash deposits of Rs 45,000-49,900 spread across 3 accounts")

        # ---- planted pattern 2: round-tripping through shell firms
        loops = [
            [controllers[0], shell_clusters[0][0], shell_clusters[0][1]],
            [controllers[1], shell_clusters[0][2], shell_clusters[1][0], shell_clusters[1][1]],
            [controllers[2], shell_clusters[1][2], shell_clusters[1][3]],
        ]
        for loop in loops:
            ids = [h.get("person_id") or h.get("firm_id") for h in loop]
            accs = [self.accounts_of(h)[0] for h in ids]
            for _ in range(int(self.rng.integers(2, 4))):
                d = self.rand_date(self.start, self.end - timedelta(days=40))
                amt = float(self.rng.uniform(2e6, 8e6))
                labels = ["ADVANCE AGAINST SUPPLY", "UNSECURED LOAN", "SHARE APPLICATION MONEY",
                          "PURCHASE INVOICE"]
                for k in range(len(accs)):
                    src, dst = accs[k], accs[(k + 1) % len(accs)]
                    self.transfer(src, dst, d, round(amt, 0), "RTGS", random.choice(labels))
                    d += timedelta(days=int(self.rng.integers(1, 6)))
                    amt *= self.rng.uniform(0.98, 0.995)   # small skim at each hop
            for h in ids:
                if h not in self.truth["round_trip"]:
                    self.truth["round_trip"].append(h)
        self.truth_notes.append("round_trip: money leaves a controller, passes 2-3 shell firms, returns within days")

        # ---- planted pattern 3: pass-through / mule accounts
        shell_accs = [self.accounts_of(f["firm_id"])[0] for c in shell_clusters for f in c]
        for p in mules:
            acc = self.accounts_of(p["person_id"])[0]
            for m in self.month_starts():
                for _ in range(int(self.rng.integers(2, 5))):
                    d = m + timedelta(days=int(self.rng.integers(0, 26)))
                    incoming = [float(self.rng.uniform(5e4, 3e5)) for _ in range(int(self.rng.integers(3, 8)))]
                    for amt in incoming:
                        self.external(acc, d, random.choice(["IMPS", "UPI", "NEFT"]), "CR",
                                      round(amt, 0), self.fake.name(), "TRANSFER")
                    out = round(sum(incoming) * self.rng.uniform(0.95, 0.995), 0)
                    if self.rng.random() < 0.5:
                        self.transfer(acc, random.choice(shell_accs), d + timedelta(days=int(self.rng.integers(0, 2))),
                                      out, "RTGS", "PAYMENT")
                    else:
                        self.external(acc, d + timedelta(days=int(self.rng.integers(0, 2))), "RTGS", "DR", out,
                                      random.choice(["ZEN CAPITAL SERVICES", "RAPID FX TRADERS", "M K ENTERPRISES"]),
                                      "RTGS PAYMENT")
            self.truth["pass_through"].append(p["person_id"])
            self.truth["income_mismatch"].append(p["person_id"])
        self.truth_notes.append("pass_through: many small credits forwarded out within 0-1 day (mule behaviour)")

        # ---- planted pattern 4: income mismatch and non-filers
        for p in mismatch_filers + non_filers:
            acc = self.accounts_of(p["person_id"])[0]
            target = float(self.rng.uniform(4e6, 1.5e7))
            n = int(self.rng.integers(4, 9))
            for amt in self.rng.dirichlet(np.ones(n)) * target:
                self.external(acc, self.rand_date(), "RTGS", "CR", round(amt, 0),
                              random.choice([self.fake.name(), self.fake.company()]),
                              random.choice(["COMMISSION", "CONSULTANCY", "SALE CONSIDERATION", "ADVANCE"]))
            self.external(acc, self.rand_date(), "RTGS", "DR", round(target * 0.6, -3),
                          "PRESTIGE ESTATES - FLAT BOOKING", "PROPERTY PURCHASE")
            key = "non_filer" if p in non_filers else "income_mismatch"
            self.truth[key].append(p["person_id"])
        self.truth_notes.append("income_mismatch / non_filer: bank credits far above declared income or no return filed")

        # ---- planted pattern 5: large round-figure cash withdrawals
        for p in round_cash:
            acc = self.accounts_of(p["person_id"])[0]
            for _ in range(int(self.rng.integers(4, 8))):
                self.external(acc, self.rand_date(), "CASH", "DR",
                              float(random.choice([250000, 500000, 500000, 1000000])), "SELF",
                              "CASH WITHDRAWAL - SELF CHEQUE")
            self.truth["round_cash"].append(p["person_id"])
        self.truth_notes.append("round_cash: repeated cash withdrawals of exactly Rs 2.5L / 5L / 10L")
        self.build_gst(normal_firms, shell_clusters)

    # ------------------------------------------------------------ GST layer
    def build_gst(self, normal_firms, shell_clusters):
        """GST B2B invoices + monthly returns. Uses its own random stream so the
        bank data above is identical with or without this layer."""
        rng = np.random.default_rng(int(self.rng.integers(0, 2**31)) + 1)
        seq = [0]
        months = list(self.month_starts())
        hsn = ["7214 - Steel bars", "5208 - Cotton fabric", "3901 - Polymers", "4802 - Paper",
               "2523 - Cement", "7403 - Copper", "8471 - Computers", "3004 - Medicines"]

        def inv(seller, buyer, d, value, eway=True, goods=None):
            seq[0] += 1
            self.gst_invoices.append({
                "invoice_id": f"INV{seq[0]:06d}", "seller_firm_id": seller["firm_id"],
                "seller_gstin": seller["gstin"], "buyer_firm_id": buyer["firm_id"],
                "buyer_gstin": buyer["gstin"], "invoice_date": self.clamp(d),
                "hsn": goods or random.choice(hsn), "taxable_value": round(value, 0),
                "gst_amount": round(value * 0.18, 0),
                "eway_bill_no": f"EWB{rng.integers(10**11, 10**12)}" if eway else None})

        # normal B2B trade: only from earlier firms to later firms (no loops), with e-way bills
        for i, f in enumerate(normal_firms):
            buyers = normal_firms[i + 1:i + 4]
            for m in months:
                for b in buyers:
                    if rng.random() < 0.6:
                        inv(f, b, m + timedelta(days=int(rng.integers(0, 27))), float(rng.uniform(2e5, 2.5e6)),
                            eway=rng.random() > 0.03)

        shells = [f for c in shell_clusters for f in c]
        # planted: circular trading among shell firms (invoices go round, no goods move)
        rings = [shell_clusters[0], shell_clusters[1]]
        for ring in rings:
            for m in months:
                v = float(rng.uniform(3e6, 9e6))
                d = m + timedelta(days=int(rng.integers(0, 10)))
                for k in range(len(ring)):
                    inv(ring[k], ring[(k + 1) % len(ring)], d + timedelta(days=k), v * rng.uniform(1.0, 1.03),
                        eway=rng.random() < 0.15, goods="7403 - Copper")
        self.truth["gst_circular"] += [f["firm_id"] for f in shells]
        self.truth["gst_no_eway"] += [f["firm_id"] for f in shells]
        # planted: fake ITC sold to real-looking businesses (paper invoices, no e-way bill)
        beneficiaries = random.sample(normal_firms[:-3], 3)
        for b in beneficiaries:
            for m in months:
                if rng.random() < 0.8:
                    inv(random.choice(shells), b, m + timedelta(days=int(rng.integers(0, 27))),
                        float(rng.uniform(8e5, 3e6)), eway=False, goods="7214 - Steel bars")
            self.truth["gst_risky_itc"].append(b["firm_id"])
        self.truth_notes.append("gst_circular: shell firms invoice each other in a ring every month (mostly no e-way bills)")
        self.truth_notes.append("gst_risky_itc: 3 ordinary-looking firms claim ITC on paper invoices bought from shell firms")

        # monthly GSTR-3B style summary
        inv_df = pd.DataFrame(self.gst_invoices)
        inv_df["period"] = pd.to_datetime(inv_df.invoice_date).dt.strftime("%Y-%m")
        out_b2b = inv_df.groupby(["seller_firm_id", "period"]).gst_amount.sum()
        in_b2b = inv_df.groupby(["buyer_firm_id", "period"]).gst_amount.sum()
        turnover = {i["holder_id"]: i["gross_receipts"] for i in self.itr}
        shell_ids = {f["firm_id"] for f in shells}
        for f in normal_firms + shells:
            fid = f["firm_id"]
            for m in months:
                p = m.strftime("%Y-%m")
                b2b_out, b2b_in = float(out_b2b.get((fid, p), 0)), float(in_b2b.get((fid, p), 0))
                if fid in shell_ids:
                    output = b2b_out
                    itc = max(b2b_in, output * rng.uniform(0.985, 1.0))   # credit wipes out almost all tax
                else:
                    output = max(turnover[fid] / 12 * 0.18, b2b_out)
                    itc = output * rng.uniform(0.55, 0.8) * 0.85 + b2b_in * 0.15
                    if fid in {b["firm_id"] for b in beneficiaries}:
                        itc = min(output, itc + b2b_in)
                cash = max(output - itc, 0)
                self.gst_returns.append({"firm_id": fid, "gstin": f["gstin"], "period": p,
                                         "outward_taxable_value": round(output / 0.18, 0),
                                         "output_tax": round(output, 0), "itc_claimed": round(itc, 0),
                                         "tax_paid_cash": round(cash, 0)})
        self.truth["gst_itc_passthrough"] += [f["firm_id"] for f in shells]
        self.truth_notes.append("gst_itc_passthrough: shell firms pay almost no GST in cash; credit offsets all output tax")

    # ------------------------------------------------------------ export
    def frames(self) -> dict[str, pd.DataFrame]:
        firms = pd.DataFrame(self.firms).drop(columns=["_shell"])
        tx = pd.DataFrame(self.txns).sort_values(["txn_date", "txn_id"]).reset_index(drop=True)
        return {"persons": pd.DataFrame(self.persons), "firms": firms,
                "accounts": pd.DataFrame(self.accounts), "transactions": tx,
                "itr_filings": pd.DataFrame(self.itr),
                "gst_invoices": pd.DataFrame(self.gst_invoices), "gst_returns": pd.DataFrame(self.gst_returns)}


def generate(out_dir: Path = config.DEMO_DIR, seed: int = config.DEMO_SEED, quiet: bool = False):
    g = Gen(seed)
    g.build()
    frames = g.frames()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    truth = {"seed": seed, "notes": g.truth_notes, "planted": g.truth}
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    if not quiet:
        print(f"Synthetic case written to {out_dir}")
        for name, df in frames.items():
            print(f"  {name:13s} {len(df):>7,} rows")
        print("  planted cases:", {k: len(v) for k, v in g.truth.items()})
    return frames, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic tax case for Kar Drishti")
    ap.add_argument("--seed", type=int, default=config.DEMO_SEED)
    ap.add_argument("--out", type=Path, default=config.DEMO_DIR)
    a = ap.parse_args()
    generate(a.out, a.seed)

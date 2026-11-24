"""Revenue & Demand Tracking: all maths in code (the AI never does arithmetic).

Inputs (any of these, from the demo folder or from "Add data"):
  revenue_targets        fy, region, charge, range, target
  collections            fy, region, charge, range, month (YYYY-MM), amount_collected
  collections_last_year  same columns, previous year, all 12 months
  demand_register        demand_id, taxpayer_name, region, charge, range, date_raised, demand_amount,
                         outstanding, disputed (Yes/No), stay (Yes/No), ...
  recovery_actions       demand_id, date, action
  refunds                region, charge, range, month, refund_amount
Everything is computed at range level and then SUMMED upwards, so totals add up at every level
(range > charge > region > all); consistency() proves it.

Year-end projection uses last year's monthly pattern scaled by this year's run-rate:
  for every month still to come: last year's collection in that month x (collected so far this year /
  collected by the same point last year). A plain run-rate projection is shown alongside.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from .base import rupees

LEVELS = ["region", "charge", "range"]
SYNONYMS = {
    "target": ["target", "target_amount", "budget_target", "annual_target"],
    "amount_collected": ["amount_collected", "collected", "collection", "amount", "collection_amount"],
    "range": ["range", "range_name", "range_id"], "charge": ["charge", "charge_name"],
    "region": ["region", "region_name"], "month": ["month", "period"],
    "outstanding": ["outstanding", "outstanding_demand", "balance"], "demand_amount": ["demand_amount", "demand"],
    "taxpayer_name": ["taxpayer_name", "name", "assessee", "assessee_name"],
    "refund_amount": ["refund_amount", "refund", "amount_refunded"],
}
DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_revenue"
AGE_BUCKETS = [(0, 1, "Under 1 year"), (1, 2, "1-2 years"), (2, 5, "2-5 years"), (5, 99, "Over 5 years")]


def _std(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rename common column spellings to the standard ones."""
    if df is None:
        return None
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for std, alts in SYNONYMS.items():
        if std not in df.columns:
            hit = next((a for a in alts if a in df.columns), None)
            if hit:
                df = df.rename(columns={hit: std})
    return df


def fy_months(fy: str) -> list[str]:
    """'2026-27' -> ['2026-04', ..., '2027-03']."""
    y = int(fy[:4])
    return [f"{y}-{m:02d}" for m in range(4, 13)] + [f"{y + 1}-{m:02d}" for m in range(1, 4)]


@dataclass
class Revenue:
    targets: pd.DataFrame
    collections: pd.DataFrame
    last_year: pd.DataFrame | None = None
    demands: pd.DataFrame | None = None
    actions: pd.DataFrame | None = None
    refunds: pd.DataFrame | None = None
    fy: str = ""
    as_on: str = ""
    label: str = ""                                  # e.g. "Synthetic demo data"
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.targets, self.collections = _std(self.targets), _std(self.collections)
        self.last_year, self.demands = _std(self.last_year), _std(self.demands)
        self.actions, self.refunds = _std(self.actions), _std(self.refunds)
        for col in ("target",):
            self.targets[col] = pd.to_numeric(self.targets[col], errors="coerce").fillna(0).round().astype("int64")
        for df in (self.collections, self.last_year):
            if df is not None:
                df["amount_collected"] = pd.to_numeric(df["amount_collected"], errors="coerce").fillna(0).round().astype("int64")
                df["month"] = df["month"].astype(str).str[:7]
        if not self.fy:
            self.fy = str(self.targets["fy"].iloc[0]) if "fy" in self.targets else self._guess_fy()
        if not self.as_on:
            last = max(self.collections["month"]) if len(self.collections) else fy_months(self.fy)[0]
            y, m = int(last[:4]), int(last[5:7])
            first_of_next = date(y + (m == 12), m % 12 + 1, 1)
            self.as_on = date.fromordinal(first_of_next.toordinal() - 1).isoformat()   # last day of that month
        if self.demands is not None:
            for c in ("demand_amount", "outstanding"):
                if c in self.demands:
                    self.demands[c] = pd.to_numeric(self.demands[c], errors="coerce").fillna(0).round().astype("int64")
            for c in ("disputed", "stay"):
                if c not in self.demands:
                    self.demands[c] = "No"
                self.demands[c] = self.demands[c].astype(str).str.strip().str.title().replace(
                    {"Y": "Yes", "N": "No", "True": "Yes", "False": "No"})

    def _guess_fy(self) -> str:
        first = min(self.collections["month"])
        y = int(first[:4]) - (int(first[5:7]) < 4)
        return f"{y}-{(y + 1) % 100:02d}"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_folder(cls, folder: Path = DEMO_DIR, label: str = "Synthetic demo data") -> "Revenue | None":
        def rd(n):
            f = folder / f"{n}.csv"
            return pd.read_csv(f) if f.exists() else None
        t, c = rd("revenue_targets"), rd("collections")
        if t is None or c is None:
            return None
        meta = {}
        if (folder / "meta.json").exists():
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(t, c, rd("collections_last_year"), rd("demand_register"), rd("recovery_actions"), rd("refunds"),
                   fy=meta.get("fy", ""), as_on=meta.get("as_on", ""), label=label)

    @classmethod
    def from_tables(cls, tables: dict, base: "Revenue | None" = None) -> "Revenue | None":
        """Build from uploaded tables (keyed by kind); anything not uploaded comes from `base` (the demo)."""
        def pick(kind, attr):
            if kind in tables:
                return tables[kind]
            return getattr(base, attr) if base is not None else None
        t, c = pick("revenue targets", "targets"), pick("collections", "collections")
        if t is None or c is None:
            return None
        uploaded = any(k in tables for k in ("revenue targets", "collections", "last year's collections",
                                             "demand register", "recovery actions", "refunds"))
        try:
            return cls(t, c, pick("last year's collections", "last_year"), pick("demand register", "demands"),
                       pick("recovery actions", "actions"), pick("refunds", "refunds"),
                       fy="" if "revenue targets" in tables else (base.fy if base else ""),
                       as_on="" if "collections" in tables else (base.as_on if base else ""),
                       label=("Uploaded files" + (" + demo data" if base is not None else "")) if uploaded
                       else (base.label if base else ""))
        except (KeyError, ValueError) as e:
            if base is not None:
                base.notes.append(f"Uploaded revenue files could not be used ({e}); showing the demo data.")
            return base

    # ------------------------------------------------------------------ time
    @property
    def months(self) -> list[str]:
        return fy_months(self.fy)

    @property
    def months_done(self) -> int:
        """Months with collections loaded (April up to the 'as on' month)."""
        return self.months.index(self.as_on[:7]) + 1 if self.as_on[:7] in self.months else \
            len(set(self.collections["month"]) & set(self.months))

    @property
    def months_left(self) -> int:
        return 12 - self.months_done

    # ------------------------------------------------------------------ filtering
    def _filter(self, df: pd.DataFrame | None, where: dict | None) -> pd.DataFrame | None:
        if df is None or not where:
            return df
        for k, v in where.items():
            if k in df.columns and v:
                df = df[df[k] == v]
        return df

    # ------------------------------------------------------------------ core numbers (range level, then summed)
    def range_table(self) -> pd.DataFrame:
        """One row per range: target, collected, gap, last year's same-period and full-year, projection."""
        keys = [k for k in LEVELS if k in self.targets.columns]
        t = self.targets.groupby(keys, as_index=False)["target"].sum()
        c = self.collections[self.collections["month"].isin(self.months)]
        col = c.groupby(keys, as_index=False)["amount_collected"].sum().rename(columns={"amount_collected": "collected"})
        out = t.merge(col, on=keys, how="left").fillna({"collected": 0})
        out["collected"] = out["collected"].astype("int64")
        done = self.months[:self.months_done]
        if self.last_year is not None and len(self.last_year):
            ly = self.last_year.copy()
            ly["pos"] = ly["month"].map({m: i for i, m in enumerate(fy_months(_prev_fy(self.fy)))})
            ly_done = ly[ly["pos"] < len(done)].groupby(keys, as_index=False)["amount_collected"].sum() \
                .rename(columns={"amount_collected": "ly_same_period"})
            ly_full = ly.groupby(keys, as_index=False)["amount_collected"].sum().rename(
                columns={"amount_collected": "ly_full_year"})
            out = out.merge(ly_done, on=keys, how="left").merge(ly_full, on=keys, how="left").fillna(0)
            out[["ly_same_period", "ly_full_year"]] = out[["ly_same_period", "ly_full_year"]].astype("int64")
            # pattern projection: rest of this year = rest of last year x (this year so far / last year so far)
            rest = out["ly_full_year"] - out["ly_same_period"]
            ratio = out["collected"] / out["ly_same_period"].where(out["ly_same_period"] > 0)
            out["projected"] = (out["collected"] + (rest * ratio.fillna(1.0))).round().astype("int64")
        else:
            out["ly_same_period"] = out["ly_full_year"] = 0
            out["projected"] = (out["collected"] / max(self.months_done, 1) * 12).round().astype("int64")
        out["run_rate_projected"] = (out["collected"] / max(self.months_done, 1) * 12).round().astype("int64")
        return self._derive(out)

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["gap"] = df["target"] - df["collected"]
        df["achieved_pct"] = (100 * df["collected"] / df["target"].where(df["target"] > 0)).round(1).fillna(0)
        left = max(self.months_left, 1)
        df["monthly_pace_needed"] = (df["gap"].clip(lower=0) / left).round().astype("int64")
        df["projected_vs_target"] = df["projected"] - df["target"]
        return df

    def rollup(self, level: str | None = None, where: dict | None = None) -> pd.DataFrame:
        """Totals by 'region' / 'charge' / 'range' (or one 'All' row when level is None), summed from ranges."""
        r = self._filter(self.range_table(), where)
        sums = ["target", "collected", "ly_same_period", "ly_full_year", "projected", "run_rate_projected"]
        if level is None:
            row = r[sums].sum().to_frame().T
            row.insert(0, "level", "All")
            return self._derive(row.astype({c: "int64" for c in sums}))
        keys = LEVELS[:LEVELS.index(level) + 1]
        g = r.groupby(keys, as_index=False)[sums].sum()
        return self._derive(g)

    def monthly(self, where: dict | None = None) -> pd.DataFrame:
        """Month by month: this year, last year, cumulative of both."""
        c = self._filter(self.collections, where)
        m = c.groupby("month", as_index=False)["amount_collected"].sum().rename(columns={"amount_collected": "collected"})
        out = pd.DataFrame({"month": self.months}).merge(m, on="month", how="left")
        if self.last_year is not None:
            ly = self._filter(self.last_year, where).groupby("month", as_index=False)["amount_collected"].sum()
            ly["month"] = ly["month"].map(dict(zip(fy_months(_prev_fy(self.fy)), self.months)))
            out = out.merge(ly.rename(columns={"amount_collected": "last_year"}), on="month", how="left")
        else:
            out["last_year"] = 0
        out["last_year"] = out["last_year"].fillna(0).astype("int64")
        return out

    def summary(self, where: dict | None = None) -> dict:
        r = self.rollup(None, where).iloc[0]
        refunds = int(self._filter(self.refunds, where)["refund_amount"].sum()) if self.refunds is not None else None
        return {"target": int(r.target), "collected": int(r.collected), "gap": int(r.gap),
                "achieved_pct": float(r.achieved_pct), "projected": int(r.projected),
                "run_rate_projected": int(r.run_rate_projected), "monthly_pace_needed": int(r.monthly_pace_needed),
                "months_done": self.months_done, "months_left": self.months_left, "refunds": refunds,
                "ly_same_period": int(r.ly_same_period), "as_on": self.as_on, "fy": self.fy}

    # ------------------------------------------------------------------ demand
    def ageing(self, where: dict | None = None) -> pd.DataFrame:
        d = self._filter(self.demands, where)
        if d is None or not len(d):
            return pd.DataFrame(columns=["age", "demands", "outstanding", "disputed_or_stayed"])
        age = (pd.Timestamp(self.as_on) - pd.to_datetime(d["date_raised"])).dt.days / 365.25
        rows = []
        for lo, hi, label in AGE_BUCKETS:
            sel = d[(age >= lo) & (age < hi) & (d["outstanding"] > 0)]
            rows.append({"age": label, "demands": len(sel), "outstanding": int(sel["outstanding"].sum()),
                         "disputed_or_stayed": int(sel.loc[(sel.disputed == "Yes") | (sel.stay == "Yes"), "outstanding"].sum())})
        return pd.DataFrame(rows)

    def recovery_list(self, where: dict | None = None) -> pd.DataFrame:
        """Outstanding demands ranked: undisputed with no stay first, then largest first. Each row has its reason."""
        d = self._filter(self.demands, where)
        if d is None or not len(d):
            return pd.DataFrame()
        d = d[d["outstanding"] > 0].copy()
        d["clear"] = (d["disputed"] == "No") & (d["stay"] == "No")
        d = d.sort_values(["clear", "outstanding"], ascending=[False, False]).reset_index(drop=True)
        d["rank"] = range(1, len(d) + 1)
        last = {}
        if self.actions is not None and len(self.actions):
            a = self.actions.sort_values("date").groupby("demand_id").last()
            last = {k: f"{v['action']} on {v['date']}" for k, v in a.iterrows()}
        age = ((pd.Timestamp(self.as_on) - pd.to_datetime(d["date_raised"])).dt.days / 365.25).round(1)
        reasons = []
        for row, yrs in zip(d.itertuples(), age):
            status = "undisputed, no stay" if row.clear else \
                ("stayed" if row.stay == "Yes" else "") + (" and " if row.stay == "Yes" and row.disputed == "Yes" else "") + \
                ("disputed in appeal" if row.disputed == "Yes" else "")
            why = (f"{rupees(row.outstanding)} outstanding, {status}, raised {row.date_raised} ({yrs:g} years ago)"
                   + (f"; last action: {last[row.demand_id]}" if row.demand_id in last else "; no action recorded yet"))
            reasons.append(why)
        d["reason"] = reasons
        d["last_action"] = d["demand_id"].map(last).fillna("none")
        cols = ["rank", "demand_id", "taxpayer_name", "range", "outstanding", "disputed", "stay", "date_raised",
                "last_action", "reason"]
        return d[[c for c in cols if c in d.columns]]

    def known_names(self) -> list[str]:
        names = list(self.demands["taxpayer_name"].astype(str).unique()) if self.demands is not None else []
        for lvl in LEVELS:
            if lvl in self.targets:
                names += list(self.targets[lvl].astype(str).unique())
        return names

    # ------------------------------------------------------------------ proof that totals add up
    def consistency(self) -> list[str]:
        """Empty list = every level adds up (range > charge > region > all, and months > year)."""
        problems = []
        r = self.range_table()
        cols = ["target", "collected", "projected", "gap"]
        total = self.rollup(None).iloc[0]
        for level in LEVELS:
            if level not in r.columns:
                continue
            g = self.rollup(level)
            for c in cols:
                if int(g[c].sum()) != int(total[c]):
                    problems.append(f"{c}: {level} totals {g[c].sum():,} ≠ overall {total[c]:,}")
        m = self.monthly()
        if int(m["collected"].fillna(0).sum()) != int(total["collected"]):
            problems.append("monthly collections do not add up to the year so far")
        if int(self.collections[self.collections["month"].isin(self.months)]["amount_collected"].sum()) != int(total["collected"]):
            problems.append("range collections do not add up to the raw file total")
        if int(self.targets["target"].sum()) != int(total["target"]):
            problems.append("range targets do not add up to the raw file total")
        return problems


def _prev_fy(fy: str) -> str:
    y = int(fy[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


def load_demo(regenerate: bool = False) -> Revenue | None:
    if regenerate or not (DEMO_DIR / "revenue_targets.csv").exists():
        from data_generator.generate_revenue import generate
        generate(DEMO_DIR, quiet=True)
    return Revenue.from_folder(DEMO_DIR)

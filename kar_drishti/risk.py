"""
Risk scoring: one explainable 0-100 score per holder.

Every red flag adds points = weight x strength (strength 0-1 = how strong the evidence is).
The score is the sum, capped at 100; any red flag lifts the score to at least RISK_MEDIUM
(so it is always reviewed). Each holder keeps a point-by-point breakdown so the officer can
see exactly where the number comes from. The ML outlier model (Isolation Forest) is only a
hint shown to the officer; it adds no points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from . import gst, link_finder, money_trail
from .data import Case
from .money_trail import inr

PATTERN_LABEL = {
    "structuring": "Cash structuring",
    "round_trip": "Round-tripping",
    "pass_through": "Pass-through / mule account",
    "income_mismatch": "Income mismatch",
    "non_filer": "Non-filer with high-value transactions",
    "round_cash": "Large round-figure cash withdrawals",
    "shell_cluster": "Linked shell-firm cluster",
    "gst_circular": "GST signal: circular trading",
    "gst_itc_passthrough": "GST signal: credit pass-through (no cash tax)",
    "gst_no_eway": "GST signal: invoices without e-way bills",
    "gst_risky_itc": "GST signal: ITC claimed from risky suppliers",
}
GST_PATTERNS = {"gst_circular", "gst_itc_passthrough", "gst_no_eway", "gst_risky_itc"}
OUTLIER_LABEL = "Unusual behaviour (ML hint)"


@dataclass
class Analysis:
    findings: pd.DataFrame
    risk: pd.DataFrame
    flows: pd.DataFrame
    chains: list
    clusters: list
    gst_cycles: list = None
    gst_on: bool = False


def anomaly_scores(case: Case, flows: pd.DataFrame) -> pd.Series:
    """Isolation Forest on behaviour features; returns 0-1 (1 = most unusual)."""
    h = case.holders[["holder_id", "declared_income", "gross_receipts"]].merge(flows, on="holder_id", how="left").fillna(0)
    if len(h) < 10:
        return pd.Series(0.0, index=h.holder_id)
    ref = np.where(h.gross_receipts > 0, h.gross_receipts, h.declared_income).astype(float)
    X = pd.DataFrame({
        "log_cr": np.log1p(h.total_credits), "log_dr": np.log1p(h.total_debits),
        "cash_share": h.cash_deposits / h.total_credits.replace(0, np.nan),
        "credit_to_declared": np.log1p(h.total_credits / np.maximum(ref, 1.0)),
        "avg_txn": np.log1p((h.total_credits + h.total_debits) / h.n_txn.replace(0, np.nan)),
        "counterparties": np.log1p(h.n_counterparties),
    }).fillna(0)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=0)
    model.fit(X)
    raw = -model.score_samples(X)
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return pd.Series(norm, index=h.holder_id)


def score(case: Case, findings: pd.DataFrame, flows: pd.DataFrame, anomaly: pd.Series) -> pd.DataFrame:
    W = config.RISK_WEIGHTS
    h = case.holders.merge(flows, on="holder_id", how="left").fillna(
        {"total_credits": 0, "total_debits": 0, "cash_deposits": 0, "n_txn": 0})
    cutoff = anomaly.quantile(0.90) if len(anomaly) else 1
    rows = []
    for r in h.itertuples(index=False):
        f = findings[findings.holder_id == r.holder_id]
        flags, reasons, breakdown = [], [], []
        for fr in f.itertuples(index=False):
            w = W.get(fr.pattern, 10)
            breakdown.append({"pattern": fr.pattern, "label": PATTERN_LABEL.get(fr.pattern, fr.pattern),
                              "weight": w, "strength": round(float(fr.severity), 2),
                              "points": round(w * float(fr.severity), 1)})
            flags.append(fr.pattern)
            reasons.append(fr.summary)
        points = round(sum(b["points"] for b in breakdown), 1)
        s = float(min(100, points))
        floored = bool(breakdown) and s < config.RISK_MEDIUM
        if floored:                                   # any red flag -> at least MEDIUM (needs review)
            s = float(config.RISK_MEDIUM)
        a = float(anomaly.get(r.holder_id, 0))
        rows.append({"holder_id": r.holder_id, "name": r.name, "holder_type": r.holder_type,
                     "city": r.city, "risk_score": round(s, 1), "points": points, "floored": floored,
                     "risk_level": "HIGH" if s >= config.RISK_HIGH else "MEDIUM" if s >= config.RISK_MEDIUM else "LOW",
                     "flags": sorted(set(flags), key=flags.index), "reasons": reasons, "breakdown": breakdown,
                     "anomaly": round(a, 3), "outlier": bool(a >= cutoff and a > 0),
                     "total_credits": r.total_credits,
                     "total_debits": r.total_debits, "cash_deposits": r.cash_deposits,
                     "declared_income": r.declared_income, "filed": r.filed})
    # rank by the uncapped points, so two holders at 100 are still ordered by the weight of evidence
    return pd.DataFrame(rows).sort_values(["points", "anomaly"], ascending=False).reset_index(drop=True)


def score_text(row) -> str:
    """'Round-tripping 52 + Income mismatch 35 = 87 points -> score 87'."""
    b = row["breakdown"]
    if not b:
        return "No red flags, score 0."
    parts = " + ".join(f"{x['label']} {x['points']:.0f}" for x in b)
    total = row["points"]
    if total > 100:
        end = f"= {total:.0f} points, capped at 100"
    elif row.get("floored"):
        end = f"= {total:.0f} points, raised to {config.RISK_MEDIUM} because every red flag needs review"
    else:
        end = f"= {total:.0f}"
    return f"{parts} {end}" if len(b) > 1 or total > 100 or row.get("floored") else f"{parts} points"


def analyze(case: Case, include_gst: bool | None = None) -> Analysis:
    """Run every check. GST checks are an optional add-on (config.GST_ENABLED or include_gst=True)."""
    if include_gst is None:
        include_gst = config.GST_ENABLED
    include_gst = bool(include_gst) and gst.has_gst(case)
    mt = money_trail.run_all(case)
    clusters = link_finder.find_clusters(case, mt["flows"])
    parts = [mt["findings"], link_finder.detect_shell_clusters(case, clusters)]
    gst_cycles = []
    if include_gst:
        gst_findings, gst_cycles = gst.detect(case)
        parts.append(gst_findings)
    findings = money_trail.concat_findings(parts)
    anomaly = anomaly_scores(case, mt["flows"])
    risk = score(case, findings, mt["flows"], anomaly)
    return Analysis(findings=findings, risk=risk, flows=mt["flows"],
                    chains=mt["round_trip_chains"], clusters=clusters, gst_cycles=gst_cycles, gst_on=include_gst)


def evaluate(analysis: Analysis, truth: dict) -> dict:
    """Compare detections with the planted cases (demo data only)."""
    planted = truth.get("planted", truth)
    if not analysis.gst_on:                     # GST add-on off: only judge the income-tax checks
        planted = {k: v for k, v in planted.items() if k not in GST_PATTERNS}
    per_pattern = {}
    for pat, ids in planted.items():
        found = set(analysis.findings.loc[analysis.findings.pattern == pat, "holder_id"])
        ids = set(ids)
        per_pattern[pat] = {"planted": len(ids), "detected": len(ids & found),
                            "recall": len(ids & found) / len(ids) if ids else None,
                            "missed": sorted(ids - found)}
    all_planted = set().union(*[set(v) for v in planted.values()])
    flagged = set(analysis.risk.loc[analysis.risk.risk_level != "LOW", "holder_id"])
    tp = flagged & all_planted
    return {"per_pattern": per_pattern,
            "overall": {"planted_entities": len(all_planted), "flagged_medium_or_high": len(flagged),
                        "true_positives": len(tp),
                        "recall": len(tp) / len(all_planted) if all_planted else None,
                        "precision": len(tp) / len(flagged) if flagged else None,
                        "false_positives": sorted(flagged - all_planted)}}


def explain_holder_text(row: pd.Series) -> str:
    """Template explanation (used when the local LLM is not running)."""
    if not row["reasons"]:
        return f"{row['name']} shows no suspicious pattern in the available data."
    lines = [f"{row['name']} ({row['holder_id']}) has a risk score of {row['risk_score']:.0f}/100 "
             f"({row['risk_level']}). Total credits {inr(row['total_credits'])}, declared income "
             f"{inr(row['declared_income'])}{'' if row['filed'] else ' (no return filed)'}.",
             f"How the score was built: {score_text(row)}."]
    lines += [f"- {PATTERN_LABEL.get(f, f)}: {r}" for f, r in zip(row["flags"], row["reasons"])]
    return "\n".join(lines)

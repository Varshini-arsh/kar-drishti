"""
Risk scoring: combines rule-based findings with an unsupervised ML anomaly
score (Isolation Forest) into one explainable 0-100 score per holder.
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
    "gst_circular": "GST circular trading",
    "gst_itc_passthrough": "GST credit pass-through (no cash tax)",
    "gst_no_eway": "Invoices without e-way bills",
    "gst_risky_itc": "ITC claimed from risky suppliers",
    "anomaly": "Statistical outlier (ML)",
}


@dataclass
class Analysis:
    findings: pd.DataFrame
    risk: pd.DataFrame
    flows: pd.DataFrame
    chains: list
    clusters: list
    gst_cycles: list = None


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
        pts, flags, reasons = 0.0, [], []
        for fr in f.itertuples(index=False):
            pts += W.get(fr.pattern, 10) * fr.severity
            flags.append(fr.pattern)
            reasons.append(fr.summary)
        a = float(anomaly.get(r.holder_id, 0))
        if a >= cutoff and a > 0:
            pts += W["anomaly"] * a
            flags.append("anomaly")
            reasons.append(f"Behaviour is a statistical outlier compared with other holders "
                           f"(ML anomaly score {a:.2f}, top 10%).")
        s = float(min(100, round(pts, 1)))
        if any(fl != "anomaly" for fl in flags):   # any rule hit -> at least MEDIUM (needs review)
            s = max(s, float(config.RISK_MEDIUM))
        rows.append({"holder_id": r.holder_id, "name": r.name, "holder_type": r.holder_type,
                     "city": r.city, "risk_score": s,
                     "risk_level": "HIGH" if s >= config.RISK_HIGH else "MEDIUM" if s >= config.RISK_MEDIUM else "LOW",
                     "flags": sorted(set(flags), key=flags.index), "reasons": reasons,
                     "anomaly": round(a, 3), "total_credits": r.total_credits,
                     "total_debits": r.total_debits, "cash_deposits": r.cash_deposits,
                     "declared_income": r.declared_income, "filed": r.filed})
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def analyze(case: Case) -> Analysis:
    mt = money_trail.run_all(case)
    clusters = link_finder.find_clusters(case, mt["flows"])
    gst_findings, gst_cycles = gst.detect(case)
    findings = money_trail.concat_findings([mt["findings"], link_finder.detect_shell_clusters(case, clusters),
                                            gst_findings])
    anomaly = anomaly_scores(case, mt["flows"])
    risk = score(case, findings, mt["flows"], anomaly)
    return Analysis(findings=findings, risk=risk, flows=mt["flows"],
                    chains=mt["round_trip_chains"], clusters=clusters, gst_cycles=gst_cycles)


def evaluate(analysis: Analysis, truth: dict) -> dict:
    """Compare detections with the planted cases (demo data only)."""
    planted = truth.get("planted", truth)
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
             f"{inr(row['declared_income'])}{'' if row['filed'] else ' (no return filed)'}."]
    lines += [f"- {PATTERN_LABEL.get(f, f)}: {r}" for f, r in zip(row["flags"], row["reasons"])]
    return "\n".join(lines)

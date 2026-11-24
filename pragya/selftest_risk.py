"""Honest self-test of the risk checks on the synthetic demo data.

Reports, without rounding up: planted cases caught and missed, borderline cases (just over a threshold)
caught, decoys (look suspicious, under every threshold) wrongly flagged, and any other false alarms.
The demo data is synthetic and the cases were planted on purpose, so this shows the rules work as
designed - it is not proof of real-world accuracy.
"""
from __future__ import annotations

import pandas as pd

from kar_drishti import risk


def risk_self_test(case, analysis) -> dict | None:
    truth = case.ground_truth
    if not truth:
        return None
    ev = risk.evaluate(analysis, truth)
    idx = analysis.risk.set_index("holder_id")
    flagged = set(analysis.risk.loc[analysis.risk.risk_level != "LOW", "holder_id"])
    per = pd.DataFrame([{"red flag": risk.PATTERN_LABEL.get(k, k), "planted": v["planted"], "caught": v["detected"],
                         "missed": ", ".join(case.holder_name(m) for m in v["missed"]) or "none"}
                        for k, v in ev["per_pattern"].items()])
    decoys = pd.DataFrame([{"name": case.holder_name(d["holder_id"]), "looks like": d["looks_like"],
                            "why it should not be flagged": d["why_not"],
                            "result": "wrongly flagged" if d["holder_id"] in flagged else "correctly not flagged",
                            "score": float(idx.loc[d["holder_id"], "risk_score"])} for d in truth.get("decoys", [])])
    border = pd.DataFrame([{"name": case.holder_name(b["holder_id"]), "rule": risk.PATTERN_LABEL.get(b["pattern"]),
                            "why it should be flagged": b["why"],
                            "result": "caught" if b["holder_id"] in set(analysis.findings.loc[
                                analysis.findings.pattern == b["pattern"], "holder_id"]) else "missed",
                            "score": float(idx.loc[b["holder_id"], "risk_score"])} for b in truth.get("borderline", [])])
    decoy_ids = {d["holder_id"] for d in truth.get("decoys", [])}
    other_fp = [h for h in ev["overall"]["false_positives"] if h not in decoy_ids]
    false_alarms = pd.DataFrame([{"name": case.holder_name(h), "score": float(idx.loc[h, "risk_score"]),
                                  "flags": ", ".join(risk.PATTERN_LABEL.get(f, f) for f in idx.loc[h, "flags"])}
                                 for h in other_fp])
    o = ev["overall"]
    return {"caught": o["true_positives"], "planted": o["planted_entities"],
            "decoys_total": len(decoys), "decoys_flagged": int((decoys["result"] == "wrongly flagged").sum()) if len(decoys) else 0,
            "borderline_total": len(border), "borderline_caught": int((border["result"] == "caught").sum()) if len(border) else 0,
            "false_alarms": len(other_fp), "precision": o["precision"], "recall": o["recall"],
            "per_pattern": per, "decoys": decoys, "borderline": border, "false_alarm_table": false_alarms,
            "gst_on": analysis.gst_on}

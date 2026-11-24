"""Audit log: one JSON line per event, kept on this computer (data/audit_log.jsonl).

Records every file added (with its SHA-256 fingerprint), every question, every agent
and tool call, every checker result and every approval. Lines are only ever appended.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

AUDIT_FILE = Path(__file__).resolve().parents[1] / "data" / "audit_log.jsonl"
_lock = threading.Lock()


def fingerprint(data: bytes) -> str:
    """SHA-256 of a file's bytes, as 64 hex characters."""
    return hashlib.sha256(data).hexdigest()


def log(event: str, path: Path | None = None, **details) -> dict:
    """Append one event. Never raises: the audit log must not break the app."""
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **details}
    try:
        target = Path(path or AUDIT_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock, target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def read(path: Path | None = None, last: int = 200) -> list[dict]:
    target = Path(path or AUDIT_FILE)
    if not target.exists():
        return []
    lines = target.read_text(encoding="utf-8").splitlines()[-last:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out

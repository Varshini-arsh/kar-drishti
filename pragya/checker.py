"""The Checker: plain code, not AI.

Every number, name, date and document number in an answer must appear in the
tool output (or the quoted document text, or the officer's own question).
Numbers are compared after normalising Indian formats, so
"Rs 1.69 crore" = "₹1.69 Cr" = "1,69,00,000" = "16900000".
A rounded figure passes when it is the correct rounding of a source number
("₹1.69 crore" matches 16,912,345 but not 17,500,000).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

UNITS = {"crore": 1e7, "crores": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
         "thousand": 1e3, "k": 1e3, "million": 1e6, "billion": 1e9}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], 1)}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
       r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

NUM_RE = re.compile(r"(?<![\w.])(?:rs\.?\s*|inr\s*|₹\s*)?(\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|crores?\b|cr\b|lakhs?\b|"
                    r"lacs?\b|l\b|thousand\b|k\b|million\b|billion\b)?", re.I)
DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),                                   # 2026-03-31
    re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"),                           # 31-03-2026, 31/03/2026
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r"\.?,?(?:\s+(\d{4}))?\b", re.I),   # 31 March 2026
    re.compile(r"\b" + _MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?(?:\s+(\d{4}))?\b", re.I),   # March 31, 2026
]
DOC_RE = re.compile(r"\b(?:no\.?|number)\s*:?\s*([A-Za-z]*[\w./-]*\d[\w./-]*)", re.I)
ENTITY_RE = re.compile(r"\b((?:[A-Z][A-Za-z'’&.]+\s){1,3}(?:Traders|Enterprises|Pvt Ltd|Industries|Exports|"
                       r"Commodities|Solutions|Agencies|Infra|Trading Co|Ltd|LLP|Co\.))(?:\s\d+)?\b")


@dataclass
class CheckResult:
    ok: bool
    problems: list = field(default_factory=list)       # plain-English reasons
    counts: dict = field(default_factory=dict)         # how many of each kind were checked

    def summary(self) -> str:
        c = self.counts
        what = ", ".join(f"{v} {k}" for k, v in c.items() if v) or "nothing to check"
        return f"Passed: {what} matched the tool output." if self.ok else \
            "Failed: " + "; ".join(self.problems[:4]) + ("…" if len(self.problems) > 4 else "")


# ---------------------------------------------------------------- numbers
def parse_numbers(text: str) -> list[tuple[float, float]]:
    """All numbers in the text as (value in base units, rounding step), e.g.
    '₹1.69 crore' -> (16900000, 100000); '45%' -> (45, 1)."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        raw, unit = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        if raw.count(".") > 1:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        mult = UNITS.get(unit, 1.0)
        out.append((v * mult, 10 ** -decimals * mult))
    return out


def _source_values(sources: str) -> list[float]:
    vals = [v for v, _ in parse_numbers(sources)]
    # Percent / ratio forms: a source share 0.4523 can be written as 45.2%
    return vals + [v * 100 for v in vals if 0 < abs(v) <= 1]


def number_supported(value: float, step: float, source_vals: list[float]) -> bool:
    tol = step / 2 + 1e-9 * max(1.0, abs(value))
    return any(abs(s - value) <= tol for s in source_vals)


# ---------------------------------------------------------------- dates
def _mon(s: str) -> int:
    return MONTHS[s[:3].lower()]


def parse_dates(text: str) -> list[tuple[int | None, int, int]]:
    """Dates as (year or None, month, day)."""
    out, t = [], text or ""
    for i, pat in enumerate(DATE_PATTERNS):
        for m in pat.finditer(t):
            try:
                if i == 0:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif i == 1:
                    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif i == 2:
                    d, mo, y = int(m.group(1)), _mon(m.group(2)), int(m.group(3)) if m.group(3) else None
                else:
                    mo, d, y = _mon(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None
                date(y or 2000, mo, d)                      # validates the day/month
                out.append((y, mo, d))
            except (ValueError, KeyError):
                continue
    return out


def _date_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for pat in DATE_PATTERNS for m in pat.finditer(text or "")]


# ---------------------------------------------------------------- main check
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("’", "'")).strip().lower()


def check(answer: str, sources: list[str] | str, known_names: list[str] | None = None,
          question: str = "") -> CheckResult:
    """Check `answer` against the tool output `sources` (strings: facts, CSV of records, quoted text).
    `known_names`: names of people/firms in the loaded data (a name from this list must be in the sources).
    The officer's own question also counts as a source (they said it)."""
    src = "\n".join(sources) if isinstance(sources, (list, tuple)) else (sources or "")
    src = src + "\n" + (question or "")
    src_n = _norm(src)
    problems, counts = [], {"numbers": 0, "names": 0, "dates": 0, "document numbers": 0}
    text = answer or ""

    # 1. document numbers ("Circular No. ADM-12/2019")
    masked = text
    for m in DOC_RE.finditer(text):
        token = m.group(1).rstrip(".,;:)")
        counts["document numbers"] += 1
        if _norm(token) not in src_n:
            problems.append(f"document number '{token}' is not in the loaded documents")
        masked = masked.replace(m.group(0), " " * len(m.group(0)))

    # 2. names: any known name, and anything that looks like a firm name
    names = set()
    low = _norm(masked)
    for n in known_names or []:
        if n and re.search(r"(?<![a-z0-9])" + re.escape(_norm(n)) + r"(?![a-z0-9])", low):
            names.add(n)
    names |= {m.group(0).strip() for m in ENTITY_RE.finditer(masked)}
    # keep only the longest of overlapping names ("Kuber Traders" inside "Kuber Traders 2")
    names = {n for n in names if not any(n != o and _norm(n) in _norm(o) for o in names)}
    for n in sorted(names):
        counts["names"] += 1
        if _norm(n) not in src_n:
            problems.append(f"name '{n}' is not in the tool output")
        masked = re.sub(re.escape(n), " " * len(n), masked, flags=re.I)

    # 3. dates
    src_dates = parse_dates(src)
    for y, mo, d in parse_dates(masked):
        counts["dates"] += 1
        if not any(sm == mo and sd == d and (y is None or sy is None or sy == y) for sy, sm, sd in src_dates):
            problems.append(f"date {d:02d}-{mo:02d}{'-' + str(y) if y else ''} is not in the tool output")
    for a, b in sorted(_date_spans(masked), reverse=True):
        masked = masked[:a] + " " * (b - a) + masked[b:]

    # 4. numbers
    src_vals = _source_values(src)
    for v, step in parse_numbers(masked):
        counts["numbers"] += 1
        if not number_supported(v, step, src_vals):
            problems.append(f"number {fmt_plain(v)} is not in the tool output")
    return CheckResult(ok=not problems, problems=problems, counts=counts)


def fmt_plain(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"

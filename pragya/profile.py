"""Officer profile: name, rank and posting.

The posting (with the rank) decides:
  * the 4 starter questions on the Ask screen
  * the signature block printed under every draft

The profile is saved on this computer only (data/officer_profile.json).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import branding

# Short form shown in menus -> full designation printed in the signature block.
RANKS = {
    "Inspector": "Income Tax Inspector",
    "ITO": "Income Tax Officer",
    "ACIT": "Assistant Commissioner of Income Tax",
    "DCIT": "Deputy Commissioner of Income Tax",
    "JCIT": "Joint Commissioner of Income Tax",
    "Addl. CIT": "Additional Commissioner of Income Tax",
    "CIT": "Commissioner of Income Tax",
    "Pr.CIT": "Principal Commissioner of Income Tax",
    "CCIT": "Chief Commissioner of Income Tax",
    "Pr.CCIT": "Principal Chief Commissioner of Income Tax",
}

POSTINGS = ["Assessing Officer", "Range head", "Investigation", "I&CI", "Central Circle", "TRO",
            "Charge head", "Region head", "Admin"]

# Four starter questions per posting, mixing the tasks that posting does most.
# "{top}" is replaced with the riskiest name in the loaded data, "{trail}" with a
# holder whose money can be traced. Questions about data that is not loaded are
# still fine: the assistant says what it can do instead.
STARTERS = {
    "Assessing Officer": ["Which cases should I investigate first?",
                          "Why is {top} flagged?",
                          "Who received more than 10 times their declared income?",
                          "How many casual leave days can be carried forward?"],
    "Range head": ["Which cases in my range need review first?",
                   "How far are we from the revenue target?",
                   "Which undisputed demands should we chase first?",
                   "Show total cash deposits by month"],
    "Investigation": ["Which cases should I investigate first?",
                      "Where did the money from {trail} go?",
                      "Show the shell firm network",
                      "Who are the non-filers with high-value transactions?"],
    "I&CI": ["Who are the non-filers with high-value transactions?",
             "Who made large round-figure cash withdrawals?",
             "Show total cash deposits by month",
             "Which accounts look like pass-through accounts?"],
    "Central Circle": ["Which cases should I investigate first?",
                       "Where did the money from {trail} go?",
                       "Show the shell firm network",
                       "Which undisputed demands should we chase first?"],
    "TRO": ["Which undisputed demands should we chase first?",
            "Show demand ageing",
            "Is Ganesh Commodities' unpaid demand worth chasing, and what does the recovery circular say?",
            "How much has been collected this year?"],
    "Charge head": ["How far are we from the revenue target?",
                    "What is the projected collection by 31 March?",
                    "Which range is furthest behind its target?",
                    "Which cases should be investigated first?"],
    "Region head": ["How far are we from the revenue target?",
                    "What is the projected collection by 31 March?",
                    "Which charge is furthest behind its target?",
                    "Which cases should be investigated first?"],
    "Admin": ["How many casual leave days are allowed?",
              "Is work from home allowed?",
              "What is the daily allowance on tour?",
              "Within how many days must a travel claim be filed?"],
}

PROFILE_FILE = Path(__file__).resolve().parents[1] / "data" / "officer_profile.json"


@dataclass
class Profile:
    name: str = ""
    rank: str = "ITO"
    posting: str = "Assessing Officer"

    @property
    def designation(self) -> str:
        return RANKS.get(self.rank, self.rank)

    @property
    def is_set(self) -> bool:
        return bool(self.name.strip())


def load_profile(path: Path = PROFILE_FILE) -> Profile:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        p = Profile(**{k: d.get(k, v) for k, v in asdict(Profile()).items()})
    except Exception:
        return Profile()
    if p.rank not in RANKS:
        p.rank = "ITO"
    if p.posting not in POSTINGS:
        p.posting = "Assessing Officer"
    return p


def save_profile(p: Profile, path: Path = PROFILE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(p), indent=2), encoding="utf-8")


def starter_questions(p: Profile, top: str = "", trail: str = "") -> list[str]:
    """The 4 starter questions for this officer's posting, with names filled in."""
    out = []
    for q in STARTERS.get(p.posting, STARTERS["Assessing Officer"]):
        if ("{top}" in q and not top) or ("{trail}" in q and not trail):
            q = "Which cases should I investigate first?" if "{top}" in q else "Show the shell firm network"
        out.append(q.format(top=top, trail=trail))
    # never show the same question twice
    seen, unique = set(), []
    for q in out:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:4]


def signature_block(p: Profile) -> str:
    """Signature block for drafts. Always carries the DRAFT mark."""
    lines = [branding.DRAFT_MARK, "", "(" + (p.name.strip() or "Name of officer") + ")", p.designation, p.posting]
    return "\n".join(lines)

"""The Library: circulars, office orders and Acts, split into passages and indexed on this computer.

Search = keyword (BM25 with a small tax-office vocabulary) plus, when the local embedding model
nomic-embed-text is installed in Ollama, meaning-based similarity. Nothing leaves the computer.

Answering rules (from the brief):
  * Answer ONLY from the library, with document number, date and the EXACT quoted sentence(s).
  * If nothing covers the question: "Not stated in the loaded documents".
    Every important word of the question must be found in the quoted passage (so "hotel allowance"
    or "Study Leave" are not answered from the daily-allowance or casual-leave rules).
  * A conflict = two documents give DIFFERENT answers on the SAME point. Silence is not a conflict,
    and two documents saying the same thing is not a conflict.
    On a conflict: show both side by side, mark the later one, and say "An officer must decide which applies".
  * Section look-up between the 1961 and 2025 Acts only from a mapping table file, never guessed.
  * The right Act for a tax year: the Income-tax Act, 2025 applies from 1 April 2026.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .checker import parse_dates

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_library"
NOT_STATED = "Not stated in the loaded documents."
OFFICER_DECIDES = "An officer must decide which applies."
ACT_2025_FROM = date(2026, 4, 1)

# Tax-office vocabulary: each concept and the phrases that mean it.
CONCEPTS = {
    "casual_leave": [r"casual leave", r"\bcl\b"],
    "earned_leave": [r"earned leave", r"\bel\b"],
    "carry_forward": [r"carr(y|ied|ies)[ -]forward", r"\blapse", r"next (calendar )?year"],
    "work_from_home": [r"work(ing)? from home", r"\bwfh\b", r"remote work"],
    "leave_application": [r"appl(y|ication|ications|ied|ying) for leave", r"leave .*in advance", r"in advance.*leave",
                          r"leave .*appl(y|ied|ication)", r"notice .*leave", r"leave .*notice"],
    "tour_advance": [r"advance of", r"(tour|travel) advance", r"advance .*(tour|travel)", r"(tour|travel).* advance"],
    "travel_class": [r"travel by", r"class of travel", r"\bac\b.*tier", r"train class", r"which class"],
    "daily_allowance": [r"daily allowance", r"\bda\b"],
    "travel_claim": [r"travel claims?", r"claims? .*(submit|file|filed)", r"(submit|file) .*claims?"],
}
# words that belong to a concept: once the concept is recognised they need not appear word-for-word
CONCEPT_WORDS = {"leave_application": {"notice", "apply", "advance", "prior", "leave", "submit"},
                 "travel_claim": {"claim", "travel", "submit", "bill", "reimbursement"},
                 "tour_advance": {"advance", "tour", "travel"}, "daily_allowance": {"allowance", "da", "tour"},
                 "travel_class": {"class", "travel", "train"}, "carry_forward": {"carry", "forward", "lapse", "unused"},
                 "casual_leave": {"casual", "leave", "cl"}, "earned_leave": {"earned", "leave", "el"},
                 "work_from_home": {"work", "home", "wfh", "remote"}}
STOP = set("""a an the of to in on at by for and or with from is are was were be been being do does did can could may might
must shall should will would what which who whom whose how many much when where why this that these those it its as
about any all per under up i we you my our your me us there their they he she his her than then so if not no yes
tell say says said please give show allowed allow permitted permit entitled officer officers staff rule rules
circular circulars office order document documents loaded day days number maximum minimum get take taken avail
need needed required require requirement long often kind type mention mentioned rule policy provision""".split())


# word forms that mean the same thing in these documents
SYNONYMS = {"applied": "apply", "application": "apply", "applying": "apply", "submitted": "submit",
            "submission": "submit", "filed": "submit", "file": "submit", "filing": "submit", "deadline": "within",
            "cap": "maximum", "limit": "maximum", "ceiling": "maximum", "annual": "year", "yearly": "year",
            "accumulate": "accumulated", "permissible": "permitted",
            "paid": "payable", "pay": "payable", "payment": "payable"}


def _stem(w: str) -> str:
    """Very light stemming: plural s, -ed, -ing (applied the same way to documents and questions)."""
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    w = SYNONYMS.get(w, w)
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


STOP = STOP | {_stem(w) for w in STOP}
CONCEPT_WORDS = {c: ws | {_stem(w) for w in ws} for c, ws in CONCEPT_WORDS.items()}


def concepts_in(text: str) -> set[str]:
    low = text.lower()
    return {c for c, pats in CONCEPTS.items() if any(re.search(p, low) for p in pats)}


@dataclass
class Passage:
    doc_id: str
    text: str            # exact sentence as written in the document
    pos: int


@dataclass
class Document:
    doc_id: str
    title: str
    number: str
    date: str            # ISO date, "" if not found
    kind: str
    text: str
    source: str = "demo"
    passages: list = field(default_factory=list)

    @property
    def date_obj(self) -> date | None:
        try:
            return date.fromisoformat(self.date)
        except ValueError:
            return None

    def label(self) -> str:
        return self.title or self.doc_id


HEADER = re.compile(r"^(fictitious demo document|\(sd/?-?\)|dated?:|subject:|.*\bno\.\s*[A-Z0-9])", re.I)


def parse_document(doc_id: str, text: str, kind: str = "circular / office order", source: str = "demo") -> Document:
    num = re.search(r"\bNo\.?\s*:?\s*([A-Z0-9][A-Za-z0-9/.\-]*\d[A-Za-z0-9/.\-]*)", text)
    dates = [d for d in parse_dates(text[:600]) if d[0]]
    d = f"{dates[0][0]:04d}-{dates[0][1]:02d}-{dates[0][2]:02d}" if dates else ""
    subj = re.search(r"^\s*Subject:\s*(.+)$", text, re.M | re.I)
    title = subj.group(1).strip() if subj else (text.strip().splitlines() or [doc_id])[0][:80]
    doc = Document(doc_id, title, num.group(1).rstrip(".") if num else "", d, kind, text, source)
    pos = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or HEADER.match(line):
            continue
        line = re.sub(r"^\(?\d{1,3}[.)]\s+|^[ivx]{1,5}[.)]\s+|^[-•*]\s+", "", line)     # drop list numbering
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z])", line):
            if len(_tokens(sent)) >= 3:
                doc.passages.append(Passage(doc_id, sent.strip(), pos))
                pos += 1
    return doc


class Library:
    """Documents + a local index. embed_url: Ollama address for nomic-embed-text (optional)."""

    def __init__(self, docs: list[Document] | None = None, section_map: pd.DataFrame | None = None,
                 embed_url: str | None = None, embed_model: str = "nomic-embed-text"):
        self.docs: dict[str, Document] = {}
        self.section_map = section_map
        self.embed_url, self.embed_model = embed_url, embed_model
        self.embeddings: dict[tuple, list] = {}
        self.embed_status = "Keyword search only (embedding model not connected)."
        for d in docs or []:
            self.add(d)

    # ------------------------------------------------------------------ building
    def add(self, doc: Document):
        self.docs[doc.doc_id] = doc
        self._index()

    def _index(self):
        self.passages = [p for d in self.docs.values() for p in d.passages]
        self.tf = [Counter(_tokens(p.text)) for p in self.passages]
        n = max(len(self.passages), 1)
        df = Counter(t for tf in self.tf for t in tf)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg_len = sum(sum(tf.values()) for tf in self.tf) / n if self.tf else 1
        self.vocab = set(df)

    @classmethod
    def from_folder(cls, folder: Path = DEMO_DIR, **kw) -> "Library":
        docs = [parse_document(f.stem, f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.txt"))
                if f.name.lower() != "readme.txt"]
        smap = folder / "section_map_1961_2025.csv"
        return cls(docs, pd.read_csv(smap) if smap.exists() else None, **kw)

    def ready(self) -> bool:
        return bool(self.passages)

    # ------------------------------------------------------------------ optional embeddings (local Ollama only)
    def try_embeddings(self) -> bool:
        if not self.embed_url or not self.passages:
            return False
        try:
            r = requests.post(f"{self.embed_url}/api/embed", json={"model": self.embed_model,
                                                                  "input": [p.text for p in self.passages]}, timeout=30)
            r.raise_for_status()
            vecs = r.json()["embeddings"]
            self.embeddings = {(p.doc_id, p.pos): v for p, v in zip(self.passages, vecs)}
            self.embed_status = f"Keyword + meaning search ({self.embed_model}, {len(vecs)} passages)."
            return True
        except Exception:
            self.embeddings = {}
            self.embed_status = (f"Keyword search only ({self.embed_model} not available; install it yourself with "
                                 f"'ollama pull {self.embed_model}' to add meaning-based search).")
            return False

    def _embed_query(self, q: str) -> list | None:
        if not self.embeddings:
            return None
        try:
            r = requests.post(f"{self.embed_url}/api/embed", json={"model": self.embed_model, "input": [q]}, timeout=15)
            return r.json()["embeddings"][0]
        except Exception:
            return None

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 8) -> list[tuple[float, Passage]]:
        q = [t for t in _tokens(query) if t not in STOP]
        qv = self._embed_query(query)
        scored = []
        for p, tf in zip(self.passages, self.tf):
            ln = sum(tf.values())
            s = sum(self.idf.get(t, 0) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * ln / self.avg_len)) for t in q if t in tf)
            if qv is not None and (p.doc_id, p.pos) in self.embeddings:
                v = self.embeddings[(p.doc_id, p.pos)]
                dot = sum(a * b for a, b in zip(qv, v))
                norm = math.sqrt(sum(a * a for a in qv)) * math.sqrt(sum(b * b for b in v)) or 1
                s += 2.0 * max(0.0, dot / norm - 0.5)
            if s > 0:
                scored.append((round(s, 4), p))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id, x[1].pos))
        return scored[:k]

    # ------------------------------------------------------------------ answering
    def content_terms(self, question: str) -> list[str]:
        """Important words of the question (not question words, not generic words, not concept phrases)."""
        low = question.lower()
        found = concepts_in(question)
        for pats in CONCEPTS.values():
            for p in pats:
                low = re.sub(p, " ", low)
        own = set().union(*[CONCEPT_WORDS.get(c, set()) for c in found]) if found else set()
        return [t for t in _tokens(low) if t not in STOP and t not in own and not t.isdigit() and len(t) > 1]

    def best_passage(self, doc: Document, question: str, q_concepts: set, terms: list[str]) -> Passage | None:
        best, best_key = None, None
        for p in doc.passages:
            pc = concepts_in(p.text)
            ptoks = set(_tokens(p.text))
            if q_concepts:
                if not q_concepts <= pc:
                    continue
                extra = len(pc - q_concepts)
            else:
                if not terms or not set(terms) <= ptoks:
                    continue
                extra = 0
            if any(t not in ptoks for t in terms):            # every important word must be in the passage
                continue
            overlap = len(set(_tokens(question)) & ptoks)
            key = (extra, -overlap, p.pos)
            if best_key is None or key < best_key:
                best, best_key = p, key
        return best

    def answer(self, question: str) -> dict:
        """{status: 'answer'|'same'|'conflict'|'not stated'|'act'|'section', quotes: [...], text: str}"""
        special = self.act_or_section(question)
        if special:
            return special
        q_concepts = concepts_in(question)
        terms = self.content_terms(question)
        unknown = [t for t in terms if t not in self.vocab]
        if unknown or (not q_concepts and not terms):
            return {"status": "not stated", "quotes": [], "text": NOT_STATED,
                    "why": ("Not found in any loaded document: " + ", ".join(unknown)) if unknown else
                    "The question does not name anything the documents cover."}
        hits = []
        for doc in self.docs.values():
            p = self.best_passage(doc, question, q_concepts, terms)
            if p:
                hits.append((doc, p))
        if not hits:
            return {"status": "not stated", "quotes": [], "text": NOT_STATED,
                    "why": "No passage in the loaded documents covers this point."}
        hits.sort(key=lambda h: (h[0].date_obj or date.min, h[0].doc_id))
        quotes = [{"doc": d.label(), "number": d.number, "date": _fmt_date(d.date), "text": p.text,
                   "doc_id": d.doc_id} for d, p in hits]
        sigs = {signature(p.text) for _, p in hits}
        if len(hits) == 1:
            q = quotes[0]
            text = f"{q['doc']} (No. {q['number']}, dated {q['date']}) says: “{q['text']}”"
            return {"status": "answer", "quotes": quotes, "text": text}
        if len(sigs) == 1:
            text = ("The loaded documents say the same on this point (not a conflict):\n"
                    + "\n".join(f"- No. {q['number']} ({q['date']}): “{q['text']}”" for q in quotes))
            return {"status": "same", "quotes": quotes, "text": text}
        quotes[-1]["later"] = True
        later = quotes[-1]
        text = ("The loaded documents give different answers on this point:\n"
                + "\n".join(f"- No. {q['number']} ({q['date']}): “{q['text']}”" for q in quotes)
                + f"\nThe later document is No. {later['number']} dated {later['date']}. {OFFICER_DECIDES}")
        return {"status": "conflict", "quotes": quotes, "text": text}

    # ------------------------------------------------------------------ Acts: which Act, section mapping
    def act_or_section(self, question: str) -> dict | None:
        low = question.lower()
        sec = re.search(r"\bsec(?:tion|\.)?\s*(\d+[a-z]{0,3})\b", low)
        if sec and re.search(r"1961|2025|new act|old act|correspond|equivalent|mapp|maps to|same as", low):
            s = sec.group(1).upper()
            after, before = low[sec.end():sec.end() + 30], low[:sec.start()]
            if re.search(r"1961|old act", after):
                from_col = "section_1961"
            elif re.search(r"2025|new act", after):
                from_col = "section_2025"
            else:       # no Act right after the number: the Act named last before it, else the 1961 Act
                last = max([(before.rfind(k), c) for k, c in [("1961", "section_1961"), ("old act", "section_1961"),
                                                               ("2025", "section_2025"), ("new act", "section_2025")]])
                from_col = last[1] if last[0] >= 0 else "section_1961"
            to_col = "section_1961" if from_col == "section_2025" else "section_2025"
            m = self.section_map
            if m is None or m.empty:
                return {"status": "not stated", "quotes": [], "text": "Not stated in the loaded documents: no section "
                        "mapping table is loaded, and Pragya never guesses a section. Add the mapping table with "
                        "Add data (columns section_1961, section_2025)."}
            row = m[m[from_col].astype(str).str.upper() == s]
            if row.empty:
                return {"status": "not stated", "quotes": [], "text": f"Not stated in the loaded mapping table: "
                        f"section {s} is not listed. Pragya never guesses a section."}
            r = row.iloc[0]
            text = (f"Section {s} of the Income-tax Act, {'1961' if from_col == 'section_1961' else '2025'} maps to "
                    f"section {r[to_col]} of the Income-tax Act, {'2025' if to_col == 'section_2025' else '1961'}"
                    + (f" ({r['subject']})" if 'subject' in r and pd.notna(r['subject']) else "")
                    + f", as per the loaded mapping table" + (f" ({r['source_document']})" if 'source_document' in r
                                                                and pd.notna(r['source_document']) else "") + ".")
            return {"status": "section", "quotes": [], "text": text}
        if re.search(r"which act|what act|act appl|act (will|should|to) (apply|use)|governed by|old act or new", low):
            years = re.findall(r"\b(20\d\d)\s*[-–/]\s*(\d{2,4})\b", question)
            if years:
                start = int(years[0][0])
                fy_start = date(start, 4, 1)
                act = "Income-tax Act, 2025" if fy_start >= ACT_2025_FROM else "Income-tax Act, 1961"
                text = (f"For {years[0][0]}-{years[0][1][-2:]} (starting {fy_start.day} {fy_start:%B %Y}) the {act} applies. "
                        "Rule used: the Income-tax Act, 2025 applies from 1 April 2026; earlier years remain under "
                        "the Income-tax Act, 1961.")
            else:
                text = ("The Income-tax Act, 2025 applies from 1 April 2026 (tax year 2026-27 onwards); earlier years "
                        "remain under the Income-tax Act, 1961. Name the year to get a specific answer.")
            return {"status": "act", "quotes": [], "text": text}
        return None


def signature(text: str) -> tuple:
    """What a passage says, reduced to its numbers and whether it forbids something.
    Two passages with the same signature give the same answer (no conflict)."""
    low = text.lower()
    nums = tuple(sorted({float(x) for x in re.findall(r"\d+(?:\.\d+)?", low)}))
    negative = bool(re.search(r"\bnot\b|\bno\b|cannot|shall lapse|prohibit|forbidden|never", low))
    return nums, negative


def _fmt_date(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
        return f"{d.day} {d:%B %Y}"
    except ValueError:
        return iso or "date not stated"

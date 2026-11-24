"""Law agent - Task 3, Circular & Law Desk. Answers ONLY from the loaded documents (pragya/library.py).

No model call: the answer is the exact quoted sentence(s) with document number and date,
"Not stated in the loaded documents", or - on a conflict - both quotes side by side with the later
one marked and "An officer must decide which applies". The same question always gets the same answer.
"""
from __future__ import annotations

from .base import AgentResult
from .library import Library


class LawAgent:
    name = "Law agent"

    def __init__(self, library: Library):
        self.lib = library

    def ready(self) -> bool:
        return self.lib.ready()

    def run(self, question: str) -> AgentResult:
        a = self.lib.answer(question)
        n_docs, n_pass = len(self.lib.docs), len(self.lib.passages)
        checked = f"{n_docs} documents, {n_pass} passages · {self.lib.embed_status.split(' (')[0]}"
        tool = {"answer": "quote", "same": "quote (documents agree)", "conflict": "conflict check",
                "not stated": "search: not stated", "act": "Act for the tax year", "section": "section mapping table"}
        steps = {"answer": "Found the passage that covers this point in one document.",
                 "same": "Found the same rule in more than one document; they agree, so this is not a conflict.",
                 "conflict": "Two documents give different answers on the same point; showing both, later one marked.",
                 "not stated": a.get("why", "No loaded document covers this."),
                 "act": "Applied the rule: Income-tax Act, 2025 from 1 April 2026.",
                 "section": "Looked up the loaded section mapping table."}[a["status"]]
        facts = [a["text"]] + [f"No. {q['number']} dated {q['date']} ({q['doc']}): {q['text']}" for q in a["quotes"]]
        return AgentResult(self.name, tool[a["status"]], ok=True, facts=facts, template=a["text"], checked=checked,
                           quotes=a["quotes"], steps=[steps], extra={"status": a["status"]})

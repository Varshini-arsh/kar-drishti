"""
Example bank for the SQL agent (few-shot retrieval).

Small local models write much better SQL when they see a worked example close to
the officer's question. For every question the agent picks the most similar
examples from this list and shows them to the model before asking.
Add new (question, SQL) pairs here whenever officers ask a new kind of question.
"""

EXAMPLES = [
    ("Top 5 people by total cash deposits",
     """SELECT h.holder_id, h.name, ROUND(SUM(t.amount)) AS total_cash_deposits, COUNT(*) AS deposits
FROM transactions t JOIN holders h USING (holder_id)
WHERE t.mode = 'CASH' AND t.direction = 'CR' AND h.holder_type = 'PERSON'
GROUP BY h.holder_id, h.name ORDER BY total_cash_deposits DESC LIMIT 5"""),

    ("Show total cash withdrawals by month",
     """SELECT strftime(t.txn_date, '%Y-%m') AS month, ROUND(SUM(t.amount)) AS total_cash_withdrawals,
       COUNT(*) AS withdrawals
FROM transactions t
WHERE t.mode IN ('CASH', 'ATM') AND t.direction = 'DR'
GROUP BY month ORDER BY month"""),

    ("Who received more than 10 times their declared income?",
     """SELECT h.holder_id, h.name, h.declared_income, ROUND(SUM(t.amount)) AS total_credits,
       ROUND(SUM(t.amount) / NULLIF(h.declared_income, 0), 1) AS times_income
FROM transactions t JOIN holders h USING (holder_id)
WHERE t.direction = 'CR' AND h.filed
GROUP BY h.holder_id, h.name, h.declared_income
HAVING SUM(t.amount) > 10 * h.declared_income ORDER BY times_income DESC LIMIT 50"""),

    ("Top 10 holders by money sent through NEFT",
     """SELECT h.holder_id, h.name, ROUND(SUM(t.amount)) AS total_neft_sent, COUNT(*) AS payments
FROM transactions t JOIN holders h USING (holder_id)
WHERE t.mode = 'NEFT' AND t.direction = 'DR'
GROUP BY h.holder_id, h.name ORDER BY total_neft_sent DESC LIMIT 10"""),

    ("Which people made more than 3 cash deposits between 20,000 and 30,000?",
     """SELECT h.holder_id, h.name, COUNT(*) AS deposits_in_range, ROUND(SUM(t.amount)) AS total
FROM transactions t JOIN holders h USING (holder_id)
WHERE t.mode = 'CASH' AND t.direction = 'CR' AND t.amount >= 20000 AND t.amount < 30000
GROUP BY h.holder_id, h.name HAVING COUNT(*) > 3 ORDER BY deposits_in_range DESC LIMIT 50"""),

    ("Non-filers who received more than 20 lakh",
     """SELECT h.holder_id, h.name, ROUND(SUM(t.amount)) AS total_credits
FROM transactions t JOIN holders h USING (holder_id)
WHERE t.direction = 'CR' AND NOT h.filed
GROUP BY h.holder_id, h.name HAVING SUM(t.amount) > 2000000 ORDER BY total_credits DESC"""),

    ("Which firms paid less than 5% of their GST liability in cash?",
     """SELECT g.firm_id, h.name, ROUND(SUM(g.output_tax)) AS output_tax, ROUND(SUM(g.itc_claimed)) AS itc_claimed,
       ROUND(SUM(g.tax_paid_cash)) AS paid_in_cash,
       ROUND(100 * SUM(g.tax_paid_cash) / NULLIF(SUM(g.output_tax), 0), 2) AS cash_paid_percent
FROM gst_returns g JOIN holders h ON h.holder_id = g.firm_id
GROUP BY g.firm_id, h.name
HAVING SUM(g.tax_paid_cash) < 0.05 * SUM(g.output_tax) ORDER BY cash_paid_percent LIMIT 50"""),

    ("Total ITC claimed by each firm",
     """SELECT g.firm_id, h.name, ROUND(SUM(g.itc_claimed)) AS total_itc_claimed
FROM gst_returns g JOIN holders h ON h.holder_id = g.firm_id
GROUP BY g.firm_id, h.name ORDER BY total_itc_claimed DESC LIMIT 50"""),

    ("Which sellers issued the most invoices without an e-way bill?",
     """SELECT i.seller_firm_id, h.name, COUNT(*) AS invoices_without_eway,
       ROUND(SUM(i.taxable_value)) AS value_without_eway
FROM gst_invoices i JOIN holders h ON h.holder_id = i.seller_firm_id
WHERE i.eway_bill_no IS NULL
GROUP BY i.seller_firm_id, h.name ORDER BY invoices_without_eway DESC LIMIT 50"""),

    ("Which accounts received money from both Ramesh Traders and Suresh Exports?\n"
     "KNOWN IDS: Ramesh Traders = F0007 (FIRM); Suresh Exports = F0012 (FIRM)",
     """SELECT t.account_id, h.name AS account_holder, COUNT(DISTINCT t.counterparty_holder_id) AS senders,
       ROUND(SUM(t.amount)) AS total_received
FROM transactions t JOIN holders h USING (holder_id)
WHERE t.direction = 'CR' AND t.counterparty_holder_id IN ('F0007', 'F0012')
GROUP BY t.account_id, h.name
HAVING COUNT(DISTINCT t.counterparty_holder_id) = 2 ORDER BY total_received DESC"""),

    ("Show all money transferred from Anil Kumar to Meena Enterprises\n"
     "KNOWN IDS: Anil Kumar = P0042 (PERSON); Meena Enterprises = F0019 (FIRM)",
     """SELECT t.txn_date, t.account_id, t.mode, ROUND(t.amount) AS amount, t.narration
FROM transactions t
WHERE t.holder_id = 'P0042' AND t.counterparty_holder_id = 'F0019' AND t.direction = 'DR'
ORDER BY t.txn_date"""),

    ("Largest 10 credits received by Kavita Rao\nKNOWN IDS: Kavita Rao = P0105 (PERSON)",
     """SELECT t.txn_date, t.account_id, t.mode, ROUND(t.amount) AS amount, t.counterparty_name, t.narration
FROM transactions t
WHERE t.holder_id = 'P0105' AND t.direction = 'CR'
ORDER BY t.amount DESC LIMIT 10"""),

    ("List the high risk holders with their flags",
     """SELECT holder_id, name, holder_type, risk_score, flags
FROM risk_scores WHERE risk_level = 'HIGH' ORDER BY risk_score DESC"""),

    ("Month-wise money in and out for Lata Shah\nKNOWN IDS: Lata Shah = P0063 (PERSON)",
     """SELECT strftime(t.txn_date, '%Y-%m') AS month,
       ROUND(SUM(CASE WHEN t.direction = 'CR' THEN t.amount ELSE 0 END)) AS money_in,
       ROUND(SUM(CASE WHEN t.direction = 'DR' THEN t.amount ELSE 0 END)) AS money_out
FROM transactions t WHERE t.holder_id = 'P0063'
GROUP BY month ORDER BY month"""),

    ("How many accounts does each person have? Show those with more than 2",
     """SELECT h.holder_id, h.name, COUNT(*) AS accounts, STRING_AGG(a.bank, ', ') AS banks
FROM accounts a JOIN holders h USING (holder_id)
WHERE h.holder_type = 'PERSON'
GROUP BY h.holder_id, h.name HAVING COUNT(*) > 2 ORDER BY accounts DESC"""),
]

STOP = set("a an the of to in by for and or with from their his her its is are was were did do does "
           "show list give me all each which who what how many much more than less that this those".split())


def _words(text: str) -> set[str]:
    import re
    out = set()
    for w in re.findall(r"[a-z0-9%]+", text.lower()):
        if w in STOP:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)   # crude singular
    return out


def pick_examples(question: str, k: int = 4) -> list[tuple[str, str]]:
    """Return the k examples whose questions share the most words with `question`."""
    qw = _words(question)
    scored = []
    for i, (q, s) in enumerate(EXAMPLES):
        ew = _words(q.split("\n")[0])
        score = len(qw & ew) / (len(ew) ** 0.5 or 1)
        scored.append((score, -i, q, s))
    scored.sort(reverse=True)
    return [(q, s) for _, _, q, s in scored[:k]]

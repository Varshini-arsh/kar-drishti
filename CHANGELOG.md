# Changelog

Plain-English notes, one line per change.

## Step 1 - Pragya rebrand, 4 screens, Settings, officer profile
- Kar Drishti is now **Pragya** ("Insight for every tax desk"); name, tagline, logo and colour live only in `branding.py`.
- Four screens: Ask (home), Cases, Revenue, Library, plus a small Settings link in the sidebar.
- Cases opens a case with tabs Summary, Evidence, Money trail, Links, GST (only when the add-on is on) and Report.
- Officer profile (name, rank, posting) in Settings; it picks the 4 starter questions and the signature block on drafts.
- Every report is marked DRAFT and ends with the officer's signature block.
- Inter font served from `static/` (works offline); light and dark mode (menu ⋮ > Settings > Theme).
- Charts, cards and badges now read well on both light and dark backgrounds.
- Settings shows the model in use and free memory; the old "Load data" page moved to Settings > Data.

## Step 2 - Multi-agent core
- One chat box, a small team behind it: Lead agent picks specialists and merges their answers.
- Risk agent wraps the existing Kar Drishti tools (ranking, profile, money trail, links, GST signals) and adds complete red-flag lists ("showing N of M" when long).
- Data agent answers questions on any loaded table with read-only SQL; common questions work even without the AI model.
- Checker (plain code) verifies every number, name, date and document number against the tool output, reading "Rs 1.69 crore" = "1,69,00,000"; a failed answer gets one retry, then a template answer built from the tool output.
- At most 2 model calls per question; temperature 0, fixed seed, thinking off; qwen3.5:4b default, qwen2.5:3b fallback, qwen3.5:9b only with 8 GB free; never downloads a model.
- Out-of-scope questions get "I can't answer this from the loaded data. Here is what I can do: ..." - never a made-up answer.
- "How I got this" under each answer: agents, tools, records checked, SQL, model calls and the checker result.
- Audit log (data/audit_log.jsonl) records every question, agent and answer.

## Step 3 - Universal upload and audit log
- One "Add data" button (sidebar) accepts CSV, XLSX/XLS (every sheet), JSON, TXT, DOCX and PDF; a scanned PDF says "Scanned PDF: text could not be read".
- Each file is recognised (bank statement, holder list, ITR summary, GST invoices/returns, targets, collections, last year's collections, demand register, recovery actions, refunds, circular/office order, Act text, other table) and shown as "Recognised as: X" with a dropdown to correct it.
- Every table, recognised or not, goes into DuckDB so it can be questioned in plain English; simple questions (totals, counts, "by" groupings, top/lowest, a named value) work even without the AI model.
- SHA-256 fingerprint and an audit-log entry for every file; corrections, profile saves and draft reports are logged as approvals. Settings > Data lists files (with Remove) and the last 100 audit events.
- Added files are kept in data/uploads/ (not in the repo) and reload on restart. New requirement: pypdf (and xlrd for old .xls).

## Step 4 - Revenue agent and Revenue screen
- Synthetic demo revenue files (data/demo_revenue, clearly labelled fake): targets, collections, last year's collections, demand register, recovery actions, refunds for 2 regions > 4 charges > 12 ranges.
- Revenue engine (pragya/revenue.py) computes everything in code at range level and sums upwards, so totals add up range > charge > region > all; a built-in check proves it on screen.
- Shows target, collected, gap, required monthly pace to 31 March, and a year-end projection from last year's monthly pattern scaled by this year's run-rate (plain run-rate shown too).
- Drill-down region > charge > range > month, demand ageing, and a recovery list (undisputed with no stay first, then largest) with the reason for each rank.
- Revenue agent answers target/gap, projection, "which range is behind", monthly, ageing, recovery list, one taxpayer's demand ("is X worth chasing?") and refunds.
- Uploaded revenue files (Add data) replace the matching demo file.

## Step 5 - Law agent and Library
- Library (pragya/library.py): documents are split into passages and indexed on this computer - keyword search with a small tax-office vocabulary, plus meaning-based search when nomic-embed-text is installed in Ollama.
- Law agent answers only from the library: exact quoted sentence(s) with document number and date, or "Not stated in the loaded documents" (every important word of the question must be in the passage, so "hotel allowance" and "Study Leave" are not answered).
- Conflicts = two documents giving different answers on the same point; shown side by side with the later one marked and "An officer must decide which applies". Two documents saying the same thing (apply 3 working days in advance) is not a conflict. Same question, same answer, every time.
- Names the right Act for a tax year (Income-tax Act, 2025 from 1 April 2026). Section look-up 1961 <-> 2025 only from a mapping table file (an empty template ships; Pragya never guesses).
- Three FICTITIOUS demo circulars (2019 leave, 2022 revised leave, 2021 travel), clearly marked as not real.
- Library screen: loaded documents, mapping status and a test-search box. Cross-task questions (e.g. Ganesh Commodities' demand + the recovery circular) call both agents and merge.

## Step 6 - Risk agent upgrades
- Demo data now has 5 decoys (look a little suspicious but stay under every rule's threshold) and 3 borderline cases (exactly at a rule's minimum); they are added last, so all earlier demo cases are unchanged.
- Cases > "Self-test on demo data": planted cases caught, borderline caught, decoys wrongly flagged, and any other false alarms - reported as they are, with a plain note that this is not real-world accuracy.
- Lists are complete, or say "showing N of M"; the review queue states how many of all people and firms need review.
- GST findings are labelled "GST signal: ..." and answers say "GST signals for income-tax risk".

## Step 7 - Self-test (60+ questions)
- `python -m pragya.selftest` asks 66 questions across all agents (risk, data, revenue, law, cross-task, out-of-scope) plus revenue-total, checker and fake-model checks, and prints a pass/fail table.
- Includes the traps ("hotel allowance", "Study Leave" -> Not stated), cross-task questions using two agents, each conflict question asked 3 times (answers must match), and revenue totals that must add up.
- Every answer must also pass the Checker; the command exits with an error if anything fails ("DO NOT SHIP").
- `--model qwen3.5:4b` runs the same questions with the real local model and prints average and slowest answer times.
- Data questions that match a worked example skip the model call (faster and always the same answer).
- README rewritten for Pragya.

## Design update - white and minimal
- Always white (dark mode removed at the officer's request); white sidebar with five links, Add data and one quiet status line.
- Quieter headings, small uppercase section labels, thinner borders, charts without frames, fewer captions.

## Officer profile in the sidebar, and general answers
- Sidebar "Officer profile": Name, Designation, Office and Domain (posting); saved on this computer and printed on every DRAFT signature. Settings keeps Model, Data and a Signature preview.
- Questions outside the loaded data now get a clearly labelled general answer from the local model ("General answer · not from your loaded data ... verify it"), never presented as checked. Without a model the "I can't answer..." message stays.
- Office-rule questions not covered by the loaded circulars (e.g. hotel allowance) still say "Not stated in the loaded documents" - no general guess.

## Lessons from the first real-model run (qwen2.5:3b: 62/78, average 13.2 s)
- Each agent now lists the key facts an answer must contain; a model answer that drops any of them gets one retry, then the code-written answer is shown.
- Questions clearly about a loaded table go straight to the Data agent (no plan call), so the model can no longer wrongly refuse them.
- If the model's SQL fails, the Data agent falls back to its built-in queries.

## Second real-model run (qwen2.5:3b: 70/80, average 12.6 s) - fixes
- In mixed answers the Law agent's part (exact quotes or "Not stated") is always shown word for word; the model only writes the other parts.
- Fixed lines are always kept (e.g. "The officer decides the action.", the ageing total, the highest month).
- Count / total / average questions on a table use the exact built-in query first (no model call), including "people and firms" style questions.
- Greeting shows "A. Sharma" instead of "A.."

- OCR update: scanned PDFs and photos of documents (PNG/JPG/TIFF) are now read with offline text recognition (RapidOCR + pypdfium2, optional). Results cached by SHA-256; officer is told to check figures against the original. Added sample_documents/ and tests/test_pragya_ocr.py. Data agent: small tables must include the top row values (fixes self-test 22).

## Add data pop-up: synthetic datasets in one click
- **Add data** opens a pop-up with three tabs: **Our datasets** (tick the synthetic files that come with the app and press Add: the full demo case or the 3 sample statements, revenue tables, demo circulars, scanned documents, and the case tables as extra datasets), **In use now** (what is loaded, from which folder, row counts, previews, Remove), and **Upload your own (optional)** (bank statements, case folder, revenue tables, circulars and documents, or any other file: CSV, Excel, JSON, TXT, Word, PDF, photos).
- After adding, the pop-up confirms what was added and where it is used, offers ready questions that open Ask with the answer, and a button to see the results in Cases.
- The previous `app.py` is kept as `app.py.bak_before_data_page`.

## Optional Groq cloud model + sharing
- **Settings > Model** has a third runtime: **Groq cloud API** (needs internet and a free API key; the key is kept only while the app is open). A clear warning says questions and the facts needed to answer them go to Groq, so use it with demo data only. Ollama (offline) stays the default; scores and numbers always come from code.
- If Ollama is not found but `GROQ_API_KEY` is set, Pragya uses Groq automatically. `run_app_groq.bat` asks for the key and starts the app with Groq.
- `HOW_TO_SHARE.md` explains how someone else runs Pragya on their own computer; `share/Pragya_share.zip` is a clean copy to send (no uploads, audit log, profile or backups).

## Online demo on Streamlit Community Cloud with Groq
- In the online demo (public_demo.flag), **Add data** is shown: visitors add the built-in synthetic datasets, and what they add is kept in a private temporary folder for their visit only. Uploading from a visitor's computer stays off.
- The online demo uses Groq automatically when `GROQ_API_KEY` is set in Streamlit Cloud **Secrets** (or the environment); visitors cannot change the model. The top line says AI wording is by Groq (cloud).
- `streamlit_cloud/` is updated to this version (plus OCR packages and `packages.txt`); `HOW_TO_DEPLOY_STREAMLIT.md` has the steps.

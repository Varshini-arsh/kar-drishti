"""Universal upload: one "Add data" button that accepts everything.

For each file:
  1. read it      CSV, XLSX/XLS (every sheet), JSON, TXT, DOCX, PDF (text PDFs only; a scanned PDF
                  says "Scanned PDF: text could not be read" - OCR comes later)
  2. recognise it from its columns or text (bank statement, holder list, ITR summary, GST invoices,
                  GST returns, revenue targets, collections, last year's collections, demand register,
                  recovery actions, refunds, circular/office order, Act text, or "other table")
  3. the officer confirms or corrects the type in a dropdown
  4. load it      every table goes into DuckDB (recognised or not) so the Data agent can answer on it;
                  documents go to the Library
  5. fingerprint  SHA-256 + an audit-log entry
Files are kept on this computer in data/uploads/ with a small manifest, so they survive a restart.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import audit, ocr

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"

TABLE_KINDS = ["bank statement", "holder list", "ITR summary", "GST invoices", "GST returns", "revenue targets",
               "collections", "last year's collections", "demand register", "recovery actions", "refunds",
               "other table"]
TEXT_KINDS = ["circular / office order", "Act text", "other document"]
ALL_KINDS = TABLE_KINDS + TEXT_KINDS

# DuckDB table name for each recognised kind ("other table" uses the file name)
TABLE_NAME = {"bank statement": "bank_statement", "holder list": "holder_list", "ITR summary": "itr_summary",
              "GST invoices": "uploaded_gst_invoices", "GST returns": "uploaded_gst_returns",
              "revenue targets": "revenue_targets", "collections": "collections",
              "last year's collections": "collections_last_year", "demand register": "demand_register",
              "recovery actions": "recovery_actions", "refunds": "refunds"}

SCANNED_PDF = "Scanned PDF: text could not be read"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class Parsed:
    name: str
    sha256: str
    size: int
    tables: list = field(default_factory=list)     # [(sheet name or "", DataFrame)]
    text: str = ""
    note: str = ""                                 # e.g. scanned PDF, unreadable file
    kind: str = "other table"                      # recognised type
    reason: str = ""                               # why it was recognised that way

    @property
    def is_table(self) -> bool:
        return bool(self.tables)


# ---------------------------------------------------------------- reading
def read_file(name: str, data: bytes) -> Parsed:
    p = Parsed(name=name, sha256=audit.fingerprint(data), size=len(data))
    ext = Path(name).suffix.lower()
    try:
        if ext == ".csv":
            p.tables = [("", _read_csv(data))]
        elif ext in (".xlsx", ".xls", ".xlsm"):
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
            p.tables = [(s, df) for s, df in sheets.items() if not df.dropna(how="all").empty]
        elif ext == ".json":
            p.tables, p.text = _read_json(data)
        elif ext == ".docx":
            p.text = docx_text(data)
        elif ext == ".pdf":
            p.text, p.note = pdf_text(data, p.sha256)
        elif ext in IMAGE_EXT:
            p.text, p.note = image_text(data, p.sha256)
        elif ext in (".txt", ".md", ".text"):
            p.text = data.decode("utf-8", errors="ignore")
        else:
            p.note = f"File type '{ext or 'unknown'}' is not supported. Use CSV, XLSX, XLS, JSON, TXT, DOCX, PDF or an image (PNG, JPG, TIFF)."
    except Exception as e:                       # a bad file must never crash the app
        p.note = f"Could not read this file: {str(e).splitlines()[0][:200]}"
    p.tables = [(s, _clean(df)) for s, df in p.tables]
    p.kind, p.reason = recognise(p)
    return p


def _read_csv(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8-sig", errors="ignore")
    sample = text[:5000]
    sep = max([",", ";", "\t", "|"], key=sample.count)
    return pd.read_csv(io.StringIO(text), sep=sep)


def _read_json(data: bytes):
    obj = json.loads(data.decode("utf-8-sig", errors="ignore"))
    if isinstance(obj, dict):
        lists = [(k, v) for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            return [(k, pd.json_normalize(v)) for k, v in lists], ""
        return [("", pd.json_normalize(obj))], ""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return [("", pd.json_normalize(obj))], ""
    return [], json.dumps(obj, indent=1, ensure_ascii=False)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all").copy()
    cols, seen = [], {}
    for c in df.columns:
        c = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower() or "col"
        if c[0].isdigit():
            c = "c_" + c
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = cols
    return df.reset_index(drop=True)


def docx_text(data: bytes) -> str:
    """Text of a .docx using only the standard library (paragraphs, one per line)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
    lines = []
    for para in paras:
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.S))
        lines.append(re.sub(r"&amp;", "&", re.sub(r"&lt;", "<", re.sub(r"&gt;", ">", t))))
    return "\n".join(lines).strip()


def _has_text(t: str) -> bool:
    return len(re.sub(r"\s", "", t or "")) >= 20


def pdf_text(data: bytes, sha256: str = "") -> tuple[str, str]:
    """Text of a PDF. Pages with no text layer (scanned) are read with offline OCR when it is installed."""
    try:
        from pypdf import PdfReader
        pages = [(pg.extract_text() or "").strip() for pg in PdfReader(io.BytesIO(data)).pages]
    except ImportError:
        if not ocr.pdf_pages_available():
            return "", "Reading PDFs needs the pypdf package: run  pip install pypdf"
        import pypdfium2 as pdfium                     # pypdfium2 can read the text layer too
        pdf = pdfium.PdfDocument(data)
        pages = [(pdf[i].get_textpage().get_text_range() or "").strip() for i in range(len(pdf))]
    scanned = [i for i, t in enumerate(pages) if not _has_text(t)]
    if not scanned:
        return "\n".join(pages).strip(), ""
    if not (ocr.available() and ocr.pdf_pages_available()):
        text = "\n".join(pages).strip()
        if _has_text(text):
            return text, f"{len(scanned)} scanned page(s) could not be read. {ocr.INSTALL_HINT}"
        return "", f"{SCANNED_PDF}. {ocr.INSTALL_HINT}"
    key = f"{sha256}-pdf" if sha256 else ""
    hit = ocr.cached(key) if key else None
    if hit is not None:
        text = hit
    else:
        read, total = ocr.pdf_text(data, scanned)
        for i, t in read.items():
            pages[i] = t
        text = "\n".join(pages).strip()
        if key:
            ocr.save_cache(key, text)
    if not _has_text(text):
        return "", f"{SCANNED_PDF}: text recognition found no readable text (is the scan clear and in English?)."
    extra = f" Only the first {ocr.MAX_PAGES} scanned pages were read." if len(scanned) > ocr.MAX_PAGES else ""
    return text, ocr.OCR_NOTE + extra


def image_text(data: bytes, sha256: str = "") -> tuple[str, str]:
    """Text from a photo or scan of a document (PNG/JPG/TIFF) using offline OCR."""
    if not ocr.available():
        return "", f"This is an image. {ocr.INSTALL_HINT}"
    key = f"{sha256}-img" if sha256 else ""
    text = ocr.cached(key) if key else None
    if text is None:
        text = ocr.image_text(data)
        if key:
            ocr.save_cache(key, text)
    if not _has_text(text):
        return "", "Text recognition found no readable text in this image."
    return text, ocr.OCR_NOTE


# ---------------------------------------------------------------- recognising
def _has(cols: set, *words) -> bool:
    return any(any(w in c for c in cols) for w in words)


def recognise(p: Parsed) -> tuple[str, str]:
    """Best guess of what the file is, with a short reason."""
    if p.is_table:
        df = max((d for _, d in p.tables), key=len)
        return recognise_table(set(df.columns), p.name, df)
    if p.text:
        return recognise_text(p.text)
    return "other document", p.note or "no readable content"


def recognise_table(cols: set, file_name: str = "", df: pd.DataFrame | None = None) -> tuple[str, str]:
    fname = file_name.lower()
    if _has(cols, "invoice") and _has(cols, "gstin", "taxable"):
        return "GST invoices", "has invoice and GSTIN / taxable value columns"
    if _has(cols, "itc", "output_tax", "tax_paid_cash") and _has(cols, "period", "gstin", "month"):
        return "GST returns", "has output tax / ITC columns by period"
    if _has(cols, "target"):
        return "revenue targets", "has a target column"
    if _has(cols, "refund"):
        return "refunds", "has refund columns"
    if _has(cols, "demand") and _has(cols, "outstanding", "raised", "disputed", "stay", "ageing", "age"):
        return "demand register", "has demand amount and status columns"
    if _has(cols, "action", "attachment", "notice") and _has(cols, "recover", "demand"):
        return "recovery actions", "has recovery action columns"
    if _has(cols, "collect"):
        if "last" in fname or "prev" in fname or _is_last_year(df):
            return "last year's collections", "collection amounts for the previous financial year"
        return "collections", "has collection amount columns"
    if _has(cols, "date") and (_has(cols, "debit", "withdrawal") or _has(cols, "credit", "deposit")) \
            or (_has(cols, "date") and _has(cols, "narration", "particulars", "description") and _has(cols, "amount")):
        return "bank statement", "has date, narration and debit / credit columns"
    if _has(cols, "declared_income", "gross_receipts", "tax_paid", "total_income", "filed"):
        return "ITR summary", "has declared income / return columns"
    if _has(cols, "name") and _has(cols, "pan", "phone", "address", "holder", "person", "firm"):
        return "holder list", "has names with PAN / phone / address"
    return "other table", "columns did not match a known type; it is still loaded so you can ask about it"


def _is_last_year(df: pd.DataFrame | None) -> bool:
    if df is None or "fy" not in df.columns:
        return False
    import config
    try:
        return str(df["fy"].astype(str).max()) < config.TAX_YEAR
    except Exception:
        return False


def recognise_text(text: str) -> tuple[str, str]:
    low = text.lower()[:6000]
    if re.search(r"\bcircular\b|office order|\bo\.o\.|memorandum", low):
        return "circular / office order", "mentions circular / office order"
    if re.search(r"\bact,? (19|20)\d\d\b", low) and re.search(r"\bsection\b|\bchapter\b|\bshall\b", low):
        return "Act text", "has Act, sections and 'shall'"
    return "other document", "text document"


# ---------------------------------------------------------------- loading
def table_name(kind: str, file_name: str, sheet: str, existing: set[str]) -> str:
    base = TABLE_NAME.get(kind) or "up_" + re.sub(r"[^0-9a-z]+", "_", Path(file_name).stem.lower()).strip("_")
    if sheet and kind not in TABLE_NAME:
        base += "_" + re.sub(r"[^0-9a-z]+", "_", sheet.lower()).strip("_")
    base = base[:48] or "uploaded"
    name, i = base, 2
    while name in existing:
        name, i = f"{base}_{i}", i + 1
    return name


def load_into_duckdb(con, name: str, df: pd.DataFrame) -> int:
    """Create (or replace) table `name` from df. Returns row count."""
    con.register("_pragya_up", df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _pragya_up')
    finally:
        con.unregister("_pragya_up")
    return len(df)


# ---------------------------------------------------------------- the manifest (files kept on this computer)
def manifest_path(folder: Path = UPLOAD_DIR) -> Path:
    return folder / "manifest.json"


def read_manifest(folder: Path = UPLOAD_DIR) -> list[dict]:
    try:
        return json.loads(manifest_path(folder).read_text(encoding="utf-8"))
    except Exception:
        return []


def save_upload(p: Parsed, data: bytes, kind: str, folder: Path = UPLOAD_DIR, audit_path=None,
                tables: list | None = None) -> dict:
    """Keep the file, record it in the manifest and the audit log. Returns the manifest entry."""
    folder.mkdir(parents=True, exist_ok=True)
    stored = folder / f"{p.sha256[:12]}_{Path(p.name).name}"
    if not stored.exists():
        stored.write_bytes(data)
    entries = [e for e in read_manifest(folder) if e["sha256"] != p.sha256]
    entry = {"name": p.name, "sha256": p.sha256, "size": p.size, "kind": kind, "recognised_as": p.kind,
             "stored": stored.name, "tables": tables or [], "note": p.note}
    entries.append(entry)
    manifest_path(folder).write_text(json.dumps(entries, indent=1), encoding="utf-8")
    audit.log("file_added", audit_path, file=p.name, sha256=p.sha256, size=p.size, recognised_as=p.kind,
              confirmed_as=kind, tables=tables or [], note=p.note)
    if kind != p.kind:
        audit.log("approval", audit_path, what="file type corrected by officer", file=p.name,
                  recognised_as=p.kind, confirmed_as=kind)
    return entry


def remove_upload(sha256: str, folder: Path = UPLOAD_DIR, audit_path=None) -> None:
    entries = read_manifest(folder)
    keep = [e for e in entries if e["sha256"] != sha256]
    gone = [e for e in entries if e["sha256"] == sha256]
    manifest_path(folder).write_text(json.dumps(keep, indent=1), encoding="utf-8")
    for e in gone:
        try:
            (folder / e["stored"]).unlink()
        except OSError:
            pass
        audit.log("file_removed", audit_path, file=e["name"], sha256=sha256)


def load_saved(con, folder: Path = UPLOAD_DIR) -> tuple[list[dict], list[dict]]:
    """Reload every kept file: tables into DuckDB, documents returned for the Library.
    Returns (tables loaded [{table, kind, rows, file}], documents [{name, kind, text, sha256}])."""
    tables, docs = [], []
    existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    for e in read_manifest(folder):
        path = folder / e["stored"]
        if not path.exists():
            continue
        p = read_file(e["name"], path.read_bytes())
        if p.is_table:
            for sheet, df in p.tables:
                name = table_name(e["kind"], e["name"], sheet, existing)
                existing.add(name)
                tables.append({"table": name, "kind": e["kind"], "rows": load_into_duckdb(con, name, df),
                               "file": e["name"], "df": df})
        elif p.text:
            docs.append({"name": e["name"], "kind": e["kind"], "text": p.text, "sha256": e["sha256"]})
    return tables, docs

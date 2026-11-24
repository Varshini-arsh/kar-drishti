"""Offline text recognition (OCR) for scanned PDFs and photos of documents.

Uses RapidOCR (ONNX models bundled inside the pip package, so nothing is downloaded at run time)
and pypdfium2 to turn PDF pages into images. Both are optional: without them Pragya still works
and simply reports that a scanned file could not be read.

    pip install rapidocr_onnxruntime pypdfium2

Results are cached on disk by the file's SHA-256, so each scanned file is recognised only once.
Recognition supports English (and Chinese) printed text; Hindi script is not recognised.
"""
from __future__ import annotations

import io
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ocr_cache"
MAX_PAGES = 40           # very long scans are cut here (noted in the result)
MIN_SCORE = 0.5          # drop low-confidence fragments
OCR_NOTE = ("Read with text recognition (OCR) from a scanned file. Check key figures, dates and "
            "document numbers against the original.")
INSTALL_HINT = "To read scanned files, install OCR once:  pip install rapidocr_onnxruntime pypdfium2"

_engine = None


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def pdf_pages_available() -> bool:
    try:
        import pypdfium2  # noqa: F401
        return True
    except ImportError:
        return False


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def _lines(result) -> list[str]:
    """Group RapidOCR boxes into reading-order lines (top to bottom, left to right)."""
    items = []
    for box, text, score in result or []:
        if float(score) < MIN_SCORE or not str(text).strip():
            continue
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append((min(ys), max(ys), min(xs), str(text).strip()))
    items.sort(key=lambda t: (t[0], t[2]))
    lines, current, cur_top, cur_bot = [], [], None, None
    for top, bot, left, text in items:
        mid = (top + bot) / 2
        if current and cur_top <= mid <= cur_bot:          # same visual line
            current.append((left, text))
            cur_bot = max(cur_bot, bot)
        else:
            if current:
                lines.append(" ".join(t for _, t in sorted(current)))
            current, cur_top, cur_bot = [(left, text)], top, bot
    if current:
        lines.append(" ".join(t for _, t in sorted(current)))
    return lines


def tidy(text: str) -> str:
    """Undo common OCR word-joining: 'OfficeCircularNo.PRG/1' -> 'Office Circular No. PRG/1'."""
    import re
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\b(No|Dated|Date|Subject|Sub)([.:])(?=\S)", r"\1\2 ", text)
    return text


def image_text(image) -> str:
    """OCR one image (PIL image, numpy array, or bytes of a PNG/JPG)."""
    import numpy as np
    from PIL import Image
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    result, _ = _get_engine()(image)
    return tidy("\n".join(_lines(result)))


def pdf_text(data: bytes, pages: list[int] | None = None) -> tuple[dict[int, str], int]:
    """OCR the given pages (0-based; all when None) of a PDF. Returns ({page: text}, total pages)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    total = len(pdf)
    wanted = [i for i in (pages if pages is not None else range(total)) if i < total][:MAX_PAGES]
    out = {}
    for i in wanted:
        img = pdf[i].render(scale=2.5).to_pil()        # ~180 dpi: good accuracy, reasonable speed
        out[i] = image_text(img)
    return out, total


def cached(sha256: str) -> str | None:
    f = CACHE_DIR / f"{sha256}.txt"
    try:
        return f.read_text(encoding="utf-8") if f.exists() else None
    except OSError:
        return None


def save_cache(sha256: str, text: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{sha256}.txt").write_text(text, encoding="utf-8")
    except OSError:
        pass

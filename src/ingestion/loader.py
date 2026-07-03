"""
loader.py — Step 1 of ingestion: PDF file -> raw text, one entry per page.

Responsibility (and ONLY this): get text out of the PDF.
No cleaning, no chunking, no interpretation — those belong to parser.py.
Keeping this boundary means we can swap the extraction backend
(pdfplumber today, a vision-based extractor in Stage 2.5) without
touching any downstream code.
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class Page:
    """One page of extracted text, with provenance."""
    number: int   # 1-based page number, matching what a human sees in a viewer
    text: str     # raw extracted text ("" if the page yielded nothing)


def load_pdf(pdf_path: str | Path) -> list[Page]:
    """
    Extract text from every page of a PDF.

    Inputs:  pdf_path — path to a text-based (non-scanned) PDF.
    Returns: list[Page], in page order. Blank pages become Page(n, "").
    Raises:  FileNotFoundError if the path is wrong (fail loudly, early).

    Complexity: O(total glyphs); a 35-page PDF takes ~1-2 seconds.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No PDF at: {pdf_path}")

    pages: list[Page] = []
    # Context manager: guarantees the file handle closes even if
    # extraction raises mid-way through the document.
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages):
            # extract_text() reconstructs reading order from glyph
            # coordinates. It returns None (not "") for blank pages,
            # hence the `or ""` normalisation.
            pages.append(Page(number=index + 1, text=page.extract_text() or ""))
    return pages

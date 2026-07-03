"""
vision_loader.py — extraction backend #2: LLM-as-parser.

WHY THIS EXISTS
The PDF's math is typeset in a broken Type 3 font, so glyph-based
extractors (pdfplumber, PyMuPDF) silently drop every formula AND every
final answer (measured loss on this doc: all math runs). The fix used in
production for damaged/complex PDFs: rasterize each page to an image and
have a vision-capable LLM transcribe what is RENDERED. Broken font
metadata cannot hurt a model that reads pixels.

Cost/benefit vs pdfplumber:
  pdfplumber: free, ~2s/doc, lossy here.   vision: ~cents/doc, ~min/doc, complete.
Rule of thumb: try glyph extraction first, escalate to vision when
validation fails. Transcriptions are CACHED to data/interim/pages/ so
the API cost is paid once per document, ever.

This module mirrors loader.py's interface (-> list[Page]), so parser.py
cannot tell the backends apart. That interchangeability was the whole
point of keeping the loader boundary clean.
"""

import base64
import re
from pathlib import Path

from .loader import Page

CACHE_DIR = Path("data/interim/pages")

# Matches "#", "##", ... at line start: the transcriber emits markdown
# headings ("# Easy Questions") but parser.py's grammar expects the bare
# line. We normalise in the READER so the parser stays dialect-agnostic.
_MD_HEADING = re.compile(r"^#{1,6}\s+")


def load_transcribed_pages(cache_dir: str | Path = CACHE_DIR) -> list[Page]:
    """
    Read cached page transcriptions (page_01.md, page_02.md, ...) -> Pages.

    Returns pages sorted by number; raises if the cache is missing so a
    half-ingested corpus can never be indexed silently.
    """
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("page_*.md"))
    if not files:
        raise FileNotFoundError(
            f"No cached transcriptions in {cache_dir}. "
            "Run transcribe_pdf() once (requires ANTHROPIC_API_KEY)."
        )
    pages = []
    for f in files:
        number = int(f.stem.split("_")[1])          # "page_07" -> 7
        text = "\n".join(_MD_HEADING.sub("", ln) for ln in
                         f.read_text(encoding="utf-8").splitlines())
        pages.append(Page(number=number, text=text))
    return pages


# ------------------------------------------------------------------ generator
TRANSCRIBE_PROMPT = """Transcribe this exam-paper page to plain text, completely and in reading order.
Rules:
- Include ALL mathematical expressions inline, e.g. 180(n - 2), 6840 / 40, x = 19.
- Keep mark tags like [1] and lines like (3 marks) exactly as printed.
- Keep the literal section headers (e.g. 'Easy Questions') and the word 'Answer' on its own line.
- For diagrams, transcribe only labelled values/letters that appear; do not describe the picture.
- Output the transcription only — no commentary."""


def transcribe_pdf(pdf_path: str | Path, cache_dir: str | Path = CACHE_DIR,
                   model: str = "claude-sonnet-4-6", dpi: int = 150) -> None:
    """
    One-time generator: PDF -> page images -> Claude -> cached page_NN.md.

    Requires: `pip install anthropic pymupdf` and ANTHROPIC_API_KEY in env.
    Skips pages already cached, so an interrupted run resumes for free
    (idempotency — always design batch jobs to be safely re-runnable).
    """
    import fitz                     # PyMuPDF: renders pages to images
    from anthropic import Anthropic  # official SDK; reads key from env

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = Anthropic()
    doc = fitz.open(str(pdf_path))

    for i, page in enumerate(doc, start=1):
        out = cache_dir / f"page_{i:02d}.md"
        if out.exists():
            continue                                     # resume support
        pix = page.get_pixmap(dpi=dpi)                   # rasterise
        img_b64 = base64.standard_b64encode(pix.tobytes("png")).decode()
        response = client.messages.create(
            model=model, max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": img_b64}},
                {"type": "text", "text": TRANSCRIBE_PROMPT},
            ]}],
        )
        out.write_text(response.content[0].text, encoding="utf-8")
        print(f"transcribed page {i}/{len(doc)}")

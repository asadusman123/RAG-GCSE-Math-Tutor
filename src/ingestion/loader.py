"""
loader.py — Stage 2 (ingestion layer).

Responsibility: turn a PDF file on disk into a list of Page objects
(page number + plain text). Nothing else. Chunking, metadata and
embeddings belong to later components.

Two loaders are provided because our source PDF contains a broken
math font (Type 3, custom encoding) that makes text-layer extraction
lossy — it silently drops most of the final answers:

    TextLayerLoader — fast, free, offline. Loses math-font glyphs.
                      Kept for comparison/tests and for healthy PDFs.
    VisionLoader    — rasterizes each page and asks Claude (vision)
                      to transcribe it to Markdown. Slow and costs
                      API tokens, but recovers everything a human can
                      see, INCLUDING diagram descriptions. Each page's
                      transcription is cached on disk, so the cost is
                      paid exactly once per document.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF — PDF engine used for both text and rasterization


@dataclass(frozen=True)
class Page:
    """One page of a document. `number` is 1-based (as printed on the page)."""
    number: int
    text: str


# ---------------------------------------------------------------------------
# Loader 1: plain text layer (fast, free — but lossy on this document)
# ---------------------------------------------------------------------------

class TextLayerLoader:
    """Extract the embedded text layer of each page."""

    def load(self, pdf_path: str | Path) -> list[Page]:
        pages: list[Page] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = self._strip_furniture(page.get_text())
                pages.append(Page(number=i, text=text))
        return pages

    @staticmethod
    def _strip_furniture(text: str) -> str:
        """Remove per-page boilerplate (footers/URLs/page numbers).

        Why: repeated boilerplate appears in every chunk, dragging all
        chunk embeddings toward a common direction and blurring the very
        distinctions retrieval depends on.
        """
        keep: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("©") or "savemyexams.com" in s.lower():
                continue
            keep.append(s)
        return "\n".join(keep)


# ---------------------------------------------------------------------------
# Loader 2: vision transcription via Claude, with a page-level disk cache
# ---------------------------------------------------------------------------

# The transcription prompt is part of the *code contract*: the parser
# downstream depends on question numbers, "[1]" mark tags, "(N marks)"
# totals and "[DIAGRAM: ...]" lines being preserved exactly.
TRANSCRIPTION_PROMPT = """\
Transcribe this exam-paper page to plain Markdown.

Rules:
1. Transcribe ALL text verbatim — question numbers, method headings,
   per-step mark tags like [1], and totals like (3 marks). Do not
   summarise, reorder or omit anything.
2. Write mathematical expressions in plain linear notation, e.g.
   180(n - 2), ((n - 2) * 180) / n, cos(54) = BM / 12.
3. For every diagram, insert one bracketed description on its own line:
   [DIAGRAM: what it shows — shapes, labels, given angles and lengths].
4. Ignore page furniture: headers, footers, copyright lines, page numbers.
5. Output ONLY the transcription — no commentary, no code fences.
"""


class VisionLoader:
    """Transcribe each page with a vision LLM; cache results to disk.

    Cost profile: ~1,600 image tokens per page at 150 DPI, paid ONCE.
    Subsequent loads are free disk reads. This expensive-once /
    cheap-forever shape is characteristic of ingestion pipelines.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        model: str = "claude-sonnet-4-6",
        dpi: int = 150,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.dpi = dpi
        self._client = None  # created lazily: cache hits need no API key

    def load(self, pdf_path: str | Path) -> list[Page]:
        pages: list[Page] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                cache_file = self.cache_dir / f"page_{i:02d}.md"
                if cache_file.exists():                      # cache hit
                    text = cache_file.read_text(encoding="utf-8")
                else:                                        # cache miss
                    text = self._transcribe(page)
                    cache_file.write_text(text, encoding="utf-8")
                pages.append(Page(number=i, text=text.strip()))
        return pages

    # -- internals ----------------------------------------------------------

    def _transcribe(self, page: "fitz.Page") -> str:
        """Rasterize one page and ask Claude to transcribe it."""
        png_bytes = page.get_pixmap(dpi=self.dpi).tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")

        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=2000,          # a dense page is ~700 tokens; headroom
            temperature=0,            # transcription wants determinism
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png",
                                "data": b64}},
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                ],
            }],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    def _get_client(self):
        """Create the Anthropic client on first use only.

        Why lazy: if every page is cached, this loader must work on a
        machine with no ANTHROPIC_API_KEY at all.
        """
        if self._client is None:
            import anthropic  # imported here so cache-only use needs no SDK
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client


if __name__ == "__main__":
    # Smoke test: compare the two loaders on the project PDF.
    pdf = Path(__file__).parents[2] / "data/raw/Angles_in_Polygons___Parallel_Lines___35_CQA.pdf"
    cache = Path(__file__).parents[2] / "data/interim/pages"

    text_pages = TextLayerLoader().load(pdf)
    print(f"TextLayerLoader: {len(text_pages)} pages, "
          f"{sum(len(p.text) for p in text_pages)} chars")

    vision_pages = VisionLoader(cache_dir=cache).load(pdf)
    print(f"VisionLoader:    {len(vision_pages)} pages, "
          f"{sum(len(p.text) for p in vision_pages)} chars")

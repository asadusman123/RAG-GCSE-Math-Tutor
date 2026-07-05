"""
diagram_renderer.py — turn a [DIAGRAM: ...] text description into a clean
black-on-white SVG figure, cached to disk.

WHY SVG, NOT AI IMAGE GENERATION
An image (diffusion) model paints plausible PIXELS — for geometry that
means wrong angles, mislabelled vertices, hallucinated sides. That would
bolt a hallucination engine onto a system whose whole value is grounding.
SVG is COORDINATES AND SHAPES — i.e. geometry expressed as math. The LLM
is far more reliable emitting drawing CODE than painting pixels, and we can
VALIDATE the code (does it parse as XML/SVG?) before trusting it. Grounded,
verifiable, with a fallback. Same philosophy as the rest of the project.

DESIGN (per the chosen options):
  - PRE-GENERATE once (this is an ingestion-time step), cache to
    data/interim/diagrams/<qid>.svg. Zero latency at demo time.
  - AUTO-DETECT: only questions whose text contains a [DIAGRAM: ...] tag.
  - The UI always shows the SVG AND the original description, so a wonky
    diagram never leaves the student worse off than text alone.

Provider abstraction mirrors the evaluator: DiagramRenderer base with a
_complete() seam; StubRenderer (offline tests) and GeminiRenderer (real).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

CACHE_DIR = Path("data/interim/diagrams")
DIAGRAM_RE = re.compile(r"\[DIAGRAM:\s*(.*?)\]", re.DOTALL)


def extract_diagram(question_text: str) -> str | None:
    """Return the description inside a [DIAGRAM: ...] tag, or None.
    Data flow: this is how the rest of the system asks 'does this question
    need a figure, and what does it depict?'"""
    m = DIAGRAM_RE.search(question_text)
    return m.group(1).strip() if m else None


def is_valid_svg(text: str) -> bool:
    """True iff `text` parses as XML with an <svg> root. Our safety gate:
    we never cache output that isn't at least well-formed SVG."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return False
    return root.tag.split("}")[-1].lower() == "svg"   # handle namespaces


SVG_SYSTEM = """You convert geometry diagram descriptions into clean, minimal SVG figures.

RULES:
- Output ONLY valid SVG code: a single <svg> element, no markdown, no prose, no code fences.
- viewBox="0 0 400 400". Black strokes on transparent background, stroke-width 2, no fill (fill="none") except small dots for labelled points.
- Label every named point/vertex with an SVG <text> element near its location.
- Prioritise correct TOPOLOGY (which points connect, relative positions, which lines extend where) over perfect scale. The figure is an aid, not a precise construction.
- Keep it simple line-art: shapes, lines, point dots, and text labels only."""

SVG_TEMPLATE = """Draw this geometry description as SVG:

{description}

Output only the <svg>...</svg> code."""


class DiagramRenderer(ABC):
    """Base: prompt build + SVG validation + disk cache. Subclasses do
    only _complete()."""

    def __init__(self, cache_dir: str | Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Provider-specific call: prompts in, raw text out (the seam)."""

    def render(self, description: str) -> str:
        """Description -> SVG string. Strips any stray code fences the model
        adds. Raises ValueError if the result isn't valid SVG, so callers
        can decide how to fall back."""
        raw = self._complete(SVG_SYSTEM, SVG_TEMPLATE.format(description=description))
        svg = raw.strip()
        if svg.startswith("```"):
            svg = svg.split("```", 2)[1]
            svg = svg[3:] if svg.startswith("svg") else svg
            svg = svg.strip()
        if not is_valid_svg(svg):
            raise ValueError(f"Model did not return valid SVG:\n{raw[:200]}")
        return svg

    def render_and_cache(self, question_id: str, description: str) -> str:
        """Render once, save to <qid>.svg, and return it. If a cache file
        exists, reuse it (idempotent — re-running ingestion is free)."""
        path = self.cache_dir / f"{question_id}.svg"
        if path.exists():
            return path.read_text(encoding="utf-8")
        svg = self.render(description)
        path.write_text(svg, encoding="utf-8")
        return svg

    def get_cached(self, question_id: str) -> str | None:
        """Read a cached SVG if present (used at serve time — never calls
        the API). Returns None if this question has no cached diagram."""
        path = self.cache_dir / f"{question_id}.svg"
        return path.read_text(encoding="utf-8") if path.exists() else None


class StubRenderer(DiagramRenderer):
    """Offline renderer: emits a fixed valid SVG. Lets tests exercise the
    extract -> validate -> cache flow with no API call."""

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return ('<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">'
                '<circle cx="200" cy="200" r="120" fill="none" stroke="black" '
                'stroke-width="2"/><text x="195" y="70">A</text></svg>')


class GeminiRenderer(DiagramRenderer):
    """Real renderer via Gemini, reusing GeminiEvaluator's working seam so
    the SDK is touched in exactly one place across the whole project."""

    def __init__(self, cache_dir: str | Path = CACHE_DIR,
                 model: str = "gemini-2.5-flash"):
        super().__init__(cache_dir)
        from src.tutor.evaluator import GeminiEvaluator
        self._backend = GeminiEvaluator(model=model)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._backend._complete(system_prompt, user_prompt)


def pregenerate_all(renderer: DiagramRenderer | None = None) -> dict[str, str]:
    """
    Ingestion-time batch: find every question with a [DIAGRAM: ...] tag and
    render+cache its SVG. Idempotent and resumable (skips cached ones).
    Returns {question_id: svg}. Run once after building the index.

    Run:  python -m src.ingestion.diagram_renderer
    """
    from src.ingestion.vision_loader import load_transcribed_pages
    from src.ingestion.parser import parse

    renderer = renderer or GeminiRenderer()
    questions = [c for c in parse(load_transcribed_pages()) if c.type == "question"]
    out: dict[str, str] = {}
    for q in questions:
        description = extract_diagram(q.text)
        if not description:
            continue
        try:
            out[q.id] = renderer.render_and_cache(q.id, description)
            print(f"  rendered {q.id}")
        except ValueError as exc:
            print(f"  SKIPPED {q.id}: {exc}")
    print(f"Done. {len(out)} diagrams cached in {renderer.cache_dir}/")
    return out


if __name__ == "__main__":
    pregenerate_all()
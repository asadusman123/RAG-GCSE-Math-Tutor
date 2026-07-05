"""Diagram renderer tests — offline via StubRenderer."""
import pytest
from src.ingestion.diagram_renderer import (
    extract_diagram, is_valid_svg, StubRenderer, DiagramRenderer)


def test_extract_finds_and_misses():
    assert extract_diagram("x [DIAGRAM: a pentagon] y") == "a pentagon"
    assert extract_diagram("no diagram here") is None


def test_valid_svg_gate():
    assert is_valid_svg('<svg xmlns="http://www.w3.org/2000/svg"><line/></svg>')
    assert not is_valid_svg("I cannot draw that")
    assert not is_valid_svg("<html><body>no</body></html>")


def test_render_validates_and_caches(tmp_path):
    r = StubRenderer(cache_dir=tmp_path)
    svg = r.render_and_cache("q1", "a circle")
    assert is_valid_svg(svg)
    assert (tmp_path / "q1.svg").exists()
    assert r.get_cached("q1") == svg              # cache hit
    assert r.get_cached("missing") is None


def test_render_rejects_non_svg(tmp_path):
    class BadRenderer(DiagramRenderer):
        def _complete(self, s, u): return "sorry, no"
    with pytest.raises(ValueError, match="valid SVG"):
        BadRenderer(cache_dir=tmp_path).render("anything")


def test_fenced_svg_is_stripped(tmp_path):
    class FencedRenderer(DiagramRenderer):
        def _complete(self, s, u):
            return '```svg\n<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>\n```'
    assert is_valid_svg(FencedRenderer(cache_dir=tmp_path).render("box"))
"""
Ingestion contract tests.

These encode the properties every later stage relies on. If a future
change (new loader, parser tweak) breaks a contract, these fail loudly
BEFORE bad chunks reach the index.

Run:  python -m pytest tests/ -q
"""

import pytest

from src.ingestion.loader import load_pdf
from src.ingestion.parser import parse, validate, EXPECTED_COUNTS

PDF = "data/raw/Angles_in_Polygons___Parallel_Lines___35_CQA.pdf"


# `scope="module"`: load+parse ONCE and share across tests — the pipeline
# is deterministic, so re-running it per test would only waste seconds.
@pytest.fixture(scope="module")
def chunks():
    return parse(load_pdf(PDF))


def test_loader_returns_all_pages():
    pages = load_pdf(PDF)
    assert len(pages) == 35
    assert pages[0].number == 1            # 1-based, like a PDF viewer


def test_validation_is_clean(chunks):
    # The parser's self-check against the cover page's own totals.
    assert validate(chunks) == []


def test_question_answer_pairing(chunks):
    # THE core contract: every question resolves to its answer BY ID,
    # and the link is symmetric. Grading depends on this being exact.
    by_id = {c.id: c for c in chunks}
    questions = [c for c in chunks if c.type == "question"]
    assert len(questions) == sum(EXPECTED_COUNTS.values())  # 13
    for q in questions:
        ans = by_id[q.pair_id]
        assert ans.type == "answer"
        assert ans.pair_id == q.id         # symmetric back-link
        assert ans.text.strip()            # never grade against emptiness


def test_no_answer_leaks_into_question(chunks):
    # A tutor that shows mark-scheme text inside the question has failed.
    # Mark-scheme lines are recognisable by "[1]" style mark tags.
    for q in (c for c in chunks if c.type == "question"):
        assert "Method 1" not in q.text, q.id
        assert not q.text.rstrip().endswith("marks)"), q.id


def test_boilerplate_stripped(chunks):
    for c in chunks:
        assert "savemyexams" not in c.text.lower(), c.id


def test_metadata_sane(chunks):
    for c in chunks:
        assert c.difficulty in EXPECTED_COUNTS, c.id
        assert c.marks > 0, c.id
        assert c.pages == sorted(c.pages), c.id
        assert c.topic in {"polygons", "parallel_lines", "triangles", "angles_general"}

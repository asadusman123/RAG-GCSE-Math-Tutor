"""
Ingestion contract tests — run against BOTH extraction backends.

pytest parametrization runs every contract twice (pdfplumber, vision);
a regression in either path fails loudly before bad chunks reach the index.
Run:  python -m pytest tests/ -q
"""

import pytest

from src.ingestion.loader import load_pdf
from src.ingestion.vision_loader import load_transcribed_pages
from src.ingestion.parser import parse, validate, EXPECTED_COUNTS

PDF = "data/raw/Angles_in_Polygons___Parallel_Lines___35_CQA.pdf"

LOADERS = {
    "pdfplumber": lambda: load_pdf(PDF),
    "vision": lambda: load_transcribed_pages(),
}


@pytest.fixture(scope="module", params=LOADERS)          # each test runs per backend
def chunks(request):
    return parse(LOADERS[request.param]())


def test_loader_returns_all_pages():
    assert len(load_pdf(PDF)) == 35
    assert len(load_transcribed_pages()) == 35


def test_validation_is_clean(chunks):
    # Self-check against the totals the PDF's cover page asserts about itself.
    assert validate(chunks) == []


def test_question_answer_pairing(chunks):
    # THE core contract: every question resolves to its answer BY ID.
    by_id = {c.id: c for c in chunks}
    questions = [c for c in chunks if c.type == "question"]
    assert len(questions) == sum(EXPECTED_COUNTS.values())      # 13
    for q in questions:
        ans = by_id[q.pair_id]
        assert ans.type == "answer"
        assert ans.pair_id == q.id                              # symmetric link
        assert ans.text.strip()


def test_no_answer_leaks_into_question(chunks):
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


def test_vision_backend_recovers_math_that_glyphs_lose():
    """Documents the Type 3 font defect and its fix, permanently.

    If this ever fails on the 'vision' side, the transcription cache is
    damaged; if the pdfplumber assertions start PASSING, the doc was
    re-exported with healthy fonts and vision is no longer needed."""
    glyph = {c.id: c for c in parse(load_pdf(PDF))}
    vision = {c.id: c for c in parse(load_transcribed_pages())}
    for qid, needle in [("medium_q1_ans", "n = 12"),
                        ("easy_q1_ans", "171°"),
                        ("hard_q4_ans", "x = 19")]:
        assert needle not in glyph[qid].text        # the defect
        assert needle in vision[qid].text           # the fix

"""
ChromaDB backend tests — and, more importantly, EQUIVALENCE tests.

The migration is only trustworthy if Chroma reproduces the same behaviour
as FAISS. These tests assert the invariants hold on BOTH backends, and that
Chroma's native pre-filtering agrees with FAISS's post-filtering.
"""

import pytest

from src.embeddings.embedder import HashingEmbedder
from src.ingestion.pipeline import build_index
from src.retrieval.retriever import Retriever, _to_chroma_where


@pytest.fixture(scope="module")
def emb():
    return HashingEmbedder()


@pytest.fixture(scope="module")
def chroma(tmp_path_factory, emb):
    d = tmp_path_factory.mktemp("chroma")
    build_index(embedder=emb, index_dir=str(d), backend="chroma")
    return Retriever(index_dir=str(d), embedder=emb, min_score=0.0,
                     backend="chroma")


@pytest.fixture(scope="module")
def faiss(tmp_path_factory, emb):
    d = tmp_path_factory.mktemp("faiss")
    build_index(embedder=emb, index_dir=d)
    return Retriever(index_dir=str(d), embedder=emb, min_score=0.0)


# ---- Chroma-specific ---------------------------------------------------
def test_where_translation():
    # Chroma needs $and for 2+ conditions — a real API quirk.
    assert _to_chroma_where(None) is None
    assert _to_chroma_where({"type": "question"}) == {"type": "question"}
    assert _to_chroma_where({"type": "question", "difficulty": "hard"}) == {
        "$and": [{"type": "question"}, {"difficulty": "hard"}]}


def test_native_prefilter_never_returns_answers(chroma):
    for q in ["polygon angle", "parallel lines", "simultaneous equations"]:
        for h in chroma.search(q, k=5, filters={"type": "question"}):
            assert h.type == "question"


def test_chroma_compound_filter(chroma):
    hard = chroma.all_questions(filters={"difficulty": "hard"})
    assert len(hard) == 4 and all(r["difficulty"] == "hard" for r in hard)


# ---- EQUIVALENCE: the migration is only safe if these hold -------------
def test_both_backends_index_same_chunks(chroma, faiss):
    assert len(chroma.all_questions()) == len(faiss.all_questions()) == 13


def test_both_backends_agree_on_top_hit(chroma, faiss):
    for query in ["interior angle of a regular polygon",
                  "parallel lines isosceles triangle"]:
        c = chroma.search(query, k=1, filters={"type": "question"})
        f = faiss.search(query, k=1, filters={"type": "question"})
        assert c and f
        assert c[0].id == f[0].id                  # same winner
        assert c[0].score == pytest.approx(f[0].score, abs=0.02)  # same score


def test_both_backends_pair_answers_exactly(chroma, faiss):
    for r in (chroma, faiss):
        for q in r.all_questions():
            assert r.get_answer_for(q["id"])["id"] == q["pair_id"]


def test_sequential_order_matches(chroma, faiss):
    assert [q["id"] for q in chroma.all_questions()] == \
           [q["id"] for q in faiss.all_questions()]
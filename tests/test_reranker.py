"""
Re-ranker tests — the two-stage retrieval pattern.
Offline: LexicalReranker + StubReranker, no model download.
"""

import pytest

from src.embeddings.embedder import HashingEmbedder
from src.ingestion.pipeline import build_index
from src.retrieval.retriever import Retriever, Hit
from src.retrieval.reranker import LexicalReranker, StubReranker


def _hit(hid, text, score=0.5):
    return Hit(score=score, record={"id": hid, "type": "question", "text": text})


def test_reranker_reorders_by_relevance():
    # Stage 1 put the WRONG chunk first; the re-ranker must fix the order.
    hits = [_hit("wrong", "A regular polygon has 72 sides.", 0.61),
            _hit("right", "one interior angle of a regular polygon with 40 sides", 0.58)]
    ranked = LexicalReranker().rerank("interior angle polygon 40 sides", hits)
    assert ranked[0].id == "right"          # promoted from 2nd to 1st


def test_stub_reranker_proves_reordering_happens():
    # Stub reverses order — an unmistakable effect.
    hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
    assert [h.id for h in StubReranker().rerank("q", hits)] == ["c", "b", "a"]


def test_rerank_respects_top_k():
    hits = [_hit(str(i), f"text {i}") for i in range(10)]
    assert len(LexicalReranker().rerank("text", hits, top_k=3)) == 3


def test_rerank_empty_is_safe():
    assert LexicalReranker().rerank("q", []) == []


def test_scores_are_replaced_by_reranker_score():
    hits = [_hit("a", "interior angle polygon", 0.99)]
    ranked = LexicalReranker().rerank("interior angle polygon", hits)
    assert ranked[0].score == pytest.approx(1.0)   # not the old 0.99


def test_retriever_two_stage_integration(tmp_path):
    emb = HashingEmbedder()
    build_index(embedder=emb, index_dir=tmp_path)
    r = Retriever(index_dir=str(tmp_path), embedder=emb, min_score=0.0,
                  reranker=LexicalReranker(), rerank_pool=10)
    hits = r.search("interior angle of a regular polygon with 40 sides",
                    k=3, filters={"type": "question"})
    assert len(hits) == 3
    assert all(h.type == "question" for h in hits)   # filters still enforced
    assert hits[0].score >= hits[1].score            # properly ordered


def test_reranking_preserves_leak_protection(tmp_path):
    emb = HashingEmbedder()
    build_index(embedder=emb, index_dir=tmp_path)
    r = Retriever(index_dir=str(tmp_path), embedder=emb, min_score=0.0,
                  reranker=LexicalReranker())
    for h in r.search("polygon angles", k=5, filters={"type": "question"}):
        assert h.type == "question"        # answers can never survive stage 2
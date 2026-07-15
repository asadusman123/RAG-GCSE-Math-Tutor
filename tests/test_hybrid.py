"""
Hybrid retrieval tests — dense + BM25 sparse, fused by RRF. Offline.
"""

import pytest

from src.embeddings.embedder import HashingEmbedder
from src.ingestion.pipeline import build_index
from src.retrieval.retriever import Retriever, Hit
from src.retrieval.hybrid import (HybridRetriever, SparseIndex,
                                  reciprocal_rank_fusion, RRF_K)


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx")
    emb = HashingEmbedder()
    build_index(embedder=emb, index_dir=d)
    return Retriever(index_dir=str(d), embedder=emb, min_score=0.0)


# ---- RRF math (pure, no index) ----------------------------------------
def _h(hid):
    return Hit(score=0.0, record={"id": hid, "type": "question", "text": hid})


def test_rrf_rewards_items_ranked_well_in_both_lists():
    list_a = [_h("x"), _h("y"), _h("z")]      # x=1, y=2, z=3
    list_b = [_h("y"), _h("x"), _h("w")]      # y=1, x=2, w=3
    fused = reciprocal_rank_fusion([list_a, list_b])
    ids = [h.id for h in fused]
    # x and y both appear high in BOTH lists -> they beat z and w
    assert set(ids[:2]) == {"x", "y"}
    assert ids[-1] in {"z", "w"}


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion([[_h("a")]])       # a is rank 1 in one list
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_empty_is_safe():
    assert reciprocal_rank_fusion([[], []]) == []


# ---- BM25 sparse index -------------------------------------------------
def test_sparse_finds_exact_keyword(dense):
    records = dense.store.records
    sparse = SparseIndex(records)
    hits = sparse.search("pentagon", k=3, where={"type": "question"})
    assert hits and all(h.type == "question" for h in hits)
    assert "pentagon" in hits[0].text.lower()          # exact term wins


def test_sparse_respects_metadata_filter(dense):
    sparse = SparseIndex(dense.store.records)
    for h in sparse.search("angle polygon", k=5, where={"type": "question"}):
        assert h.record["type"] == "question"


# ---- HybridRetriever integration --------------------------------------
def test_hybrid_returns_k_questions(dense):
    hybrid = HybridRetriever(dense, pool=10)
    hits = hybrid.search("interior angle of a regular polygon", k=3,
                         filters={"type": "question"})
    assert len(hits) == 3
    assert all(h.type == "question" for h in hits)     # leak protection holds


def test_hybrid_preserves_leak_protection(dense):
    hybrid = HybridRetriever(dense, pool=15)
    for h in hybrid.search("triangle parallel lines", k=5,
                           filters={"type": "question"}):
        assert h.type == "question"                    # no answers, either half


def test_hybrid_combines_both_signals(dense):
    # A keyword-heavy query: the sparse half must contribute.
    hybrid = HybridRetriever(dense, pool=10)
    hits = hybrid.search("pentagon inscribed in a circle", k=3,
                         filters={"type": "question"})
    assert hits[0].record["id"] == "very_hard_q2"      # the pentagon+circle Q
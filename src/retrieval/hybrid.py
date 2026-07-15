"""
hybrid.py — hybrid retrieval: dense (semantic) + sparse (BM25), fused by RRF.

WHY
Dense and sparse retrieval have MIRROR-IMAGE blind spots:
  dense  (embeddings): finds paraphrases, but can miss exact terms
  sparse (BM25):       nails exact terms, but is blind to paraphrases
Running both and fusing catches what either alone would miss — which is why
production RAG systems commonly use hybrid retrieval.

THE FUSION PROBLEM
Dense scores are cosine similarities (0..1); BM25 scores are unbounded.
You cannot just add them — different scales. RECIPROCAL RANK FUSION (RRF)
sidesteps this by using RANK POSITION, not raw score:
    rrf(item) = sum over lists of  1 / (k + rank_in_list)      # k ~ 60
An item near the top of BOTH lists wins; scale mismatch never matters.
No tuning, no score normalisation — this is why RRF is a production default.
"""

from __future__ import annotations

import re

from src.retrieval.retriever import Retriever, Hit

RRF_K = 60          # standard dampening constant


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseIndex:
    """BM25 keyword index over the chunk records — the sparse half.

    BM25 is the classic lexical ranking function: it scores by term overlap,
    weighting rare words more and dampening very long documents. It knows
    nothing about meaning — that is exactly what the dense half is for."""

    def __init__(self, records: list[dict]):
        from rank_bm25 import BM25Okapi
        self.records = records
        self._corpus_tokens = [_tokenize(r["text"]) for r in records]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int, where=None) -> list[Hit]:
        """Top-k by BM25, with the SAME metadata filtering as dense search
        (so leak protection and modes still hold on the sparse path)."""
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for i in ranked:
            rec = self.records[i]
            if where and not all(rec.get(kk) == vv for kk, vv in where.items()):
                continue
            if scores[i] <= 0:                       # no term overlap at all
                continue
            hits.append(Hit(score=float(scores[i]), record=rec))
            if len(hits) == k:
                break
        return hits


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], k: int = RRF_K,
                           top_k: int | None = None) -> list[Hit]:
    """
    Fuse several ranked Hit-lists into one by RRF.

    Each item's fused score = sum of 1/(k + rank) over the lists it appears
    in (rank is 1-based). Items strong in MULTIPLE lists rise to the top.
    Returns Hits with .score set to the fused RRF score.
    """
    fused: dict[str, float] = {}
    record_by_id: dict[str, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (k + rank)
            record_by_id[hit.id] = hit.record
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    result = [Hit(score=score, record=record_by_id[hid]) for hid, score in ordered]
    return result[:top_k] if top_k else result


class HybridRetriever:
    """
    Wraps a dense Retriever and adds a BM25 sparse index, fusing both with
    RRF. Same search() shape as Retriever, so callers are unchanged.
    """

    def __init__(self, retriever: Retriever, pool: int = 20):
        self.dense = retriever
        self.pool = pool                              # how deep each list goes
        # Build the sparse index over the SAME records the dense store holds.
        records = (self.dense.store.all_records()
                   if self.dense.backend == "chroma" else self.dense.store.records)
        self.sparse = SparseIndex(records)

    def search(self, query: str, k: int = 3, filters: dict | None = None,
               apply_threshold: bool = False) -> list[Hit]:
        """
        Data flow:
          query -> dense search (pool)  ┐
                -> sparse search (pool)  ├─ RRF fuse -> top-k
                                         ┘
        Metadata filters apply to BOTH halves, so leak protection holds.
        """
        dense_hits = self.dense.search(query, k=self.pool, filters=filters,
                                       apply_threshold=apply_threshold)
        sparse_hits = self.sparse.search(query, k=self.pool, where=filters)
        return reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=k)
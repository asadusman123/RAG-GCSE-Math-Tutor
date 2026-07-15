"""
retriever.py — query-time POLICY over the vector store.

Division of labour:
  VectorStore (Stage 3)  -> geometry: "what vectors are nearest?"
  Retriever   (here)     -> policy:   "what am I ALLOWED to return,
                                        is it close ENOUGH, and give me
                                        the PAIRED answer by exact id."

Nothing here knows about FAISS internals or tutor modes; it is the waist
of the hourglass. Tutor modes (Stage 5) sit on top; the store sits below.
"""

from dataclasses import dataclass

import numpy as np

from src.embeddings.embedder import get_embedder
from src.retrieval.vector_store import VectorStore

# Relevance floor, CALIBRATED to measurements on this corpus + MiniLM:
#   on-topic best ~0.56-0.68 ; off-topic ("biryani") best ~0.10.
# 0.30 sits in the empty gap between them. It is a tuned default, not a
# universal constant — re-measure if you swap the embedder or the document.
DEFAULT_MIN_SCORE = 0.30


def _to_chroma_where(filters: dict | None) -> dict | None:
    """Translate our simple {k: v} filters into Chroma's `where` syntax.
    Chroma needs an explicit $and for two or more conditions — a real API
    quirk worth knowing."""
    if not filters:
        return None
    if len(filters) == 1:
        return dict(filters)
    return {"$and": [{k: v} for k, v in filters.items()]}


@dataclass
class Hit:
    """One retrieval result: the chunk record plus its cosine score."""
    score: float
    record: dict            # full chunk: id, type, difficulty, marks, topic, pair_id, text

    # Convenience accessors so callers read hit.id, not hit.record["id"].
    @property
    def id(self) -> str: return self.record["id"]
    @property
    def type(self) -> str: return self.record["type"]
    @property
    def text(self) -> str: return self.record["text"]


class Retriever:
    def __init__(self, index_dir: str = "data/index", embedder=None,
                 min_score: float = DEFAULT_MIN_SCORE, backend: str = "faiss",
                 reranker=None, rerank_pool: int = 20):
        # Load the persisted index once; hold the embedder once (MiniLM
        # costs ~2s to load, so we never reload it per query).
        # backend="chroma" swaps the store for ChromaDB. Nothing else in this
        # class changes — that is the point of keeping the store boundary thin.
        self.backend = backend
        # STAGE 2 (optional): a cross-encoder that re-orders the shortlist.
        # rerank_pool = how wide a net stage 1 casts before re-ranking. Wider
        # pool = better recall for stage 2 to work with, at more compute.
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        if backend == "chroma":
            from src.retrieval.chroma_store import ChromaStore
            self.store = ChromaStore.load(index_dir)
        else:
            self.store = VectorStore.load(index_dir)
        self.embedder = embedder or get_embedder("auto")
        self.min_score = min_score
        # id -> record, for O(1) exact lookups (answer-by-pair_id).
        records = (self.store.all_records() if backend == "chroma"
                   else self.store.records)
        self._by_id = {r["id"]: r for r in records}

    # ---------------------------------------------------------------- helpers
    def _embed(self, text: str) -> np.ndarray:
        """Text -> one unit-length query vector (same rule as stored vectors)."""
        return self.embedder.embed([text])[0]

    @staticmethod
    def _passes(record: dict, filters: dict | None) -> bool:
        """True iff record matches EVERY key=value in filters (AND semantics)."""
        if not filters:
            return True
        return all(record.get(key) == value for key, value in filters.items())

    # ---------------------------------------------------------------- core API
    def search(self, query: str, k: int = 3, filters: dict | None = None,
               apply_threshold: bool = True) -> list[Hit]:
        """
        POST-FILTER strategy: over-fetch, then drop disallowed rows.

        Data flow:
          query text -> embed -> store.search(over_k) -> filter by metadata
                     -> drop below min_score -> keep top k

        Why over-fetch (k + slack, min 15): filtering removes rows, so we
        must fetch extra to still have k survivors. Slack is a safety
        margin; pre_search() below avoids the guessing entirely.
        """
        # Two-stage retrieval: when a re-ranker is attached, stage 1 fetches a
        # WIDE pool (recall), stage 2 re-orders it (precision), then we cut to k.
        if self.reranker is not None:
            pool = self._retrieve(query, k=self.rerank_pool, filters=filters,
                                  apply_threshold=apply_threshold)
            return self.reranker.rerank(query, pool, top_k=k)
        return self._retrieve(query, k=k, filters=filters,
                              apply_threshold=apply_threshold)

    def _retrieve(self, query: str, k: int, filters: dict | None,
                  apply_threshold: bool) -> list[Hit]:
        """STAGE 1 only: vector search + metadata filtering (no re-ranking)."""
        query_vector = self._embed(query)

        if self.backend == "chroma":
            # NATIVE PRE-FILTER: Chroma restricts the search space before
            # searching, so k results always come from the allowed set — no
            # over-fetching, no guessing at slack. This is the capability
            # FAISS lacks and the main reason to use a vector DB.
            raw = self.store.search(query_vector, k=k,
                                    where=_to_chroma_where(filters))
            return [Hit(score=s, record=r) for s, r in raw
                    if not apply_threshold or s >= self.min_score]

        # FAISS path: POST-FILTER — over-fetch, then drop disallowed rows.
        over_k = max(15, k * 5)                       # generous margin
        raw = self.store.search(query_vector, k=over_k)

        hits: list[Hit] = []
        for score, record in raw:
            if apply_threshold and score < self.min_score:
                continue                              # off-topic / too weak
            if not self._passes(record, filters):
                continue                              # wrong type/difficulty/topic
            hits.append(Hit(score=score, record=record))
            if len(hits) == k:
                break
        return hits

    def pre_search(self, query: str, k: int = 3, filters: dict | None = None,
                   apply_threshold: bool = True) -> list[Hit]:
        """
        PRE-FILTER strategy: restrict FIRST, then rank the allowed set.

        Exact (never comes up short), because we score the query against
        every allowed vector directly. Affordable here only because the
        corpus is tiny; at 10^6 vectors this is what a real vector DB does
        for you with a native filtered index.

        Because stored vectors are unit-length, cosine == dot product, so
        ranking is a single matrix-vector multiply.
        """
        if self.backend == "chroma":
            # Chroma pre-filters natively, so the hand-rolled workaround
            # below is unnecessary — search() IS pre-filtered.
            return self.search(query, k=k, filters=filters,
                               apply_threshold=apply_threshold)

        query_vector = self._embed(query)
        # Indices of rows allowed through the filter.
        allowed = [i for i, r in enumerate(self.store.records)
                   if self._passes(r, filters)]
        if not allowed:
            return []
        # Reconstruct the allowed vectors from FAISS and score them.
        matrix = np.vstack([self.store.index.reconstruct(i) for i in allowed])
        scores = matrix @ query_vector                # cosine, all at once
        order = np.argsort(scores)[::-1]              # high -> low
        hits: list[Hit] = []
        for rank in order:
            score = float(scores[rank])
            if apply_threshold and score < self.min_score:
                break                                 # sorted: rest are lower too
            hits.append(Hit(score=score, record=self.store.records[allowed[rank]]))
            if len(hits) == k:
                break
        return hits

    # ------------------------------------------------------- exact-id lookups
    def get(self, chunk_id: str) -> dict:
        """Fetch one chunk by exact id. Raises KeyError if absent."""
        return self._by_id[chunk_id]

    def get_answer_for(self, question_id: str) -> dict:
        """
        The mark scheme for a question — by pair_id, NEVER by search.
        This is the grounded, non-fuzzy path grading (Stage 5) depends on.
        """
        question = self._by_id[question_id]
        if question["type"] != "question":
            raise ValueError(f"{question_id} is not a question")
        return self._by_id[question["pair_id"]]

    # --------------------------------------------------------- catalogue view
    def all_questions(self, filters: dict | None = None) -> list[dict]:
        """
        Every question record (optionally filtered), in document order.
        Powers sequential mode and 'list topics/difficulties' in the tutor.
        No vector search — this is a metadata scan.
        """
        base = {"type": "question"}
        if filters:
            base.update(filters)
        if self.backend == "chroma":
            return self.store.all_records(where=_to_chroma_where(base))
        return [r for r in self.store.records if self._passes(r, base)]
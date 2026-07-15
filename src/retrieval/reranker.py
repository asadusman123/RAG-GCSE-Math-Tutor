"""
reranker.py — stage 2 of the two-stage retrieval pattern.

THE PROBLEM
Vector search uses a BI-ENCODER: query and chunk are embedded SEPARATELY,
then compared. Fast (chunk vectors are pre-computed) but lossy — the model
never sees the query and the chunk together, so fine-grained relevance is
crushed into 384 numbers.

THE FIX
A CROSS-ENCODER feeds (query, chunk) into the model TOGETHER and outputs a
relevance score. Its attention can weigh interactions between the query's
words and the chunk's words, so it is far more accurate. But it cannot
pre-compute anything: every pair needs a forward pass. Scoring a million
chunks per query is infeasible.

THE PATTERN (what production RAG actually does)
    millions of chunks
      -> STAGE 1 RETRIEVE  (bi-encoder + ANN)  fast, wide net: top-50
      -> STAGE 2 RE-RANK   (cross-encoder)     slow, accurate: reorder those 50
      -> top-5 into the prompt
Retrieval optimises RECALL (don't miss it). Re-ranking optimises PRECISION
(put it first). Cross-encoder accuracy at bi-encoder cost.

Three implementations behind one interface (same pattern as our embedders
and evaluators):
  CrossEncoderReranker — the real thing (ms-marco MiniLM cross-encoder).
  LexicalReranker      — offline; scores by query-term overlap. A genuinely
                         different signal from the embeddings, so it shows
                         what re-ranking DOES without a model download.
  StubReranker         — deterministic, for tests.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class Reranker(ABC):
    """Takes a query + candidate hits, returns them re-ordered by relevance."""

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text. Higher = more relevant. The one method
        a backend must implement."""

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        """
        Re-order `hits` (Retriever.Hit objects) by re-ranker score.

        Data flow: hits from stage 1 -> score each (query, hit.text) pair ->
        sort descending -> keep top_k. Each Hit's .score is REPLACED with the
        re-ranker score, so downstream code reads one consistent number.

        Complexity: O(n) model passes for n candidates — which is why you
        only ever re-rank a shortlist, never the whole corpus.
        """
        if not hits:
            return []
        scores = self.score(query, [h.text for h in hits])
        for hit, new_score in zip(hits, scores):
            hit.score = float(new_score)
        ranked = sorted(hits, key=lambda h: h.score, reverse=True)
        return ranked[:top_k] if top_k else ranked


class CrossEncoderReranker(Reranker):
    """
    The real re-ranker: a cross-encoder trained on MS MARCO (a large
    query/passage relevance dataset). Model is ~80MB, runs on CPU.

    Note it outputs a RELEVANCE LOGIT, not a cosine similarity — the scale
    differs from stage-1 scores (can be negative, unbounded). That is fine:
    we only use it to ORDER candidates, not to threshold them. Mixing the two
    scales without noticing is a classic re-ranking bug.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        # The model takes PAIRS — this is the whole point: query and text
        # are processed together, not separately.
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self.model.predict(pairs)]


class LexicalReranker(Reranker):
    """
    Offline re-ranker using term overlap (a BM25-flavoured signal).

    Why this is more than a test double: it scores on a DIFFERENT signal from
    the dense embeddings — exact term matching. Re-ranking with an independent
    signal is a real technique, and it demonstrates the core idea (reorder a
    shortlist by a second, sharper opinion) with zero downloads.
    """

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def score(self, query: str, texts: list[str]) -> list[float]:
        q_tokens = self._tokens(query)
        if not q_tokens:
            return [0.0] * len(texts)
        scores = []
        for text in texts:
            t_tokens = self._tokens(text)
            overlap = len(q_tokens & t_tokens)
            # normalise by query length so long chunks don't win automatically
            scores.append(overlap / len(q_tokens))
        return scores


class StubReranker(Reranker):
    """Deterministic: reverses the input order. Useless for relevance, but it
    PROVES the reranking stage actually reorders — a test double should make
    its effect unmistakable."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(i) for i in range(len(texts))]     # last becomes first
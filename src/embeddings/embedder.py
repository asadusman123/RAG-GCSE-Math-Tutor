"""
embedder.py — text -> unit-length vectors.

One interface, two backends (same trick as the ingestion loaders):

  SentenceTransformerEmbedder  DENSE, semantic. The real model
      (all-MiniLM-L6-v2)       (downloads ~80MB from HuggingFace on
                               first use; runs offline afterwards).

  HashingEmbedder              SPARSE-style, lexical. Deterministic,
                               zero dependencies — used as a test double
                               and as a live demonstration of sparse vs
                               dense retrieval.

CONTRACT (what the rest of the system relies on):
  embed(list[str]) -> np.ndarray of shape (len(texts), dim), float32,
  every row L2-normalised to unit length.
Unit length matters: it makes cosine similarity equal to a plain dot
product, so FAISS's inner-product index becomes a cosine index.
"""

import hashlib
import re

import numpy as np

DIM = 384  # MiniLM's native size; the hashing double matches it so the
           # FAISS index shape is identical whichever backend built it.


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length. Guard against divide-by-zero for
    pathological empty texts by treating zero-norm rows as norm 1."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    """Dense semantic embeddings via sentence-transformers (local model)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Import inside __init__, not at module top: the whole module stays
        # importable (tests, hashing backend) on machines without the
        # sentence-transformers package installed.
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()  # 384

    def embed(self, texts: list[str]) -> np.ndarray:
        # The library batches internally and can normalise for us; we
        # normalise ourselves anyway so the contract lives in ONE place.
        vectors = self.model.encode(texts, convert_to_numpy=True,
                                    show_progress_bar=False)
        return _l2_normalize(vectors)


class HashingEmbedder:
    """
    Lexical embeddings via the hashing trick (a.k.a. feature hashing).

    How it works: tokenize -> for each unigram and bigram, hash the token
    string to a bucket index in [0, DIM) -> count occurrences -> normalise.
    Texts sharing words/phrases share buckets => high dot product.
    Texts with disjoint vocabulary => ~0. No notion of MEANING: this is
    sparse retrieval wearing a dense vector's coat.

    Why md5 and not Python's hash(): hash() is salted per process for
    security (PYTHONHASHSEED), so results would differ between runs —
    a nondeterministic embedder corrupts any saved index. Classic gotcha.
    """

    dim = DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9°]+", text.lower())
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                digest = hashlib.md5(gram.encode()).digest()
                bucket = int.from_bytes(digest[:4], "little") % DIM
                matrix[row, bucket] += 1.0
        return _l2_normalize(matrix)


def get_embedder(prefer: str = "auto"):
    """
    Factory. prefer = "dense" | "hashing" | "auto".
    "auto": use the real model if available, else fall back loudly —
    a silent fallback would let a lexical index masquerade as semantic.
    """
    if prefer == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:
        if prefer == "dense":
            raise RuntimeError(
                "Dense embedder unavailable — pip install sentence-transformers"
            ) from exc
        print(f"[embedder] WARNING: falling back to HashingEmbedder ({type(exc).__name__})")
        return HashingEmbedder()

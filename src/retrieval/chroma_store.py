"""
chroma_store.py — the same store interface, backed by ChromaDB.

WHY THIS EXISTS
FAISS is a similarity-search LIBRARY: it stores vectors and finds nearest
neighbours, nothing else. Chroma is a vector DATABASE: it wraps that same
kind of search with storage, IDs, metadata, persistence, and — crucially —
NATIVE METADATA PRE-FILTERING.

What Chroma removes from our hand-rolled FAISS store:
  FAISS store                          Chroma
  ------------------------------       ---------------------------------
  chunks.json metadata sidecar    ->   documents + metadata stored inline
  drift-guard (ntotal == len)     ->   impossible: one record per item
  manual row-position -> id map   ->   real string IDs
  save()/load() to disk           ->   PersistentClient persists for you
  post-filter (fetch 15, drop)    ->   where={...} pre-filters natively
  pre_search() using FAISS
    index.reconstruct() internals ->   deleted — native filtering does it

Interface matches VectorStore so Retriever cannot tell them apart:
  add(vectors, records) / search(query_vector, k, where) / load(dir)
That interchangeability is the payoff of keeping the store boundary thin.

Cosine note: Chroma returns DISTANCES, not similarities. With the cosine
space, similarity = 1 - distance. We convert so scores mean the same thing
as they did with FAISS (higher = closer), keeping the Retriever's threshold
logic valid across both backends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLLECTION = "chunks"


class ChromaStore:
    def __init__(self, directory: str | Path = "data/chroma"):
        import chromadb
        self.directory = str(directory)
        # PersistentClient writes to disk automatically — no save()/load().
        self.client = chromadb.PersistentClient(path=self.directory)
        # A "collection" is Chroma's table. hnsw:space=cosine tells it to
        # rank by cosine distance (matching our unit-norm embeddings).
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"})

    # ------------------------------------------------------------ write
    def add(self, vectors: np.ndarray, records: list[dict]) -> None:
        """
        Store vectors WITH their text and metadata in one call.

        Contrast with FAISS: there we added vectors to the index and kept a
        parallel records list, then persisted a separate chunks.json and
        wrote a guard against the two drifting apart. Here the id, document,
        metadata and embedding are ONE record — drift is impossible.

        Chroma requires metadata values to be scalars (str/int/float/bool),
        so we flatten the `pages` list to a comma-separated string.
        """
        if vectors.shape[0] != len(records):
            raise ValueError(f"{vectors.shape[0]} vectors vs {len(records)} records")

        ids = [r["id"] for r in records]
        documents = [r["text"] for r in records]
        metadatas = []
        for r in records:
            meta = {k: v for k, v in r.items() if k != "text"}
            meta["pages"] = ",".join(str(p) for p in meta.get("pages", []))
            metadatas.append(meta)

        self.collection.upsert(                      # upsert = idempotent add
            ids=ids,
            embeddings=[v.tolist() for v in vectors],
            documents=documents,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------- read
    def search(self, query_vector: np.ndarray, k: int = 5,
               where: dict | None = None) -> list[tuple[float, dict]]:
        """
        k nearest records, optionally PRE-FILTERED by metadata.

        `where={"type": "question"}` restricts the search space BEFORE
        searching — so we always get k results from the allowed set. This is
        the thing FAISS could not do natively, and the reason our FAISS path
        needed post-filtering (fetch extra, drop, hope enough survive).

        Returns [(similarity, record)] — same shape as VectorStore.search,
        so the Retriever is unchanged.
        """
        result = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(k, max(1, self.count())),
            where=where or None,                     # None == no filter
            include=["documents", "metadatas", "distances"],
        )
        hits: list[tuple[float, dict]] = []
        for doc, meta, dist in zip(result["documents"][0],
                                   result["metadatas"][0],
                                   result["distances"][0]):
            record = dict(meta)
            record["text"] = doc                     # rebuild the full record
            record["pages"] = [int(p) for p in str(meta.get("pages", "")).split(",") if p]
            similarity = 1.0 - float(dist)           # cosine distance -> similarity
            hits.append((similarity, record))
        return hits

    # --------------------------------------------------------- catalogue
    def all_records(self, where: dict | None = None) -> list[dict]:
        """Every stored record (optionally filtered) — a metadata scan, no
        vector search. Powers sequential mode and the modes' question lists."""
        result = self.collection.get(where=where or None,
                                     include=["documents", "metadatas"])
        records = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            record = dict(meta)
            record["text"] = doc
            record["pages"] = [int(p) for p in str(meta.get("pages", "")).split(",") if p]
            records.append(record)
        # Chroma does not guarantee insertion order; sort for deterministic
        # sequential mode (difficulty order, then question number).
        order = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}
        records.sort(key=lambda r: (order.get(r.get("difficulty"), 9),
                                    r.get("question_number", 0),
                                    r.get("type") != "question"))
        return records

    def count(self) -> int:
        return self.collection.count()

    # ------------------------------------------------------- persistence
    def save(self, directory: str | Path | None = None) -> None:
        """No-op: PersistentClient already wrote to disk. Kept so the
        interface matches VectorStore (the Retriever calls neither, but
        pipeline.py does)."""
        return None

    @classmethod
    def load(cls, directory: str | Path = "data/chroma") -> "ChromaStore":
        """Open an existing Chroma collection. Raises if it is empty, so a
        missing/unbuilt index fails loudly rather than returning nothing."""
        store = cls(directory)
        if store.count() == 0:
            raise RuntimeError(
                f"Chroma collection at {directory} is empty — "
                "run: python -m src.ingestion.pipeline --chroma")
        return store
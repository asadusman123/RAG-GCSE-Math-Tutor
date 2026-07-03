"""
vector_store.py — a thin, honest wrapper around FAISS.

Responsibility: store (vector, record) pairs; given a query vector,
return the k most similar records with scores; persist/restore to disk.
NOT its job: embedding text (embedder.py) or deciding what to search
for and how to filter (retriever.py, Stage 4). Thin layers swap easily —
this is the only file that would change in a Qdrant/Pinecone migration.

Design notes:
- IndexFlatIP = exact inner-product search. Our vectors are unit-length,
  so inner product == cosine similarity. Exact is correct at our scale
  (26 vectors); approximate indexes (IVF/HNSW) earn their complexity
  around 10^5+ vectors.
- FAISS returns ROW POSITIONS, not IDs. self.records maps position ->
  chunk metadata, and is persisted as a JSON sidecar. The load() check
  enforces that index and sidecar never drift apart.
"""

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # IP = inner product
        self.records: list[dict] = []         # row position -> metadata

    # ------------------------------------------------------------ write
    def add(self, vectors: np.ndarray, records: list[dict]) -> None:
        """Append vectors + their records, keeping both in lockstep."""
        if vectors.shape[0] != len(records):
            raise ValueError(f"{vectors.shape[0]} vectors vs {len(records)} records")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != index dim {self.dim}")
        # FAISS is a C++ library with strict expectations: float32,
        # C-contiguous memory. np.ascontiguousarray is cheap insurance.
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.records.extend(records)

    # ------------------------------------------------------------- read
    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[float, dict]]:
        """
        k nearest records for ONE query vector.
        Returns [(cosine_score, record), ...] best-first.
        """
        k = min(k, len(self.records))          # can't return more than we hold
        if k == 0:
            return []
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, positions = self.index.search(q, k)   # shapes: (1, k), (1, k)
        return [(float(s), self.records[p])
                for s, p in zip(scores[0], positions[0])
                if p != -1]                    # -1 pads impossible slots

    # ------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "vectors.faiss"))
        records = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if index.ntotal != len(records):       # the drift guard
            raise RuntimeError(
                f"Corrupt index dir: {index.ntotal} vectors but "
                f"{len(records)} records — rebuild with pipeline.py")
        store = cls(dim=index.d)
        store.index = index
        store.records = records
        return store

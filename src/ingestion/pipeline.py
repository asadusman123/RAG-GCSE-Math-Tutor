"""
pipeline.py — the offline ingestion entry point. Runs ONCE per document.

    vision pages -> parse -> validate -> embed -> VectorStore -> data/index/

Every piece already exists; this file only orchestrates, which is why
it is short. Orchestrators should be boring.

Run:  python -m src.ingestion.pipeline
"""

from dataclasses import asdict
from pathlib import Path

from src.embeddings.embedder import get_embedder
from src.ingestion.parser import parse, validate
from src.ingestion.vision_loader import load_transcribed_pages
from src.retrieval.vector_store import VectorStore

INDEX_DIR = Path("data/index")


def build_index(embedder=None, index_dir: str | Path = INDEX_DIR,
                backend: str = "faiss"):
    """
    Returns the built store (handy for tests) and persists it to disk.

    Refuses to index invalid chunks: an index built from a broken parse
    would poison every downstream answer, so we fail here, loudly.
    """
    chunks = parse(load_transcribed_pages())
    problems = validate(chunks)
    if problems:
        raise RuntimeError(f"Refusing to index invalid chunks: {problems}")

    embedder = embedder or get_embedder("auto")
    # Embed the chunk TEXT; store the full chunk (text + metadata) as the
    # record, so a search hit needs no second lookup anywhere.
    vectors = embedder.embed([c.text for c in chunks])

    if backend == "chroma":
        # Chroma stores vectors + documents + metadata together and persists
        # itself — no dim argument, no save() needed.
        from src.retrieval.chroma_store import ChromaStore
        store = ChromaStore(index_dir)
    else:
        store = VectorStore(dim=vectors.shape[1])

    store.add(vectors, [asdict(c) for c in chunks])
    store.save(index_dir)
    print(f"Indexed {len(chunks)} chunks -> {index_dir}/  (backend={backend})")
    return store


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma", action="store_true",
                    help="build a ChromaDB index instead of FAISS")
    args = ap.parse_args()
    if args.chroma:
        build_index(index_dir="data/chroma", backend="chroma")
    else:
        build_index()
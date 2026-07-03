"""
Contract tests for the embedding + vector-store layer.

All tests use HashingEmbedder: deterministic, no model download, and its
lexical behaviour is predictable enough to assert orderings against.
The dense backend obeys the same contracts (shape, norms, determinism);
run these locally with SentenceTransformerEmbedder to confirm.
"""

import numpy as np
import pytest

from src.embeddings.embedder import HashingEmbedder, DIM
from src.ingestion.pipeline import build_index
from src.retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def emb():
    return HashingEmbedder()


def test_shape_norm_and_determinism(emb):
    texts = ["interior angles", "exterior angles of a polygon"]
    a, b = emb.embed(texts), emb.embed(texts)
    assert a.shape == (2, DIM) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(a, b)          # md5, not salted hash(): stable


def test_similarity_ordering(emb):
    v = emb.embed(["interior angle of a regular polygon",
                   "regular polygon interior angles",
                   "recipe for tomato pasta sauce"])
    paraphrase, unrelated = v[0] @ v[1], v[0] @ v[2]
    assert paraphrase > unrelated
    assert unrelated < 0.1


def test_store_roundtrip_and_self_retrieval(tmp_path, emb):
    texts = ["alpha beta gamma", "polygon angles", "parallel lines"]
    vectors = emb.embed(texts)
    store = VectorStore(dim=DIM)
    store.add(vectors, [{"id": f"c{i}"} for i in range(3)])
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    # A chunk's own text must retrieve itself first with cosine ~ 1.0
    score, record = loaded.search(vectors[1], k=1)[0]
    assert record["id"] == "c1"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_lockstep_violations_raise(tmp_path, emb):
    store = VectorStore(dim=DIM)
    with pytest.raises(ValueError):                       # count mismatch
        store.add(emb.embed(["one text"]), [{"id": "a"}, {"id": "b"}])
    with pytest.raises(ValueError):                       # dim mismatch
        store.add(np.ones((1, 7), dtype=np.float32), [{"id": "a"}])


def test_pipeline_end_to_end(tmp_path, emb):
    store = build_index(embedder=emb, index_dir=tmp_path)
    assert store.index.ntotal == 26                       # 13 Q + 13 A
    hits = store.search(
        emb.embed(["interior angle of a regular polygon with 40 sides"])[0], k=3)
    assert hits[0][1]["id"] == "easy_q1"                  # the right chunk wins
    # every record kept full metadata through the pipeline
    assert {"id", "type", "difficulty", "marks", "topic", "pair_id", "text"} \
        <= set(hits[0][1].keys())

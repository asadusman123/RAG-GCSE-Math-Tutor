"""
Retriever contract tests.

Uses HashingEmbedder + a low threshold so assertions are deterministic
and network-free. The INVARIANTS asserted here (never return an answer as
a question; pre==post; answer-by-id is exact) hold for any embedder; the
exact scores do not, which is why we assert structure, not magnitudes.
"""

import numpy as np
import pytest

from src.embeddings.embedder import HashingEmbedder
from src.ingestion.pipeline import build_index
from src.retrieval.retriever import Retriever, Hit


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    # Build a throwaway index with the deterministic embedder, then point
    # a Retriever at it. min_score=0.0 disables threshold for most tests so
    # lexical scores don't accidentally filter everything out.
    d = tmp_path_factory.mktemp("index")
    emb = HashingEmbedder()
    build_index(embedder=emb, index_dir=d)
    return Retriever(index_dir=str(d), embedder=emb, min_score=0.0)


def test_question_filter_never_returns_an_answer(retriever):
    # THE invariant motivated by the play.py experiments: answers must
    # never surface when we ask for questions.
    for query in ["polygon interior angle", "parallel lines triangle",
                  "simultaneous equations", "regular pentagon"]:
        for h in retriever.search(query, k=5, filters={"type": "question"}):
            assert h.type == "question", (query, h.id)


def test_pre_and_post_filter_agree(retriever):
    # Two independent strategies -> same ranking. If they ever diverge,
    # one of them has a bug.
    q = "interior angle of a regular polygon"
    post = retriever.search(q, k=3, filters={"type": "question"})
    pre = retriever.pre_search(q, k=3, filters={"type": "question"})
    assert [h.id for h in post] == [h.id for h in pre]


def test_difficulty_and_topic_filters(retriever):
    hard = retriever.all_questions(filters={"difficulty": "hard"})
    assert len(hard) == 4 and all(r["difficulty"] == "hard" for r in hard)
    polys = retriever.all_questions(filters={"topic": "polygons"})
    assert polys and all(r["topic"] == "polygons" for r in polys)


def test_threshold_rejects_offtopic():
    # With a realistic floor, a nonsense query yields nothing rather than
    # a forced bad match. Uses the hashing embedder where 'biryani' shares
    # no vocabulary with the corpus -> ~0 score -> below any positive floor.
    import tempfile
    from src.embeddings.embedder import HashingEmbedder
    with tempfile.TemporaryDirectory() as d:
        emb = HashingEmbedder()
        build_index(embedder=emb, index_dir=d)
        r = Retriever(index_dir=d, embedder=emb, min_score=0.3)
        assert r.search("how do I make biryani", k=3,
                        filters={"type": "question"}) == []


def test_answer_by_id_is_exact(retriever):
    for q in retriever.all_questions():
        ans = retriever.get_answer_for(q["id"])
        assert ans["id"] == q["pair_id"]
        assert ans["type"] == "answer"
        assert ans["text"].strip()
    with pytest.raises(ValueError):          # asking an answer for its answer
        retriever.get_answer_for("hard_q1_ans")


def test_hit_accessors(retriever):
    h = retriever.search("polygon", k=1, filters={"type": "question"})[0]
    assert h.id == h.record["id"] and h.text == h.record["text"]

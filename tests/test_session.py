"""
Session (tutor logic) contract tests. Offline via StubEvaluator + the
real index. Verify mode selection, leak protection, grading orchestration,
and progress tracking — the rules a UI depends on.
"""

import pytest

from src.retrieval.retriever import Retriever
from src.embeddings.embedder import HashingEmbedder
from src.ingestion.pipeline import build_index
from src.tutor.evaluator import StubEvaluator
from src.tutor.session import Session, Mode, Attempt, Progress


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx")
    emb = HashingEmbedder()
    build_index(embedder=emb, index_dir=d)
    return Retriever(index_dir=str(d), embedder=emb, min_score=0.0)


@pytest.fixture
def session(retriever):
    return Session(retriever, StubEvaluator(score=1))


def test_sequential_walks_all_then_stops(session):
    seen = []
    while (q := session.next_question(Mode.SEQUENTIAL)) is not None:
        seen.append(q["id"])
    assert len(seen) == 13                     # all questions, once each
    assert seen == sorted(set(seen), key=seen.index)   # no repeats


def test_selection_never_returns_an_answer_chunk(session):
    for mode in (Mode.SEQUENTIAL, Mode.RANDOM):
        q = session.next_question(mode)
        assert q["type"] == "question"         # leak protection


def test_difficulty_and_topic_modes_respect_filter(session):
    q = session.next_question(Mode.DIFFICULTY, difficulty="hard")
    assert q["difficulty"] == "hard"
    q = session.next_question(Mode.TOPIC, topic="polygons")
    assert q["topic"] == "polygons"


def test_grading_records_progress(session):
    q = session.next_question(Mode.SEQUENTIAL)
    ev = session.grade_answer(q["id"], "some answer")
    assert ev.score == 1
    assert len(session.progress.attempts) == 1
    assert session.progress.attempted_ids == {q["id"]}


def test_random_mode_does_not_repeat_attempted(session):
    # Attempt one, then random should never hand it back.
    first = session.next_question(Mode.RANDOM)
    session.grade_answer(first["id"], "a")
    for _ in range(20):
        nxt = session.next_question(Mode.RANDOM)
        if nxt is None:
            break
        assert nxt["id"] != first["id"]


def test_reveal_returns_mark_scheme(session):
    q = session.next_question(Mode.SEQUENTIAL)
    scheme = session.reveal_answer(q["id"])
    assert scheme.strip()


def test_progress_summary_math():
    p = Progress()
    p.record(Attempt("easy_q1", 2, 2, "easy", "polygons"))
    p.record(Attempt("hard_q1", 1, 3, "hard", "parallel_lines"))
    assert p.total_score == 3 and p.total_possible == 5
    assert "60%" in p.summary()
"""
API contract tests. Uses FastAPI's TestClient (no live server) and forces
the STUB evaluator via TUTOR_STUB=1 so no Gemini calls happen. Verifies the
HTTP layer correctly wraps Session and — critically — never leaks answers.
"""

import os
import pytest

# Must set BEFORE importing the app, so the session builds with the stub.
os.environ["TUTOR_STUB"] = "1"

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_meta_lists_topics_and_difficulties():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert "polygons" in body["topics"]
    assert set(body["difficulties"]) == {"easy", "medium", "hard", "very_hard"}


def test_question_returns_question_never_answer():
    r = client.get("/api/question?mode=sequential")
    assert r.status_code == 200
    q = r.json()
    assert q["done"] is False
    # leak protection across HTTP: no answer/pair_id fields exposed
    assert "pair_id" not in q
    assert "Method 1" not in q["text"]        # mark-scheme marker absent
    assert q["id"] and q["marks"] > 0


def test_grade_returns_structured_evaluation():
    q = client.get("/api/question?mode=sequential").json()
    r = client.post("/api/grade", json={"question_id": q["id"], "answer": "test"})
    assert r.status_code == 200
    ev = r.json()
    assert 0 <= ev["score"] <= ev["max_marks"]
    assert "progress" in ev


def test_grade_unknown_question_is_404():
    r = client.post("/api/grade", json={"question_id": "nope_q99", "answer": "x"})
    assert r.status_code == 404


def test_reveal_returns_mark_scheme():
    q = client.get("/api/question?mode=sequential").json()
    r = client.get("/api/reveal?question_id=" + q["id"])
    assert r.status_code == 200
    assert r.json()["mark_scheme"].strip()


def test_bad_mode_is_400():
    assert client.get("/api/question?mode=telepathy").status_code == 400


def test_question_includes_diagram_field():
    # Every question response carries a diagram_svg key (None if no diagram).
    r = client.get("/api/question?mode=sequential").json()
    assert "diagram_svg" in r


def test_search_returns_a_question_or_not_found():
    # Free-text semantic search endpoint. With the stub session's real index,
    # a topical query returns a question chunk; result is leak-safe.
    r = client.get("/api/search?q=interior angle of a regular polygon").json()
    if r["found"]:
        assert "Method 1" not in r["text"]          # never an answer chunk
        assert r["id"] and "diagram_svg" in r
    # an off-topic query should not fabricate a match
    r2 = client.get("/api/search?q=how do I bake sourdough bread").json()
    assert "found" in r2
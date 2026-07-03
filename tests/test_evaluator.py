"""
Evaluator + prompt contract tests. All offline (StubEvaluator) — no API
key, no network. They verify the logic that must hold for ANY backend:
context injection, JSON parsing, and score clamping.
"""

import json
import pytest

from src.tutor.prompts import build_grading_messages
from src.tutor.evaluator import StubEvaluator, Evaluator, Evaluation


# ---- prompt construction (context injection) -------------------------
def test_mark_scheme_is_injected_into_prompt():
    _, user = build_grading_messages(
        question="Q", mark_scheme="THE_OFFICIAL_SCHEME_TEXT",
        max_marks=3, student_answer="my answer")
    assert "THE_OFFICIAL_SCHEME_TEXT" in user      # the RAG grounding
    assert "<mark_scheme>" in user                 # fenced unambiguously
    assert "my answer" in user
    assert "0 to 3" in user                        # ceiling stated to model


def test_blank_answer_becomes_explicit_marker():
    _, user = build_grading_messages("Q", "scheme", 2, "   ")
    assert "[NO ANSWER GIVEN]" in user


# ---- parsing + validation (defending against the model) --------------
def test_full_pipeline_via_stub():
    ev = StubEvaluator(score=1)
    result = ev.grade("Q", "scheme", max_marks=2, student_answer="ans")
    assert isinstance(result, Evaluation)
    assert result.score == 1 and result.max_marks == 2
    assert result.correct_points and result.hint


def test_score_is_clamped_to_max():
    # Stub always claims max_marks 999; grade() must clamp to the real max.
    ev = StubEvaluator(score=50)
    assert ev.grade("Q", "s", max_marks=2, student_answer="a").score == 2


def test_negative_score_clamped_to_zero():
    ev = StubEvaluator(score=-5)
    assert ev.grade("Q", "s", max_marks=2, student_answer="a").score == 0


def test_json_fences_are_stripped():
    # Simulate a model that wraps JSON in markdown fences (common).
    class FencedEvaluator(Evaluator):
        def _complete(self, s, u):
            return '```json\n{"score": 1, "explanation": "ok"}\n```'
    result = FencedEvaluator().grade("Q", "s", max_marks=3, student_answer="a")
    assert result.score == 1 and result.explanation == "ok"


def test_invalid_json_raises_loudly():
    class BrokenEvaluator(Evaluator):
        def _complete(self, s, u):
            return "I cannot grade this, sorry!"     # not JSON at all
    with pytest.raises(ValueError, match="valid JSON"):
        BrokenEvaluator().grade("Q", "s", max_marks=2, student_answer="a")


def test_fraction_property():
    assert StubEvaluator(score=1).grade("Q","s",4,"a").fraction == 0.25
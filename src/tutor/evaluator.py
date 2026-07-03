"""
evaluator.py — grades a student answer against the retrieved mark scheme.

Architecture (provider abstraction — the key lesson of this stage):

    Evaluator (abstract)         defines grade() + parsing/validation.
      ├── StubEvaluator          deterministic, offline — powers tests.
      └── GeminiEvaluator        real LLM; ONE method to fill from Google's
                                 docs (clearly marked). Swapping providers
                                 = adding one subclass, nothing else changes.

The LLM is NOT the source of truth about the answer — the mark scheme
(injected by prompts.py) is. This module's job is to call the model, then
DEFEND against its output: parse the JSON safely and clamp the score into
the legal range. Never trust raw model text.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.tutor.prompts import build_grading_messages


@dataclass
class Evaluation:
    """The structured grade returned to the tutor/UI."""
    score: int
    max_marks: int
    correct_points: list[str] = field(default_factory=list)
    missed_points: list[str] = field(default_factory=list)
    hint: str = ""
    explanation: str = ""

    @property
    def fraction(self) -> float:
        """Score as a 0..1 fraction — handy for progress tracking."""
        return self.score / self.max_marks if self.max_marks else 0.0


class Evaluator(ABC):
    """
    Base class. Subclasses implement ONLY _complete() — the raw LLM call.
    Everything else (prompt building, JSON parsing, score validation) is
    shared here so every backend behaves identically and safely.
    """

    @abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompts to the model, return its raw text reply. The single
        provider-specific method — the seam."""

    def grade(self, question: str, mark_scheme: str, max_marks: int,
              student_answer: str) -> Evaluation:
        """
        Full grading flow: build prompt -> call model -> parse -> validate.
        This is what the tutor calls; it never touches provider details.
        """
        system_prompt, user_prompt = build_grading_messages(
            question, mark_scheme, max_marks, student_answer)
        raw = self._complete(system_prompt, user_prompt)
        return self._parse_and_validate(raw, max_marks)

    @staticmethod
    def _parse_and_validate(raw: str, max_marks: int) -> Evaluation:
        """
        Turn the model's raw text into a validated Evaluation.

        Defensive on purpose — LLMs misbehave in predictable ways:
          - wrap JSON in ```json fences  -> we strip them
          - emit a score above max/below 0 -> we clamp it
          - omit a field                  -> dataclass defaults cover it
        A parsing failure raises ValueError with the raw text, so a broken
        backend is loud, not silently wrong.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # remove a leading ```json / ``` and the trailing ```
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON:\n{raw}") from exc

        # Clamp the score into [0, max_marks] — the guardrail in code, not
        # trusting the prompt alone to have been obeyed.
        score = int(data.get("score", 0))
        score = max(0, min(score, max_marks))

        return Evaluation(
            score=score,
            max_marks=max_marks,
            correct_points=list(data.get("correct_points", [])),
            missed_points=list(data.get("missed_points", [])),
            hint=str(data.get("hint", "")),
            explanation=str(data.get("explanation", "")),
        )


class StubEvaluator(Evaluator):
    """
    Offline, deterministic evaluator for tests and no-key development.
    Returns a fixed, valid JSON payload so the FULL pipeline (prompt ->
    parse -> validate -> Evaluation) can be exercised without a network
    call or an API key. Same test-double principle as HashingEmbedder.
    """

    def __init__(self, score: int = 1):
        self._score = score

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({
            "score": self._score,
            "max_marks": 999,   # deliberately wrong -> proves clamping works
            "correct_points": ["stub: recognised the correct final value"],
            "missed_points": ["stub: did not show full method"],
            "hint": "stub: state the formula you used",
            "explanation": "Stubbed evaluation for testing.",
        })


class GeminiEvaluator(Evaluator):
    """
    Real grader backed by Google's Gemini API (free tier).

    Reads GEMINI_API_KEY from the environment (loaded from .env by
    python-dotenv). The ONLY provider-specific code is _complete() below.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        from dotenv import load_dotenv
        load_dotenv()                       # read .env into os.environ
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY not found. Put it in a .env file at the "
                "project root: GEMINI_API_KEY=your-key")
        self.model = model
        # The google-genai client auto-detects GEMINI_API_KEY from the env.
        from google import genai
        self._client = genai.Client()

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        # ============================================================
        # >>> GEMINI SEAM — Implemented using google-genai SDK <<<
        # ============================================================
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        
        if not response.text:
            raise RuntimeError("Gemini returned an empty response or was blocked by safety settings.")
            
        return response.text
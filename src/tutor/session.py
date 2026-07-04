"""
session.py — the tutor's brain. Framework-independent: knows about modes,
question selection, grading orchestration, and progress. Knows NOTHING
about terminals, FastAPI, or any UI. A front end (cli.py, or a web app)
drives a Session by calling its methods.

Why this separation: swap the CLI for a web UI and this file is untouched.
All tutoring rules live here, in one testable place.

Data flow for one round:
  next_question(mode)         -> a question record (NO answer attached)
  ...front end shows it, collects the student's typed answer...
  grade_answer(qid, answer)   -> fetch mark scheme BY ID -> Evaluator ->
                                 record the result -> return Evaluation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from src.retrieval.retriever import Retriever
from src.tutor.evaluator import Evaluator, Evaluation


class Mode(str, Enum):
    """The four required selection modes. `str` mix-in lets us compare to
    plain strings and serialise easily."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TOPIC = "topic"
    DIFFICULTY = "difficulty"


@dataclass
class Attempt:
    """One graded attempt — the unit of progress tracking."""
    question_id: str
    score: int
    max_marks: int
    difficulty: str
    topic: str


@dataclass
class Progress:
    """Aggregates attempts into a simple performance picture."""
    attempts: list[Attempt] = field(default_factory=list)

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    @property
    def attempted_ids(self) -> set[str]:
        return {a.question_id for a in self.attempts}

    @property
    def total_score(self) -> int:
        return sum(a.score for a in self.attempts)

    @property
    def total_possible(self) -> int:
        return sum(a.max_marks for a in self.attempts)

    def summary(self) -> str:
        if not self.attempts:
            return "No questions attempted yet."
        pct = (100 * self.total_score / self.total_possible
               if self.total_possible else 0)
        lines = [f"Attempted {len(self.attempts)} question(s): "
                 f"{self.total_score}/{self.total_possible} marks ({pct:.0f}%)"]
        # per-difficulty breakdown
        by_diff: dict[str, list[Attempt]] = {}
        for a in self.attempts:
            by_diff.setdefault(a.difficulty, []).append(a)
        for diff, items in by_diff.items():
            got = sum(a.score for a in items)
            pos = sum(a.max_marks for a in items)
            lines.append(f"  {diff}: {got}/{pos} across {len(items)} question(s)")
        return "\n".join(lines)


class Session:
    """Orchestrates a tutoring session over one index + one evaluator."""

    def __init__(self, retriever: Retriever, evaluator: Evaluator):
        self.retriever = retriever
        self.evaluator = evaluator
        self.progress = Progress()
        self._seq_pos = 0                      # cursor for sequential mode

    # ---------------------------------------------------------- selection
    def next_question(self, mode: Mode = Mode.SEQUENTIAL,
                      topic: str | None = None,
                      difficulty: str | None = None) -> dict | None:
        """
        Return the next question record for the chosen mode, or None when
        exhausted. The returned dict is the QUESTION chunk only — its
        answer is never attached (leak protection).

        No vector search here: selecting *which* question to ask is a
        metadata operation, not a similarity one.
        """
        if mode == Mode.SEQUENTIAL:
            questions = self.retriever.all_questions()
            if self._seq_pos >= len(questions):
                return None                    # finished the set
            q = questions[self._seq_pos]
            self._seq_pos += 1
            return q

        if mode == Mode.TOPIC:
            pool = self.retriever.all_questions(filters={"topic": topic})
        elif mode == Mode.DIFFICULTY:
            pool = self.retriever.all_questions(filters={"difficulty": difficulty})
        else:  # RANDOM
            pool = self.retriever.all_questions()

        # For non-sequential modes, avoid repeating already-attempted ones.
        remaining = [q for q in pool if q["id"] not in self.progress.attempted_ids]
        if not remaining:
            return None
        return random.choice(remaining)

    # ------------------------------------------------------------ grading
    def grade_answer(self, question_id: str, student_answer: str) -> Evaluation:
        """
        Grade one answer and record the result.

        Fetches the mark scheme BY ID (exact, never by search), grades via
        the evaluator, then logs an Attempt for progress tracking.
        """
        question = self.retriever.get(question_id)
        answer_chunk = self.retriever.get_answer_for(question_id)   # by pair_id

        evaluation = self.evaluator.grade(
            question=question["text"],
            mark_scheme=answer_chunk["text"],
            max_marks=question["marks"],
            student_answer=student_answer,
        )
        self.progress.record(Attempt(
            question_id=question_id,
            score=evaluation.score,
            max_marks=evaluation.max_marks,
            difficulty=question["difficulty"],
            topic=question["topic"],
        ))
        return evaluation

    # -------------------------------------------------------- reveal / help
    def reveal_answer(self, question_id: str) -> str:
        """The official mark scheme text, for 'show me the answer'."""
        return self.retriever.get_answer_for(question_id)["text"]

    def available_topics(self) -> list[str]:
        return sorted({q["topic"] for q in self.retriever.all_questions()})

    def available_difficulties(self) -> list[str]:
        return sorted({q["difficulty"] for q in self.retriever.all_questions()})
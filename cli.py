"""
cli.py — the terminal front end for the RAG tutor.

Responsibility: ONLY user interaction (printing, reading input). All
tutoring decisions are delegated to Session. This file could be replaced
by a web UI without changing a line of tutor logic.

Run:  python cli.py
      python cli.py --mode difficulty --difficulty hard
      python cli.py --stub          (grade offline, no API key/cost)
"""

import argparse
import sys

from src.retrieval.retriever import Retriever
from src.tutor.evaluator import GeminiEvaluator, StubEvaluator
from src.tutor.session import Session, Mode


def build_session(use_stub: bool) -> Session:
    """Wire the real components together. --stub swaps the grader for the
    offline evaluator so you can demo the flow without an API key."""
    retriever = Retriever()                    # loads data/index/
    evaluator = StubEvaluator() if use_stub else GeminiEvaluator()
    return Session(retriever, evaluator)


def print_evaluation(ev) -> None:
    """Render an Evaluation for the terminal."""
    print(f"\n  SCORE: {ev.score}/{ev.max_marks}")
    if ev.correct_points:
        print("  What you got right:")
        for p in ev.correct_points:
            print(f"    + {p}")
    if ev.missed_points:
        print("  What was missing:")
        for p in ev.missed_points:
            print(f"    - {p}")
    if ev.hint:
        print(f"  Hint: {ev.hint}")
    if ev.explanation:
        print(f"  Why: {ev.explanation}")


def run(mode: Mode, topic: str | None, difficulty: str | None,
        use_stub: bool) -> None:
    session = build_session(use_stub)
    print("=" * 60)
    print("  RAG TUTOR — answer the question, get graded from the")
    print("  official mark scheme. Type 'skip', 'reveal', or 'quit'.")
    print("=" * 60)

    while True:
        question = session.next_question(mode, topic=topic, difficulty=difficulty)
        if question is None:
            print("\nNo more questions in this mode. Well done!")
            break

        print(f"\n[{question['difficulty']} · {question['topic']} · "
              f"{question['marks']} mark(s)]")
        print(f"\n{question['text']}\n")

        answer = input("Your answer (or skip/reveal/quit)> ").strip()

        if answer.lower() == "quit":
            break
        if answer.lower() == "reveal":
            print("\n--- OFFICIAL MARK SCHEME ---")
            print(session.reveal_answer(question["id"]))
            print("--- (not graded) ---")
            continue
        if answer.lower() == "skip" or not answer:
            print("Skipped.")
            continue

        try:
            evaluation = session.grade_answer(question["id"], answer)
            print_evaluation(evaluation)
        except Exception as exc:                # API/network failures
            print(f"\n  [grading failed: {exc}]")
            print("  (Your answer wasn't scored. Try again or check your API key.)")

    print("\n" + "=" * 60)
    print(session.progress.summary())
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Tutor CLI")
    parser.add_argument("--mode", choices=[m.value for m in Mode],
                        default="sequential")
    parser.add_argument("--topic", help="for --mode topic")
    parser.add_argument("--difficulty", help="for --mode difficulty "
                        "(easy/medium/hard/very_hard)")
    parser.add_argument("--stub", action="store_true",
                        help="use offline stub grader (no API key needed)")
    args = parser.parse_args()

    try:
        run(Mode(args.mode), args.topic, args.difficulty, args.stub)
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
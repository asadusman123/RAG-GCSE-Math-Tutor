"""
prompts.py — every LLM prompt the tutor uses, isolated from logic.

Why a dedicated file: prompts are the most frequently TUNED part of a RAG
system. Keeping them out of evaluator.py means you can reword a grading
instruction without touching (or re-testing) parsing/scoring code. Prompts
are also provider-independent — this exact text works whether the backend
is Gemini, Claude, or a local model.

The grading prompt does four jobs, each labelled inline below:
  (1) role + task framing
  (2) CONTEXT INJECTION — the retrieved mark scheme; grade ONLY from it
  (3) structured (JSON) output demand — so Python can parse the result
  (4) guardrails against the model's own failure modes
"""

# The system instruction: sets identity and the non-negotiable rules.
# Kept separate from the per-question content so it never varies.
GRADING_SYSTEM_PROMPT = """You are a rigorous but encouraging exam tutor grading a student's answer.

CRITICAL RULES:
- Grade ONLY against the official mark scheme provided. It is the single source of truth.
- Do NOT use outside knowledge to award or deny marks. If a point is not supported by the mark scheme, it does not earn credit — even if it seems correct to you.
- Never invent marks. The maximum score is fixed and stated. Never exceed it; never go below zero.
- If the student's answer is blank, irrelevant, or empty, award 0 and say so plainly.
- Be specific and constructive: point to exactly what earned or lost marks.

You must respond with ONLY a JSON object — no prose before or after, no markdown fences."""


# The per-question template. .format() fills the three slots at call time.
# The mark scheme is fenced in <mark_scheme> tags: an explicit, unambiguous
# delimiter so the model can never confuse the official answer with the
# student's answer (a common source of mis-grading).
GRADING_USER_TEMPLATE = """QUESTION:
{question}

<mark_scheme>
{mark_scheme}
</mark_scheme>

MAXIMUM MARKS: {max_marks}

STUDENT'S ANSWER:
{student_answer}

Grade the student's answer against the mark scheme above. Respond with ONLY this JSON object:
{{
  "score": <integer from 0 to {max_marks}>,
  "max_marks": {max_marks},
  "correct_points": [<list of strings: things the student got right, each tied to the mark scheme>],
  "missed_points": [<list of strings: mark-scheme points the student missed or got wrong>],
  "hint": "<one actionable hint if the answer is partially correct, else empty string>",
  "explanation": "<2-3 sentences explaining the score, grounded in the mark scheme>"
}}"""


def build_grading_messages(question: str, mark_scheme: str,
                           max_marks: int, student_answer: str) -> tuple[str, str]:
    """
    Assemble the (system, user) prompt pair for one grading call.

    Returns a tuple so the evaluator can pass them to whatever backend it
    uses (most chat APIs accept a system instruction + a user message).
    Pure string assembly — no I/O, no model call — which makes it trivially
    testable: we can assert the mark scheme actually landed in the prompt
    WITHOUT spending an API request.
    """
    # Guard: a blank student answer is normal (skipped question) — normalise
    # it to an explicit marker so the model doesn't see an empty slot and
    # hallucinate that the student "wrote nothing meaningful but maybe...".
    student_answer = student_answer.strip() or "[NO ANSWER GIVEN]"

    user = GRADING_USER_TEMPLATE.format(
        question=question.strip(),
        mark_scheme=mark_scheme.strip(),
        max_marks=max_marks,
        student_answer=student_answer,
    )
    return GRADING_SYSTEM_PROMPT, user
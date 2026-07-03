"""
parser.py — Step 2 of ingestion: raw pages -> cleaned, structured chunks.

This is a STRUCTURE-AWARE chunker. Instead of splitting every N
characters, it parses the document's own grammar, discovered by
inspecting the PDF:

    "Easy Questions" / "Medium Questions" / ...   -> difficulty section
    line starting with a number (between blocks)  -> new question
    the literal line "Answer"                     -> mark scheme begins
    "(N marks)"                                   -> block ends

Output: Chunk objects. One chunk per QUESTION, one per ANSWER,
linked by pair_id so grading can fetch the exact mark scheme by ID
(never by fuzzy vector search).
"""

import re
from dataclasses import dataclass, field, asdict

from .loader import Page

# ---------------------------------------------------------------- markers
# The document's grammar, as compiled regexes.
# ^...$ anchors mean "the whole line", which is what makes these safe.
SECTION_RE = re.compile(r"^(Easy|Medium|Hard|Very Hard) Questions$")
MARKS_RE = re.compile(r"^\((\d+)\s+marks?\)$")   # e.g. "(2 marks)"
QSTART_RE = re.compile(r"^(\d+)\b")              # e.g. "1 Calculate..." / "2 (a)"
ANSWER_LINE = "Answer"

# Any line containing these is publisher boilerplate (footer / copyright),
# not content. It would pollute embeddings with identical noise on every
# chunk, so we strip it during cleaning.
FOOTER_MARKERS = ("Save My Exams", "savemyexams.com")


@dataclass
class Chunk:
    """One retrievable unit + everything we know about it (metadata)."""
    id: str                 # e.g. "hard_q2"  /  "hard_q2_ans"
    type: str               # "question" | "answer"
    difficulty: str         # "easy" | "medium" | "hard" | "very_hard"
    question_number: int    # number within its section (1, 2, 3...)
    marks: int              # total marks for the whole question
    topic: str              # coarse label used for topic-mode retrieval
    pair_id: str            # id of the partner chunk (question <-> answer)
    pages: list[int]        # source pages — provenance for debugging/citations
    text: str = ""


# ---------------------------------------------------------------- cleaning
def clean_pages(pages: list[Page]) -> list[tuple[str, int]]:
    """
    Flatten pages into (line, page_number) tuples, dropping boilerplate.

    Why tuples and not plain lines? Provenance: every line remembers the
    page it came from, so every chunk can report its source pages.
    """
    lines: list[tuple[str, int]] = []
    for page in pages:
        for raw in page.text.splitlines():
            line = raw.strip()
            if not line:
                continue                       # blank line
            if any(m in line for m in FOOTER_MARKERS):
                continue                       # footer/copyright noise
            lines.append((line, page.number))
    return lines


# ---------------------------------------------------------------- topics
def infer_topic(question_text: str) -> str:
    """
    Coarse keyword-based topic labelling (checked in priority order).

    Honest limitation: keyword matching is crude — the grown-up version
    embeds each question and clusters, or asks an LLM to label. For a
    single worksheet with three known themes, keywords are proportionate.
    """
    t = question_text.lower()
    if "parallel" in t:            # also catches "parallelogram" — acceptable:
        return "parallel_lines"    # those problems use parallel-line reasoning
    if any(k in t for k in ("polygon", "pentagon", "interior angle", "exterior angle")):
        return "polygons"
    if "triangle" in t:
        return "triangles"
    return "angles_general"


# ---------------------------------------------------------------- parsing
def parse(pages: list[Page]) -> list[Chunk]:
    """
    State machine: (line, page) stream -> list[Chunk].

    Modes:
      idle     — before the first section header (cover page): ignore all
      expect   — between blocks: a digit-leading line starts a NEW question;
                 anything else continues the CURRENT question (e.g. part (b))
      question — collecting question text
      answer   — collecting mark-scheme text

    Complexity: O(number of lines) — single pass, no backtracking.
    """
    chunks: list[Chunk] = []
    mode = "idle"
    difficulty: str | None = None

    # Accumulator for the question currently being built.
    number = 0
    q_lines: list[str] = []
    a_lines: list[str] = []
    marks = 0
    page_set: set[int] = set()

    def flush() -> None:
        """Finish the current question: emit its question+answer chunks."""
        nonlocal number, q_lines, a_lines, marks, page_set
        if not q_lines:            # nothing accumulated (e.g. section start)
            return
        q_text = "\n".join(q_lines).strip()
        a_text = "\n".join(a_lines).strip()
        qid = f"{difficulty}_q{number}"
        topic = infer_topic(q_text)
        srcpages = sorted(page_set)
        chunks.append(Chunk(id=qid, type="question", difficulty=difficulty,
                            question_number=number, marks=marks, topic=topic,
                            pair_id=f"{qid}_ans", pages=srcpages, text=q_text))
        chunks.append(Chunk(id=f"{qid}_ans", type="answer", difficulty=difficulty,
                            question_number=number, marks=marks, topic=topic,
                            pair_id=qid, pages=srcpages, text=a_text))
        # reset accumulator
        number, marks = 0, 0
        q_lines, a_lines, page_set = [], [], set()

    for line, pageno in clean_pages(pages):
        section = SECTION_RE.match(line)
        if section:
            flush()                                    # close previous block
            difficulty = section.group(1).lower().replace(" ", "_")
            mode = "expect"
            continue

        if mode == "idle":
            continue                                   # cover-page noise

        if line == ANSWER_LINE:
            mode = "answer"
            continue

        m = MARKS_RE.match(line)
        if m and mode == "answer":
            marks += int(m.group(1))                   # multi-part: totals add up
            mode = "expect"
            continue

        if mode == "expect":
            qm = QSTART_RE.match(line)
            if qm:                                     # a NEW question begins
                flush()
                number = int(qm.group(1))
                rest = line[qm.end():].strip()         # drop the leading number
                q_lines = [rest] if rest else []
                mode = "question"
            else:                                      # same question continues
                q_lines.append(line)                   # (e.g. "(b) ..." part)
                mode = "question"
            page_set.add(pageno)
            continue

        if mode == "question":
            q_lines.append(line)
        elif mode == "answer":
            a_lines.append(line)
        page_set.add(pageno)

    flush()                                            # last question in the file
    return chunks


# ---------------------------------------------------------------- validation
# Ground truth transcribed from the PDF's own cover page — an ingestion
# pipeline should always be checked against something the document asserts
# about itself.
EXPECTED_COUNTS = {"easy": 3, "medium": 3, "hard": 4, "very_hard": 3}
EXPECTED_MARKS = {"easy": 7, "medium": 7, "hard": 18, "very_hard": 21}


def validate(chunks: list[Chunk]) -> list[str]:
    """Return a list of problems (empty list == pipeline is healthy)."""
    problems: list[str] = []
    questions = [c for c in chunks if c.type == "question"]

    for diff, want in EXPECTED_COUNTS.items():
        got = sum(1 for q in questions if q.difficulty == diff)
        if got != want:
            problems.append(f"{diff}: expected {want} questions, parsed {got}")

    for diff, want in EXPECTED_MARKS.items():
        got = sum(q.marks for q in questions if q.difficulty == diff)
        if got != want:
            problems.append(f"{diff}: expected {want} total marks, parsed {got}")

    by_id = {c.id: c for c in chunks}
    for q in questions:
        ans = by_id.get(q.pair_id)
        if ans is None or not ans.text.strip():
            problems.append(f"{q.id}: missing or empty answer chunk")
        if not q.text.strip():
            problems.append(f"{q.id}: empty question text")
    return problems

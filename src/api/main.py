"""
Run:  uvicorn src.api.main:app --reload
      then open  http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.retrieval.retriever import Retriever
from src.tutor.evaluator import GeminiEvaluator, StubEvaluator
from src.tutor.session import Session, Mode

app = FastAPI(title="RAG Tutor")

# --- one shared Session for the server's lifetime -----------------------
# Loading the index + embedder is expensive (~seconds); we do it ONCE at
# startup, not per request. USE_STUB lets you demo with no API cost/key.
_USE_STUB = os.environ.get("TUTOR_STUB") == "1"
_session: Session | None = None
# Backend is env-driven, like TUTOR_STUB — flip without editing code.
# TUTOR_BACKEND=chroma  -> ChromaDB at data/chroma
# (unset)               -> FAISS at data/index  (default)
_BACKEND = os.environ.get("TUTOR_BACKEND", "faiss")
_INDEX_DIR = "data/chroma" if _BACKEND == "chroma" else "data/index"

def get_session() -> Session:
    global _session
    if _session is None:
        retriever = Retriever(index_dir=_INDEX_DIR, backend=_BACKEND)
        evaluator = StubEvaluator() if _USE_STUB else GeminiEvaluator()
        _session = Session(retriever, evaluator)
    return _session

# --- request/response schemas (FastAPI validates these automatically) ---
class GradeRequest(BaseModel):
    question_id: str
    answer: str


# --- API endpoints ------------------------------------------------------
@app.get("/api/meta")
def meta():
    """Topics + difficulties, to populate the UI's mode menus."""
    s = get_session()
    return {"topics": s.available_topics(),
            "difficulties": s.available_difficulties()}


@app.get("/api/question")
def question(mode: str = "random", topic: str | None = None,
             difficulty: str | None = None):
    """Next question for the chosen mode. Returns the QUESTION only —
    never its mark scheme (leak protection preserved across HTTP)."""
    s = get_session()
    try:
        q = s.next_question(Mode(mode), topic=topic, difficulty=difficulty)
    except ValueError:
        raise HTTPException(400, f"unknown mode: {mode}")
    if q is None:
        return {"done": True}
    # Attach a cached diagram SVG if one exists. Read the cache file directly
    # (no renderer object, no API key needed) — diagrams are pre-generated at
    # ingestion time, so serving is pure disk read.
    from src.ingestion.diagram_renderer import CACHE_DIR
    svg_path = CACHE_DIR / f"{q['id']}.svg"
    svg = svg_path.read_text(encoding="utf-8") if svg_path.exists() else None
    # Whitelist the fields we send — never leak pair_id logic or answer text.
    return {"done": False, "id": q["id"], "text": q["text"],
            "difficulty": q["difficulty"], "topic": q["topic"],
            "marks": q["marks"], "diagram_svg": svg}


@app.post("/api/grade")
def grade(req: GradeRequest):
    """Grade an answer via the (server-side) evaluator. This is the call
    that talks to Gemini; the key stays here, never in the browser."""
    s = get_session()
    try:
        ev = s.grade_answer(req.question_id, req.answer)
    except KeyError:
        raise HTTPException(404, f"unknown question: {req.question_id}")
    except Exception as exc:                      # API/network failure
        raise HTTPException(502, f"grading failed: {exc}")
    return {"score": ev.score, "max_marks": ev.max_marks,
            "correct_points": ev.correct_points,
            "missed_points": ev.missed_points,
            "hint": ev.hint, "explanation": ev.explanation,
            "progress": s.progress.summary()}


@app.get("/api/search")
def search(q: str):
    """Free-text semantic search: embed the query, return the best-matching
    QUESTION (leak protection: type==question filter). This is the endpoint
    that actually exercises the embedding vectors at query time. Applies the
    relevance threshold, so an off-topic query returns no match rather than
    a forced bad one."""
    s = get_session()
    hits = s.retriever.search(q, k=1, filters={"type": "question"})
    if not hits:
        return {"found": False}
    rec = hits[0].record
    from src.ingestion.diagram_renderer import CACHE_DIR
    svg_path = CACHE_DIR / f"{rec['id']}.svg"
    svg = svg_path.read_text(encoding="utf-8") if svg_path.exists() else None
    return {"found": True, "score": round(hits[0].score, 3),
            "id": rec["id"], "text": rec["text"], "difficulty": rec["difficulty"],
            "topic": rec["topic"], "marks": rec["marks"], "diagram_svg": svg}


@app.get("/api/reveal")
def reveal(question_id: str):
    """Official mark scheme for 'show answer'."""
    s = get_session()
    try:
        return {"mark_scheme": s.reveal_answer(question_id)}
    except KeyError:
        raise HTTPException(404, f"unknown question: {question_id}")


# --- serve the frontend -------------------------------------------------
# The UI is a static folder; the API lives under /api so the two never clash.
_FRONTEND = Path(__file__).parent.parent.parent / "frontend"
if _FRONTEND.exists():
    @app.get("/")
    def index():
        return FileResponse(_FRONTEND / "index.html")
    app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")
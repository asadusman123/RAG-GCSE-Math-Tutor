# RAG GCSE Math Tutor

A Retrieval-Augmented Generation (RAG) tutoring application that quizzes you from a real exam paper and grades your answers against the **official mark scheme** — using an LLM that never invents the answer, because the mark scheme is retrieved and injected into the prompt.

Built from scratch (custom parser, embedder, vector store, retriever, evaluator) to understand every layer, then extended with the techniques used in production RAG systems: ChromaDB, cross-encoder re-ranking, and hybrid dense+sparse retrieval.

> **Why "grounded"?** LLMs hallucinate because they predict plausible text from frozen training weights. This system moves the model's job from *recall* (unreliable) to *reading comprehension over retrieved text* (reliable): the correct mark scheme is fetched by exact ID and placed in the prompt, so the model only *compares and explains* — it never fabricates a grade or a mark scheme.

---

## What it does

- Presents a question from the ingested exam paper (question only — the mark scheme is never leaked to the student).
- Accepts the student's typed answer.
- Grades it against the **retrieved official mark scheme** via the Gemini API, returning a structured evaluation: score, what was correct, what was missing, a hint, and an explanation.
- Supports four selection modes — random, sequential, by topic, by difficulty — plus **free-text semantic search** ("give me a question about angles on a straight line").
- Renders **auto-generated SVG diagrams** for geometry questions (LLM emits SVG code, *not* AI-generated pixels — see below).
- Tracks progress across a session.

---

## Architecture

RAG systems have two pipelines that run on different cadences:

```
INGESTION (offline, run once)
  PDF ──► load ──► parse (structure-aware chunking + metadata)
              ──► embed (MiniLM) ──► store (FAISS or ChromaDB)

QUERY (online, per interaction)
  pick question (metadata filter)  ──► student answers
       ──► fetch mark scheme BY ID (exact, not search)
       ──► grade via LLM (Gemini)  ──► structured feedback
```

**Design principle:** each layer has one responsibility and a thin interface, so components swap without touching the rest. This is proven in the repo — the store was migrated FAISS → ChromaDB by changing one file, and the retrieval layer supports both backends behind the same interface.

### Folder structure

```
src/
  ingestion/     loader.py, vision_loader.py, parser.py, pipeline.py,
                 diagram_renderer.py   — PDF → structured, embedded chunks
  embeddings/    embedder.py           — text → unit-length vectors (MiniLM)
  retrieval/     vector_store.py (FAISS), chroma_store.py (ChromaDB),
                 retriever.py, reranker.py, hybrid.py  — search & ranking
  tutor/         session.py, prompts.py, evaluator.py  — tutoring logic + grading
  evaluation/    metrics.py, faithfulness.py, run_eval.py  — RAG evaluation
  api/           main.py               — FastAPI backend
  langchain_demo.py                    — the same pipeline in LangChain idioms
frontend/        index.html            — dark-mode web UI
tests/           82 tests across 12 files
cli.py                                 — terminal tutor
```

---

## How RAG works here (the interesting decisions)

**Structure-aware chunking, not fixed-size.** The exam paper has a Q&A grammar (question → "Answer" → mark scheme → "(N marks)"). Naive fixed-size chunking would split mark schemes mid-solution and glue questions to their answers — leaking the solution. Instead, `parser.py` is a **state machine** that reads the document's own grammar, so each chunk is one complete question *or* one complete answer, tagged with metadata (`difficulty`, `topic`, `marks`, `type`, `pair_id`) and validated against the totals the PDF states on its cover page.

**A vision loader for broken fonts.** The PDF's math used a broken Type 3 font that standard text extraction silently dropped — every formula and final answer lost. `vision_loader.py` rasterizes each page and has a vision model transcribe the *rendered* pixels (which broken font metadata can't corrupt), cached once. *Ingestion quality is the ceiling on the whole system.*

**Fuzzy retrieval where safe, exact where it matters.** Vector search (semantic) selects questions and powers free-text search. But grading fetches the mark scheme by exact `pair_id`, never by search — grading against a similar-but-wrong scheme would be catastrophic.

**Embeddings + FAISS/ChromaDB.** `all-MiniLM-L6-v2` turns text into 384-dim unit vectors so cosine similarity equals a dot product. FAISS was the initial store (a library — no server, nothing hidden — ideal for understanding internals); the project also migrates to **ChromaDB**, whose native metadata pre-filtering replaces the hand-rolled post-filter workaround.

**Two-stage re-ranking.** Optional cross-encoder stage: retrieve a wide pool cheaply (bi-encoder + vector search, optimizing recall), then re-score each (query, chunk) pair with a cross-encoder that sees both together (optimizing precision).

**Hybrid search.** Dense (semantic) + sparse (BM25) retrieval fused with **Reciprocal Rank Fusion** — because each has the other's blind spot (dense finds paraphrases; BM25 nails exact terms). RRF fuses by rank position, sidestepping the incompatible score scales.

**LLM-to-SVG diagrams, not image generation.** Geometry questions need figures. An AI *image* generator would hallucinate wrong geometry — contradicting the grounded premise. Instead the LLM emits **SVG code** (geometry as math), which is validated to parse before caching, with the authoritative text description always shown alongside.

**Provider abstraction.** Embedders, evaluators, re-rankers, and diagram renderers each define one interface with a real backend and a deterministic stub, so the full pipeline is testable offline with no API key.

---

## Anti-hallucination: how grading stays grounded

Three layers, because instructions alone aren't enough — models drift:

1. **Exact-ID retrieval** — the correct mark scheme is fetched by `pair_id`, so the model always sees the right answer.
2. **Prompt grounding** — the mark scheme is fenced in the prompt with an instruction to grade *only* from it.
3. **Code-level enforcement** — the score is clamped to the valid range in Python, regardless of what the model returns.

---

## Setup

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Unix:     source .venv/bin/activate

# 2. install dependencies
python -m pip install -r requirements.txt

# 3. add your Gemini API key (free tier: https://aistudio.google.com)
#    create a .env file in the project root:
echo "GEMINI_API_KEY=your-key-here" > .env

# 4. build the search index from the cached page transcriptions
python -m src.ingestion.pipeline            # FAISS
python -m src.ingestion.pipeline --chroma   # or ChromaDB

# 5. run
python -m pytest tests -q                   # run the test suite
python cli.py                               # terminal tutor
python -m uvicorn src.api.main:app --reload # web app at http://127.0.0.1:8000
```

Environment flags: `TUTOR_STUB=1` uses an offline stub grader (no API cost); `TUTOR_BACKEND=chroma` runs on ChromaDB instead of FAISS.

`.env.example` documents the required environment variables. Your real `.env` is gitignored and never committed.

---

## Usage

```bash
python cli.py --mode difficulty --difficulty hard   # hard questions only
python cli.py --mode topic --topic polygons         # polygon questions
python cli.py --stub                                 # offline demo, no API key
```

In the web UI: pick a mode or type a free-text query ("Find by meaning"), answer the question, and watch the animated grade reveal. `reveal` shows the mark scheme; `skip` moves on.

---

## Evaluation

RAG must be evaluated in two stages, because retrieval and generation fail independently (a right-looking grade can hide broken retrieval):

- **Retrieval** (deterministic): retrieval accuracy on a labeled query set, and **pairing exactness** — every question maps to its correct mark scheme (100% by construction). See `src/evaluation/metrics.py`.
- **Generation** (LLM-as-judge): a **faithfulness** check that verifies the grader's feedback is grounded in the mark scheme, not invented. See `src/evaluation/faithfulness.py`.

```bash
python -m src.evaluation.run_eval           # deterministic report (free)
python -m src.evaluation.run_eval --judge   # + LLM faithfulness (uses Gemini)
```

---

## Testing

82 tests across 12 files, runnable offline (deterministic stubs stand in for the model download and API calls). Contract tests enforce the key invariants — most importantly that **an answer chunk can never be returned as a question**, verified at the retriever, session, and HTTP layers.

```bash
python -m pytest tests -q
```

---

## Tech stack

Python · FastAPI · FAISS · ChromaDB · sentence-transformers (MiniLM, cross-encoder) · rank-bm25 · Gemini API · LangChain (parallel implementation)

---

## Known limitations

- **Single document.** Built and tuned for one exam paper; the relevance threshold is calibrated to this corpus and embedding model.
- **Diagram accuracy.** LLM-generated SVGs are a best-effort aid — topologically reasonable but not always precise — which is why the authoritative text description is always shown alongside.
- **Scale.** Uses exact search and in-memory indexes, correct at this corpus size; a large corpus would need approximate indexing (IVF/HNSW), a managed vector DB, and batched ingestion.

---

*Built as a deep-dive learning project: hand-built core first to understand every layer, then extended with production RAG techniques.*
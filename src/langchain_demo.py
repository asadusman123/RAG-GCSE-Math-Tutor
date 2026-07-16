"""
langchain_demo.py — YOUR RAG tutor, re-expressed in LangChain idioms.

Purpose: NOT to replace your hand-built pipeline, but to make LangChain
code legible on day one by mapping every piece to what you already built.

  YOUR CODE                     LANGCHAIN
  Chunk                    ->   Document(page_content, metadata)
  embedder.py              ->   Embeddings
  chroma_store.py          ->   Chroma vector store
  retriever.py             ->   vectorstore.as_retriever()
  prompts.py               ->   PromptTemplate
  evaluator.py (Gemini)    ->   ChatModel
  the query flow           ->   an LCEL chain:  prompt | llm | parser

Run:  python -m src.langchain_demo
"""

from __future__ import annotations

from src.ingestion.vision_loader import load_transcribed_pages
from src.ingestion.parser import parse


# ---------------------------------------------------------------- 1. Documents
def chunks_to_documents():
    """
    Convert our Chunk objects into LangChain Documents.

    A Document is just: text + metadata — i.e. EXACTLY our Chunk. page_content
    is our .text; metadata is our difficulty/topic/type/pair_id. When you see
    `Document` in LangChain code, read it as "a Chunk".
    """
    from langchain_core.documents import Document
    chunks = parse(load_transcribed_pages())
    docs = []
    for c in chunks:
        docs.append(Document(
            page_content=c.text,
            metadata={"id": c.id, "type": c.type, "difficulty": c.difficulty,
                      "topic": c.topic, "marks": c.marks, "pair_id": c.pair_id},
        ))
    return docs


# ------------------------------------------------------------- 2. Vector store
def build_langchain_vectorstore(docs):
    """
    Build a Chroma vector store the LangChain way.

    Compare to your pipeline.py: there you embedded chunks and called
    store.add() yourself. Here `Chroma.from_documents` does embed + store +
    persist in ONE call — the convenience LangChain buys you. It uses the
    SAME MiniLM model under the hood (HuggingFaceEmbeddings), so the vectors
    are identical to what your embedder.py produces.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=docs, embedding=embeddings)


# ----------------------------------------------------------------- 3. Retriever
def make_retriever(vectorstore):
    """
    `.as_retriever()` turns a vector store into a retriever — the same role
    as your retriever.py. The search_kwargs are your k and metadata filter:
    `filter={"type": "question"}` IS your leak protection, one line.
    """
    return vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"type": "question"}})


# ------------------------------------------------------------ 4. Prompt + chain
def build_grading_chain():
    """
    An LCEL chain: prompt | llm | parser.

    This mirrors your prompts.py + evaluator.py, composed with the `|`
    operator. Data flows left to right: the input dict fills the prompt, the
    filled prompt goes to the LLM, the LLM's reply is parsed to a string.

    NOTE: this needs a configured Gemini chat model. The structure is what
    matters — see how the pieces you built by hand become a one-line pipe.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    # from langchain_google_genai import ChatGoogleGenerativeAI  # your Gemini

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You grade a student's answer ONLY against the provided "
                   "mark scheme. Never invent marks."),
        ("human", "QUESTION:\n{question}\n\nMARK SCHEME:\n{mark_scheme}\n\n"
                  "STUDENT ANSWER:\n{answer}\n\nGrade it out of {max_marks}."),
    ])
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # return prompt | llm | StrOutputParser()
    #        └─────────── the LCEL pipe: your whole query flow, one line ─────┘
    return prompt   # returning the prompt alone so the demo runs without a key


# --------------------------------------------------------------------- demo
def main():
    docs = chunks_to_documents()
    print(f"1. {len(docs)} Documents (== our Chunks)")
    print(f"   example metadata: {docs[0].metadata}")

    vectorstore = build_langchain_vectorstore(docs)
    print("2. Chroma vector store built (embed+store+persist in one call)")

    retriever = make_retriever(vectorstore)
    hits = retriever.invoke("interior angle of a regular polygon")
    print(f"3. retriever.invoke() -> {[d.metadata['id'] for d in hits]}")
    print("   (filter={'type':'question'} == our leak protection)")

    prompt = build_grading_chain()
    filled = prompt.invoke({"question": "Find the interior angle...",
                            "mark_scheme": "165 degrees", "answer": "165",
                            "max_marks": 2})
    print("4. prompt.invoke() filled the template:")
    print("   ", str(filled.messages[1].content)[:70], "...")


if __name__ == "__main__":
    main()
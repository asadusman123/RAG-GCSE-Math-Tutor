"""
LangChain demo tests — the parts that run offline (no model download).
Verifies our Chunks map cleanly to LangChain Documents and that the
LCEL PromptTemplate fills correctly.
"""

from src.langchain_demo import chunks_to_documents, build_grading_chain


def test_chunks_become_documents_losslessly():
    docs = chunks_to_documents()
    assert len(docs) == 26
    d = docs[0]
    # page_content == our text; metadata carries all our chunk fields
    assert d.page_content.strip()
    for key in ("id", "type", "difficulty", "topic", "marks", "pair_id"):
        assert key in d.metadata


def test_documents_preserve_leak_relevant_metadata():
    # The `type` field must survive — LangChain's filter relies on it.
    docs = chunks_to_documents()
    types = {d.metadata["type"] for d in docs}
    assert types == {"question", "answer"}


def test_prompt_template_fills():
    prompt = build_grading_chain()
    filled = prompt.invoke({"question": "Q", "mark_scheme": "MS",
                            "answer": "A", "max_marks": 2})
    # both the system rule and the human message with our data are present
    contents = " ".join(m.content for m in filled.messages)
    assert "mark scheme" in contents.lower()
    assert "MS" in contents and "A" in contents
from rag_platform.core import DocumentStore


def test_grounded_answer_and_replacement(tmp_path):
    store = DocumentStore(tmp_path / "test.db")
    try:
        assert store.ingest("guide", "The service uses Python. It stores documents locally.") == 1
        result = store.answer("Which language does the service use?")
        assert result["citations"][0]["source"] == "guide"
        assert result["citations"][0]["excerpt"] in result["answer"]
        store.ingest("guide", "The service now uses Rust.")
        assert store.answer("Python")["citations"] == []
        assert store.answer("Rust")["citations"][0]["source"] == "guide"
    finally:
        store.close()


def test_empty_question_abstains(tmp_path):
    store = DocumentStore(tmp_path / "test.db")
    try:
        assert store.answer("!!!")["citations"] == []
    finally:
        store.close()

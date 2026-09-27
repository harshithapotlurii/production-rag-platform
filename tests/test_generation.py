import json
from unittest.mock import patch

from rag_platform.core import DocumentStore
from rag_platform.generation import ModelConfig, generate_answer


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.data

    def __init__(self, content):
        self.data = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()


def test_valid_citations_and_rejection(tmp_path):
    store = DocumentStore(tmp_path / "test.db")
    try:
        store.ingest("guide", "Python supports this sample service.")
        config = ModelConfig("https://example.test/v1", "example-model")
        with patch("urllib.request.urlopen", return_value=Response({"answer": "Python is used.", "citation_ids": [1]})):
            result = generate_answer(store, "Python service", config)
        assert result["mode"] == "model"
        assert result["citations"][0]["source"] == "guide"
        with patch("urllib.request.urlopen", return_value=Response({"answer": "Unsupported", "citation_ids": [9]})):
            result = generate_answer(store, "Python service", config)
        assert result["mode"] == "extractive_fallback"
    finally:
        store.close()

"""Unit test for the token handler."""

from typing import Any
from unittest import TestCase, mock

from ols.utils.token_handler import TokenHandler


class MockRetrievedNode:
    """Class to create mock data for retrieved node."""

    def __init__(self, node_detail: dict):
        """Initialize the class instance."""
        self._text = node_detail["text"]
        self._score = node_detail["score"]
        self._metadata = node_detail["metadata"]

    def get_text(self) -> str:
        """Mock get_text."""
        return self._text

    @property
    def score(self) -> float:
        """Mock score."""
        return self._score

    @property
    def metadata(self) -> dict[str, Any]:
        """Mock metadata."""
        return self._metadata


class TestTokenHandler(TestCase):
    """Test cases for TokenHandler."""

    def setUp(self):
        """Set up mock data."""
        node_data = [
            {
                "text": "a text text text text",
                "score": 0.6,
                "metadata": {"docs_url": "data/doc1.pdf"},
            },
            {
                "text": "b text text text text",
                "score": 0.55,
                "metadata": {"docs_url": "data/doc2.pdf"},
            },
            {
                "text": "c text text text text",
                "score": 0.55,
                "metadata": {"docs_url": "data/doc3.pdf"},
            },
            {
                "text": "d text text text text",
                "score": 0.4,
                "metadata": {"docs_url": "data/doc4.pdf"},
            },
        ]
        self._mock_retrieved_obj = [MockRetrievedNode(data) for data in node_data]
        self._token_handler_obj = TokenHandler()

    def test_token_handler(self):
        """Test token handler for context."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        context = self._token_handler_obj.truncate_rag_context(retrieved_nodes)

        assert len(context) == len(retrieved_nodes)
        for idx, data in enumerate(context):
            assert data["text"][0] == self._mock_retrieved_obj[idx].get_text()[0]
            assert (
                data["docs_url"] == self._mock_retrieved_obj[idx].metadata["docs_url"]
            )

    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        context = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context) == 2
        assert (
            context[1]["text"].split()
            == self._mock_retrieved_obj[1].get_text().split()[:2]
        )

    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        context = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context) == 1

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        context = self._token_handler_obj.truncate_rag_context([], 5)

        assert len(context) == 0

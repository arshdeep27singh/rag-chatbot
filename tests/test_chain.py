import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chain import ask_question


class TestRAGChain(unittest.TestCase):
    def test_ask_question_returns_answer_and_sources(self):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "The document discusses AI safety.",
            "source_documents": [
                MagicMock(
                    page_content="AI safety is an important research area that focuses on..." + "x" * 200,
                    metadata={"page": 1, "source": "test.pdf"},
                )
            ],
        }

        result = ask_question(mock_chain, "What does the document discuss?")

        self.assertEqual(result["answer"], "The document discusses AI safety.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["page"], 1)
        self.assertEqual(result["sources"][0]["source"], "test.pdf")

    def test_ask_question_handles_no_sources(self):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "I don't have enough information.",
            "source_documents": [],
        }

        result = ask_question(mock_chain, "What is quantum physics?")

        self.assertEqual(result["answer"], "I don't have enough information.")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()

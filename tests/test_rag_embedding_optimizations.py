"""Regression tests for retrieval performance caches."""

from __future__ import annotations

import unittest

from src.rag.embeddings import EmbeddingIndex, EmbeddingRecord, cosine_similarity, sparse_dot


class EmbeddingIndexOptimizationTests(unittest.TestCase):
    def build_index(self) -> EmbeddingIndex:
        index = object.__new__(EmbeddingIndex)
        index.records = {
            "a": EmbeddingRecord("a", "bge-m3", "test", "a", [1.0, 0.0], {"1": 2.0}),
            "b": EmbeddingRecord("b", "bge-m3", "test", "b", [1.0, 1.0], {"1": 1.0, "2": 3.0}),
        }
        index._dense_matrix = None
        index._dense_row_by_id = {}
        index._sparse_postings = None
        return index

    def test_vectorized_dense_scores_match_scalar_cosine(self) -> None:
        index = self.build_index()
        query = [0.0, 1.0]
        scores = index.dense_scores(["a", "b"], query)

        self.assertAlmostEqual(scores["a"], cosine_similarity(query, [1.0, 0.0]), places=6)
        self.assertAlmostEqual(scores["b"], cosine_similarity(query, [1.0, 1.0]), places=6)

    def test_sparse_posting_scores_match_scalar_dot(self) -> None:
        index = self.build_index()
        query = {"1": 0.5, "2": 2.0}
        scores = index.sparse_scores(["a", "b"], query)

        self.assertAlmostEqual(scores["a"], sparse_dot(query, {"1": 2.0}), places=6)
        self.assertAlmostEqual(scores["b"], sparse_dot(query, {"1": 1.0, "2": 3.0}), places=6)


if __name__ == "__main__":
    unittest.main()

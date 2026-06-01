"""Tests for strong-signal PDF page filtering."""

from __future__ import annotations

import unittest

from src.rag.pdf_page_filter import filter_pdf_chunks_by_page_signals
from src.rag.retrieval_intent import build_retrieval_intent


def chunk(source_id: str, chunk_id: str, start: int | str, end: int | str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_type": "pdf",
        "page_start": start,
        "page_end": end,
        "content_type": "text",
    }


class PdfPageFilterTests(unittest.TestCase):
    def test_keeps_chunks_overlapping_hit_page_neighbors(self) -> None:
        chunks = [
            chunk("pdf_a", "a1", 1, 1),
            chunk("pdf_a", "a2", 2, 3),
            chunk("pdf_a", "a4", 4, 4),
            chunk("pdf_b", "b1", 1, 1),
        ]
        page_signals = [
            {"source_id": "pdf_a", "page": 1, "entities": [], "predicate_hints": []},
            {"source_id": "pdf_a", "page": 2, "entities": ["Mars"], "predicate_hints": ["moon_count"]},
            {"source_id": "pdf_a", "page": 3, "entities": [], "predicate_hints": []},
            {"source_id": "pdf_a", "page": 4, "entities": [], "predicate_hints": []},
            {"source_id": "pdf_b", "page": 1, "entities": [], "predicate_hints": []},
        ]

        filtered = filter_pdf_chunks_by_page_signals(
            chunks,
            page_signals,
            build_retrieval_intent("How many moons does Mars have?"),
            class_members={},
        )

        self.assertEqual([item["chunk_id"] for item in filtered], ["a1", "a2"])

    def test_drops_no_hit_pdf_when_another_pdf_has_hits(self) -> None:
        chunks = [
            chunk("pdf_a", "a1", 1, 1),
            chunk("pdf_b", "b1", 1, 1),
        ]
        page_signals = [
            {"source_id": "pdf_a", "page": 1, "entities": ["Mars"], "predicate_hints": ["moon_count"]},
            {"source_id": "pdf_b", "page": 1, "entities": [], "predicate_hints": []},
        ]

        filtered = filter_pdf_chunks_by_page_signals(
            chunks,
            page_signals,
            build_retrieval_intent("How many moons does Mars have?"),
            class_members={},
        )

        self.assertEqual([item["chunk_id"] for item in filtered], ["a1"])

    def test_falls_back_to_all_chunks_when_no_pdf_has_hits(self) -> None:
        chunks = [
            chunk("pdf_a", "a1", 1, 1),
            chunk("pdf_b", "b1", 1, 1),
        ]
        page_signals = [
            {"source_id": "pdf_a", "page": 1, "entities": ["Venus"], "predicate_hints": []},
            {"source_id": "pdf_b", "page": 1, "entities": [], "predicate_hints": []},
        ]

        filtered = filter_pdf_chunks_by_page_signals(
            chunks,
            page_signals,
            build_retrieval_intent("How many moons does Mars have?"),
            class_members={},
        )

        self.assertEqual(filtered, chunks)

    def test_entity_and_predicate_are_both_required_when_available(self) -> None:
        chunks = [
            chunk("pdf_a", "a1", 1, 1),
            chunk("pdf_b", "b1", 1, 1),
            chunk("pdf_c", "c1", 1, 1),
        ]
        page_signals = [
            {"source_id": "pdf_a", "page": 1, "entities": ["Mars"], "predicate_hints": ["moon_count"]},
            {"source_id": "pdf_b", "page": 1, "entities": [], "predicate_hints": ["moon_count"]},
            {"source_id": "pdf_c", "page": 1, "entities": ["Mars"], "predicate_hints": []},
        ]

        filtered = filter_pdf_chunks_by_page_signals(
            chunks,
            page_signals,
            build_retrieval_intent("How many moons does Mars have?"),
            class_members={},
        )

        self.assertEqual([item["chunk_id"] for item in filtered], ["a1"])


if __name__ == "__main__":
    unittest.main()

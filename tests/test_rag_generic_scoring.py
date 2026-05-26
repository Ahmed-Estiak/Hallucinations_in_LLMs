"""Regression tests for query-dependent RAG heuristic scoring."""

from __future__ import annotations

import unittest

from src.question_classifier import QuestionClassifier
from src.rag.retrieval_intent import build_retrieval_intent, score_target_class_evidence
from src.rag.retriever import RagRetriever, RetrievedChunk


EMPTY_TIME_CONSTRAINTS = {"phrases": [], "years": [], "months": []}


class GenericScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = object.__new__(RagRetriever)

    def score_chunk(self, question: str, text: str, predicate_hints: list[str]) -> tuple[float, list[str]]:
        intent = build_retrieval_intent(question)
        return self.retriever._score_chunk(
            {
                "title": "Evidence",
                "section": "Facts",
                "text": text,
                "predicate_hints": predicate_hints,
            },
            intent=intent,
            time_constraints=EMPTY_TIME_CONSTRAINTS,
        )

    def test_location_filter_bonus_requires_requested_value(self) -> None:
        q9 = "Which dwarf planet located in the Kuiper Belt was discovered first?"
        _, q9_reasons = self.score_chunk(
            q9,
            "Pluto is a dwarf planet in the Kuiper Belt and was discovered in 1930.",
            ["classification", "location", "discovered_on"],
        )
        self.assertIn("filter:location:kuiper_belt", q9_reasons)

        asteroid_question = "Which dwarf planet located in the Asteroid Belt was discovered first?"
        _, irrelevant_reasons = self.score_chunk(
            asteroid_question,
            "Pluto is a dwarf planet in the Kuiper Belt and was discovered in 1930.",
            ["classification", "location", "discovered_on"],
        )
        self.assertNotIn("filter:location:asteroid_belt", irrelevant_reasons)

        _, relevant_reasons = self.score_chunk(
            asteroid_question,
            "Ceres is a dwarf planet located in the Asteroid Belt and was discovered in 1801.",
            ["classification", "location", "discovered_on"],
        )
        self.assertIn("filter:location:asteroid_belt", relevant_reasons)

    def test_ordering_bonus_requires_requested_attribute(self) -> None:
        size_question = "Which dwarf planet located in the Kuiper Belt is ranked first by size?"
        _, discovery_reasons = self.score_chunk(
            size_question,
            "In order of discovery, Pluto was discovered in 1930 and Eris was discovered later.",
            ["classification", "location", "discovered_on"],
        )
        self.assertFalse(any(reason.startswith("ordering:") for reason in discovery_reasons))
        self.assertNotIn("ordered_discovery_section", discovery_reasons)

        _, size_reasons = self.score_chunk(
            size_question,
            "Pluto has the largest diameter among these dwarf planets in the Kuiper Belt.",
            ["classification", "location"],
        )
        self.assertIn("ordering:size", size_reasons)

        q9 = "Which dwarf planet located in the Kuiper Belt was discovered first?"
        _, q9_reasons = self.score_chunk(
            q9,
            "In order of discovery, Pluto was discovered in 1930 and Eris was discovered later.",
            ["classification", "location", "discovered_on"],
        )
        self.assertIn("ordering:discovered_on", q9_reasons)
        self.assertIn("ordered_discovery_section", q9_reasons)

    def test_target_class_bonus_is_symmetric(self) -> None:
        cases = [
            ("dwarf_planets", "This is a dwarf planet."),
            ("planets", "This is a planet."),
            ("moons", "This is a moon."),
            ("asteroids", "This is an asteroid."),
            ("comets", "This is a comet."),
        ]
        for target_class, text in cases:
            with self.subTest(target_class=target_class):
                score, reasons = score_target_class_evidence(text, target_class, weight=2.0)
                self.assertEqual(score, 2.0)
                self.assertEqual(reasons, [f"target_class:{target_class}"])


class MetadataRoutingTests(unittest.TestCase):
    @staticmethod
    def route(source_id: str, route_id: str, route_type: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk={
                "source_id": source_id,
                "route_id": route_id,
                "chunk_id": route_id,
                "route_type": route_type,
                "title": source_id,
                "url": "",
                "text": "",
                "word_start": 0,
                "word_end": 100,
            },
            score=score,
            reasons=[],
        )

    def test_metadata_route_is_capped_prior_not_primary_content_hit(self) -> None:
        retriever = object.__new__(RagRetriever)
        retriever.chunks = []
        retriever.question_classifier = QuestionClassifier()
        route_items = [
            self.route("broad", "broad__metadata", "metadata", 100.0),
            self.route("focused", "focused__section_0000", "section_window", 5.0),
            self.route("broad", "broad__section_0000", "section_window", 1.0),
        ]

        selection, _ = retriever._select_sources_from_routes(
            "Which object is relevant?",
            route_items=route_items,
            top_n_sources=2,
        )

        self.assertEqual(selection.scores[0].source_id, "focused")
        self.assertEqual(selection.selected_source_ids[0], "focused")
        broad_score = next(score for score in selection.scores if score.source_id == "broad")
        self.assertIn("metadata_prior:broad__metadata:2.0000", broad_score.reasons)


if __name__ == "__main__":
    unittest.main()

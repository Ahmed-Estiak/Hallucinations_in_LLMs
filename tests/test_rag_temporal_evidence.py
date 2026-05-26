"""Regression tests for validated temporal RAG evidence."""

from __future__ import annotations

import unittest

from src.question_classifier import QuestionClassifier
from src.rag.retrieval_intent import build_retrieval_intent
from src.rag.retriever import RagRetrievalResult, RagRetriever, RetrievedChunk
from src.rag.retriever_terms import build_query_terms
from src.rag.structured_satellite_facts import (
    extract_explicit_current_count_facts,
    extract_explicit_temporal_count_facts,
    extract_satellite_count_facts,
)
from src.rag.temporal_evidence import compatible_current_chunks, compatible_temporal_chunks


TABLE_TEXT = """
Satellites of Saturn:
3
I
Titan
1655
Discoverer
IAU WGPSN
S/2019 S1
2019
Discoverer
MPEC 2021-W14
S/2020 S1
2020
Discoverer
MPEC 2023-K04
Satellites of Neptune:
2
I
Triton
1846
Discoverer
IAU WGPSN
S/2021 N1
2021
Discoverer
MPEC 2024-D112
Satellites of Dwarf Planet Pluto:
1
I
Charon
1978
Discoverer
IAU WGPSN
"""


def fact_chunk(fact, chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_id": "source",
        "title": "Source",
        "section": fact.heading,
        "text": fact.text,
        "predicate_hints": ["moon_count"],
        "trust_level": "reference",
        "temporal_fact": fact.metadata(),
    }


class TemporalFactExtractionTests(unittest.TestCase):
    def test_generic_section_boundary_excludes_next_subject_rows(self) -> None:
        facts = extract_satellite_count_facts(TABLE_TEXT)
        neptune = next(fact for fact in facts if fact.fact_id == "neptune_satellite_count_as_of_2024_02")
        self.assertEqual(neptune.value, 2)

    def test_mismatched_declared_total_rejects_derived_facts(self) -> None:
        bad = TABLE_TEXT.replace("Satellites of Neptune:\n2", "Satellites of Neptune:\n1")
        facts = extract_satellite_count_facts(bad)
        self.assertFalse(any(fact.subject == "Neptune" for fact in facts))

    def test_validated_event_timeline_creates_next_change_interval(self) -> None:
        facts = extract_satellite_count_facts(TABLE_TEXT)
        saturn = next(fact for fact in facts if fact.fact_id == "saturn_satellite_count_as_of_2021_11")
        self.assertEqual(saturn.value, 2)
        self.assertEqual(saturn.valid_until_exclusive, "2023-05")
        self.assertEqual(saturn.interval_semantics, "carry_forward_until_next_validated_change")

    def test_explicit_dated_sentence_is_a_separate_high_grade_fact(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "As of November 2021, Saturn had 83 confirmed moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].evidence_type, "explicit_dated_sentence")
        self.assertEqual(facts[0].claim_type, "confirmed_moons")

    def test_explicit_dated_sentence_accepts_day_precision(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "As of March 25, 2025, Saturn had 274 confirmed moons."
        )
        self.assertEqual(facts[0].observed_at, "2025-03-25")

    def test_undated_direct_sentence_becomes_current_assertion_only(self) -> None:
        facts = extract_explicit_current_count_facts(
            "Neptune has 16 known moons. and has 5 moons. As of 2026, Saturn has 292 confirmed moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Neptune")
        self.assertEqual(facts[0].evidence_type, "explicit_current_sentence")
        self.assertEqual(facts[0].claim_type, "known_moons")


class TemporalCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        facts = extract_satellite_count_facts(TABLE_TEXT)
        self.chunks = [fact_chunk(fact, fact.fact_id) for fact in facts]

    def test_intermediate_month_is_supported_only_by_derived_interval(self) -> None:
        intent = build_retrieval_intent(
            "As of December 2021, how many confirmed moons did Saturn have?"
        )
        matches = compatible_temporal_chunks(self.chunks, intent)
        self.assertEqual([(chunk["temporal_fact"]["value"], kind) for chunk, kind in matches], [(2, "timeline_interval")])

    def test_last_anchor_does_not_assume_future_validity(self) -> None:
        intent = build_retrieval_intent(
            "As of March 2024, how many confirmed moons did Neptune have?"
        )
        self.assertEqual(compatible_temporal_chunks(self.chunks, intent), [])

    def test_explicit_fact_outranks_derived_anchor(self) -> None:
        explicit = extract_explicit_temporal_count_facts(
            "As of November 2021, Saturn had 2 confirmed moons."
        )[0]
        intent = build_retrieval_intent(
            "As of November 2021, how many confirmed moons did Saturn have?"
        )
        matches = compatible_temporal_chunks(
            [*self.chunks, fact_chunk(explicit, "explicit")],
            intent,
        )
        self.assertEqual(matches[0][1], "explicit_exact")

    def test_temporal_gate_blocks_conflicting_validated_facts(self) -> None:
        table_fact = next(
            fact for fact in extract_satellite_count_facts(TABLE_TEXT)
            if fact.fact_id == "saturn_satellite_count_as_of_2021_11"
        )
        explicit = extract_explicit_temporal_count_facts(
            "As of November 2021, Saturn had 3 confirmed moons."
        )[0]
        retriever = object.__new__(RagRetriever)
        retriever.question_classifier = QuestionClassifier()
        retriever.chunks = [fact_chunk(table_fact, "table"), fact_chunk(explicit, "explicit")]
        result = RagRetrievalResult(
            retrieved_chunks=[
                RetrievedChunk(chunk=retriever.chunks[0], score=1.0, reasons=[])
            ],
            retrieval_mode="global",
        )
        gated = retriever._apply_moon_count_evidence_gate(
            "As of November 2021, how many confirmed moons did Saturn have?",
            result,
            top_k=4,
        )
        self.assertEqual(gated.temporal_evidence_status, "conflict")

    def test_temporal_gate_removes_unscoped_current_count_claims(self) -> None:
        table_fact = next(
            fact for fact in extract_satellite_count_facts(TABLE_TEXT)
            if fact.fact_id == "saturn_satellite_count_as_of_2021_11"
        )
        raw_current = {
            "chunk_id": "raw_current",
            "source_id": "current_source",
            "title": "Saturn",
            "section": "Moons",
            "text": "Saturn has 292 confirmed moons in its orbit.",
            "predicate_hints": ["moon_count"],
        }
        retriever = object.__new__(RagRetriever)
        retriever.question_classifier = QuestionClassifier()
        retriever.chunks = [fact_chunk(table_fact, "table"), raw_current]
        result = RagRetrievalResult(
            retrieved_chunks=[
                RetrievedChunk(chunk=raw_current, score=20.0, reasons=[]),
            ],
            retrieval_mode="global",
        )
        gated = retriever._apply_moon_count_evidence_gate(
            "As of November 2021, how many confirmed moons did Saturn have?",
            result,
            top_k=4,
        )
        self.assertEqual(gated.temporal_evidence_status, "supported")
        self.assertNotIn("raw_current", [item.chunk["chunk_id"] for item in gated.retrieved_chunks])

    def test_outer_planet_alias_expansion_is_symmetric(self) -> None:
        self.assertIn("uranian", build_query_terms("Uranus"))
        self.assertIn("neptunian", build_query_terms("Neptune"))


class CurrentCountResolutionTests(unittest.TestCase):
    def test_current_resolver_rejects_claim_below_validated_history(self) -> None:
        timeline = extract_explicit_temporal_count_facts(
            "As of 2026, Saturn had 292 confirmed moons."
        )[0]
        current = extract_explicit_current_count_facts(
            "Saturn has 274 confirmed moons."
        )[0]
        intent = build_retrieval_intent("How many moons does Saturn have?")
        self.assertEqual(
            compatible_current_chunks(
                [fact_chunk(timeline, "dated"), fact_chunk(current, "current")],
                intent,
            ),
            [],
        )

    def test_current_resolver_selects_highest_admissible_claim(self) -> None:
        older = extract_explicit_current_count_facts("Neptune has 14 known moons.")[0]
        newer = extract_explicit_current_count_facts("Neptune has 16 known moons.")[0]
        intent = build_retrieval_intent("How many moons does Neptune have?")
        matches = compatible_current_chunks(
            [fact_chunk(older, "older"), fact_chunk(newer, "newer")],
            intent,
        )
        self.assertEqual([item[0]["temporal_fact"]["value"] for item in matches], [16])

    def test_current_gate_removes_wrong_subject_numeric_answer_chunks(self) -> None:
        current = extract_explicit_current_count_facts("Neptune has 16 known moons.")[0]
        older_current = extract_explicit_current_count_facts("Neptune has 14 known moons.")[0]
        uranus_raw = {
            "chunk_id": "uranus_raw",
            "source_id": "uranus_source",
            "title": "Uranus",
            "section": "Moons",
            "text": "Uranus has 29 known natural satellites.",
            "predicate_hints": ["moon_count"],
        }
        retriever = object.__new__(RagRetriever)
        retriever.question_classifier = QuestionClassifier()
        retriever.chunks = [
            fact_chunk(current, "neptune_current"),
            fact_chunk(older_current, "neptune_old"),
            uranus_raw,
        ]
        result = RagRetrievalResult(
            retrieved_chunks=[
                RetrievedChunk(chunk=uranus_raw, score=20.0, reasons=[]),
                RetrievedChunk(chunk=fact_chunk(older_current, "neptune_old"), score=19.0, reasons=[]),
            ],
            retrieval_mode="global",
        )
        gated = retriever._apply_moon_count_evidence_gate(
            "How many moons does Neptune have?",
            result,
            top_k=4,
        )
        self.assertEqual(gated.current_evidence_status, "supported")
        self.assertEqual(
            [item.chunk["chunk_id"] for item in gated.retrieved_chunks],
            ["neptune_current"],
        )


if __name__ == "__main__":
    unittest.main()

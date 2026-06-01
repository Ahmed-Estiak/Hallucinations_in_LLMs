"""Regression tests for validated temporal RAG evidence."""

from __future__ import annotations

import unittest

from src.question_classifier import QuestionClassifier
from src.rag.retrieval_intent import build_retrieval_intent
from src.rag.retriever import (
    RagRetrievalResult,
    RagRetriever,
    RetrievedChunk,
    score_moon_count_support_chunk,
)
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

    def test_dated_count_sentence_accepts_year_before_count(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "Updating the count of Saturn's moons in 2019, the planet now has 82 named moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Saturn")
        self.assertEqual(facts[0].value, 82)
        self.assertEqual(facts[0].observed_at, "2019")
        self.assertEqual(facts[0].claim_type, "named_moons")

    def test_dated_count_sentence_does_not_require_claim_wording(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "In 2024, Jupiter has 95 small outer moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Jupiter")
        self.assertEqual(facts[0].value, 95)
        self.assertEqual(facts[0].observed_at, "2024")
        self.assertEqual(facts[0].claim_type, "moon_count")

    def test_dated_count_sentence_accepts_generic_year_context(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "With 2024 observations, Jupiter has 95 known moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Jupiter")
        self.assertEqual(facts[0].observed_at, "2024")
        self.assertEqual(facts[0].claim_type, "known_moons")

    def test_adjacent_dated_context_supports_now_count_sentence(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "Tens of new moons around both Jupiter and Saturn have been announced in late 2022 and early 2023. "
            "Jupiter now has 95 and Saturn 145 con\ufb01rmed moons."
        )
        saturn = [fact for fact in facts if fact.subject == "Saturn"]
        self.assertEqual(len(saturn), 1)
        self.assertEqual(saturn[0].value, 145)
        self.assertEqual(saturn[0].observed_at, "2023")
        self.assertEqual(saturn[0].claim_type, "confirmed_moons")

    def test_adjacent_dated_context_requires_current_bridge_wording(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "The mission ended in 2017. Saturn has 82 known moons."
        )
        self.assertEqual(facts, [])

    def test_dated_count_sentence_does_not_borrow_previous_sentence_date(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "The mission ended in 2017. Saturn has 82 known moons."
        )
        self.assertEqual(facts, [])

    def test_dated_count_sentence_skips_ambiguous_pronoun_subject(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "In 2024, the planet has 95 known moons."
        )
        self.assertEqual(facts, [])

    def test_dated_count_sentence_skips_multiple_possessive_subjects(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "In 2024, Jupiter's moons and Saturn's moons are listed; the planet has 95 known moons."
        )
        self.assertEqual(facts, [])

    def test_undated_direct_sentence_becomes_current_assertion_only(self) -> None:
        facts = extract_explicit_current_count_facts(
            "Neptune has 16 known moons. and has 5 moons. As of 2026, Saturn has 292 confirmed moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Neptune")
        self.assertEqual(facts[0].evidence_type, "explicit_current_sentence")
        self.assertEqual(facts[0].claim_type, "known_moons")

    def test_undated_sentence_count_becomes_current_assertion_without_claim_wording(self) -> None:
        facts = extract_explicit_current_count_facts(
            "Saturn has 82 small outer moons."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Saturn")
        self.assertEqual(facts[0].value, 82)
        self.assertEqual(facts[0].evidence_type, "explicit_current_sentence")
        self.assertEqual(facts[0].claim_type, "moon_count")

    def test_year_and_moon_without_count_phrase_is_not_a_fact(self) -> None:
        facts = extract_explicit_temporal_count_facts(
            "In 2019, Saturn's moons were studied by astronomers."
        )
        self.assertEqual(facts, [])


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
        self.assertIn("next known count change in May 2023", matches[0][0]["text"])

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

    def test_cross_source_timeline_clips_single_source_interval(self) -> None:
        explicit = extract_explicit_temporal_count_facts(
            "As of January 2022, Saturn had 3 confirmed moons."
        )[0]
        intent = build_retrieval_intent(
            "As of February 2022, how many confirmed moons did Saturn have?"
        )
        matches = compatible_temporal_chunks(
            [*self.chunks, fact_chunk(explicit, "explicit_jan_2022")],
            intent,
        )
        self.assertEqual([(chunk["temporal_fact"]["value"], kind) for chunk, kind in matches], [(3, "timeline_interval")])
        self.assertIn("next known count change in May 2023", matches[0][0]["text"])

    def test_cross_source_timeline_keeps_earlier_value_before_intermediate_anchor(self) -> None:
        explicit = extract_explicit_temporal_count_facts(
            "As of January 2022, Saturn had 3 confirmed moons."
        )[0]
        intent = build_retrieval_intent(
            "As of December 2021, how many confirmed moons did Saturn have?"
        )
        matches = compatible_temporal_chunks(
            [*self.chunks, fact_chunk(explicit, "explicit_jan_2022")],
            intent,
        )
        self.assertEqual([(chunk["temporal_fact"]["value"], kind) for chunk, kind in matches], [(2, "timeline_interval")])
        self.assertIn("next known count change in January 2022", matches[0][0]["text"])

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

    def test_temporal_gate_keeps_context_when_interval_is_unresolved(self) -> None:
        dated = extract_explicit_temporal_count_facts(
            "Updating the count of Saturn's moons in 2019, the planet now has 82 named moons."
        )[0]
        source_chunk = {
            "chunk_id": "saturn_source",
            "source_id": "notes",
            "title": "Saturn notes",
            "section": "Moons",
            "text": "Updating the count of Saturn's moons in 2019, the planet now has 82 named moons.",
            "entities": ["Saturn"],
            "predicate_hints": ["moon_count"],
        }
        retriever = object.__new__(RagRetriever)
        retriever.question_classifier = QuestionClassifier()
        retriever.chunks = [fact_chunk(dated, "dated_anchor"), source_chunk]
        result = RagRetrievalResult(
            retrieved_chunks=[
                RetrievedChunk(chunk=source_chunk, score=20.0, reasons=["ranked_source"]),
            ],
            retrieval_mode="global",
        )
        gated = retriever._apply_moon_count_evidence_gate(
            "As of November 2021, how many confirmed moons did Saturn have?",
            result,
            top_k=4,
        )
        self.assertEqual(gated.temporal_evidence_status, "unresolved")
        self.assertEqual(
            {item.chunk["chunk_id"] for item in gated.retrieved_chunks},
            {"saturn_source", "dated_anchor"},
        )

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

    def test_confirmed_wording_uses_same_current_count_family(self) -> None:
        known = extract_explicit_current_count_facts("Saturn has 292 known moons.")[0]
        older_confirmed = extract_explicit_current_count_facts(
            "Saturn has 274 confirmed moons."
        )[0]
        intent = build_retrieval_intent("How many confirmed moons does Saturn have?")
        matches = compatible_current_chunks(
            [fact_chunk(known, "known"), fact_chunk(older_confirmed, "confirmed")],
            intent,
        )
        self.assertEqual([item[0]["temporal_fact"]["value"] for item in matches], [292])

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

    def test_moon_count_support_requires_number_and_moon_sentence(self) -> None:
        weak = {
            "chunk_id": "weak",
            "title": "Mars",
            "section": "Moons",
            "text": "Mars is near Earth. Its small companions orbit close to the planet.",
        }
        self.assertIsNone(score_moon_count_support_chunk(weak, "mars"))

    def test_moon_count_support_allows_pronoun_count_sentence(self) -> None:
        pronoun = {
            "chunk_id": "pronoun",
            "title": "Mars",
            "section": "Overview",
            "text": "Mars is a terrestrial planet. It has two tiny moons, Phobos and Deimos.",
        }
        scored = score_moon_count_support_chunk(pronoun, "mars")
        self.assertIsNotNone(scored)
        _score, reasons = scored
        self.assertIn("adjacent_sentence_entity_count_moon", reasons)

    def test_moon_count_support_prefers_same_sentence_entity(self) -> None:
        same = {
            "chunk_id": "same",
            "title": "",
            "section": "",
            "text": "Mars has two tiny moons, Phobos and Deimos.",
        }
        adjacent = {
            "chunk_id": "adjacent",
            "title": "",
            "section": "",
            "text": "Mars is a terrestrial planet. It has two tiny moons, Phobos and Deimos.",
        }
        same_score, _same_reasons = score_moon_count_support_chunk(same, "mars")
        adjacent_score, _adjacent_reasons = score_moon_count_support_chunk(adjacent, "mars")
        self.assertGreater(same_score, adjacent_score)


if __name__ == "__main__":
    unittest.main()

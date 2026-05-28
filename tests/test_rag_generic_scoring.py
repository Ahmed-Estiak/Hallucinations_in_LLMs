"""Regression tests for query-dependent RAG heuristic scoring."""

from __future__ import annotations

from collections import Counter
import unittest

from src.question_classifier import LogicalModifier, QuestionClassifier
from src.rag.retrieval_intent import build_retrieval_intent, score_filter_evidence, score_target_class_evidence
from src.rag.retriever import (
    RagRetriever,
    RetrievedChunk,
    cap_per_source,
    re_moon_count_claim,
    score_orbit_order_context,
    suppress_near_duplicate_chunks,
)
from src.rag.source_selector import SourceProfile, apply_constraint_coverage


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

    def test_relational_filters_preserve_attribute_and_reference_entity(self) -> None:
        q11 = "Which planets orbit beyond Earth yet have fewer moons than Jupiter?"
        classified = QuestionClassifier().classify(q11)
        self.assertEqual(classified.target_entity_class, "planets")
        self.assertEqual(classified.major_entities, ["Earth", "Jupiter"])
        self.assertIsNone(classified.ordering_attribute)
        self.assertEqual(classified.reference_entity, "Jupiter")
        self.assertIn(LogicalModifier.COMPARISON, classified.logical_modifiers)
        self.assertIn(LogicalModifier.FILTER, classified.logical_modifiers)
        conditions = build_retrieval_intent(q11).filter_conditions
        self.assertIn(
            {"operator": ">", "attribute": "distance_from_sun", "reference_entity": "Earth"},
            conditions,
        )
        self.assertIn(
            {"operator": "<", "attribute": "moon_count", "reference_entity": "Jupiter"},
            conditions,
        )

        counterfactual = "Which planets orbit beyond Mars yet have fewer moons than Saturn?"
        counterfactual_conditions = build_retrieval_intent(counterfactual).filter_conditions
        self.assertIn(
            {"operator": ">", "attribute": "distance_from_sun", "reference_entity": "Mars"},
            counterfactual_conditions,
        )
        self.assertIn(
            {"operator": "<", "attribute": "moon_count", "reference_entity": "Saturn"},
            counterfactual_conditions,
        )

        single_relation_conditions = build_retrieval_intent(
            "Which planets have fewer moons than Saturn?"
        ).filter_conditions
        self.assertIn(
            {"operator": "<", "attribute": "moon_count", "reference_entity": "Saturn"},
            single_relation_conditions,
        )

    def test_relational_filters_are_clause_order_independent(self) -> None:
        questions = [
            "Which planets orbit beyond Mars yet have fewer rings than Saturn?",
            "Which planets have fewer rings than Saturn and orbit beyond Mars?",
        ]
        for question in questions:
            with self.subTest(question=question):
                classified = QuestionClassifier().classify(question)
                self.assertEqual(classified.target_entity_class, "planets")
                self.assertEqual(set(classified.major_entities), {"Mars", "Saturn"})
                self.assertIsNone(classified.ordering_attribute)
                self.assertEqual(classified.reference_entity, "Saturn")
                conditions = build_retrieval_intent(question).filter_conditions
                self.assertIn(
                    {"operator": ">", "attribute": "distance_from_sun", "reference_entity": "Mars"},
                    conditions,
                )
                self.assertIn(
                    {"operator": "<", "attribute": "ring_count", "reference_entity": "Saturn"},
                    conditions,
                )

    def test_relational_filters_generalize_to_other_astronomy_attributes(self) -> None:
        dwarf_question = "Which dwarf planets located beyond Neptune have more mass than Ceres?"
        dwarf_classified = QuestionClassifier().classify(dwarf_question)
        self.assertEqual(dwarf_classified.target_entity_class, "dwarf_planets")
        self.assertEqual(dwarf_classified.major_entities, ["Neptune", "Ceres"])
        dwarf_conditions = build_retrieval_intent(dwarf_question).filter_conditions
        self.assertIn(
            {"operator": ">", "attribute": "distance_from_sun", "reference_entity": "Neptune"},
            dwarf_conditions,
        )
        self.assertIn(
            {"operator": ">", "attribute": "mass", "reference_entity": "Ceres"},
            dwarf_conditions,
        )

        moon_question = "Which moons orbit Jupiter and have larger diameter than Europa?"
        moon_classified = QuestionClassifier().classify(moon_question)
        self.assertEqual(moon_classified.target_entity_class, "moons")
        self.assertEqual(moon_classified.major_entities, ["Jupiter", "Europa"])
        moon_conditions = moon_classified.entity_filter_conditions
        self.assertIn(
            {"operator": "==", "attribute": "host_body", "reference_entity": "Jupiter"},
            moon_conditions,
        )
        self.assertIn(
            {"operator": ">", "attribute": "size", "reference_entity": "Europa"},
            moon_conditions,
        )

    def test_comparison_terms_are_expanded_from_the_attribute(self) -> None:
        intent = build_retrieval_intent(
            "Which planets orbit beyond Earth yet have fewer rings than Jupiter?"
        )
        self.assertIn("ring_count", intent.predicate_terms)
        self.assertIn("rings", intent.query_terms)
        self.assertNotIn("moons", intent.query_terms)
        self.assertNotIn("satellites", intent.query_terms)

    def test_filter_scoring_distinguishes_constraint_reference_from_candidate(self) -> None:
        conditions = [{"operator": "<", "attribute": "moon_count", "reference_entity": "Jupiter"}]
        _, reference_reasons = score_filter_evidence(
            "jupiter has 95 moons.",
            {"moon_count"},
            conditions,
            weight=2.5,
        )
        _, candidate_reasons = score_filter_evidence(
            "mars has 2 moons.",
            {"moon_count"},
            conditions,
            weight=2.5,
        )
        self.assertIn("constraint_reference:moon_count:jupiter", reference_reasons)
        self.assertIn("constraint_candidate:moon_count", candidate_reasons)
        self.assertNotIn("constraint_reference:moon_count:jupiter", candidate_reasons)

    def test_orbital_ordinal_evidence_is_symmetric(self) -> None:
        scores = []
        for planet, ordinal in [
            ("Mars", "fourth"),
            ("Jupiter", "fifth"),
            ("Saturn", "sixth"),
            ("Uranus", "seventh"),
            ("Neptune", "eighth"),
        ]:
            with self.subTest(planet=planet):
                score, reasons = score_orbit_order_context(
                    {"title": planet, "section": "Facts"},
                    f"{planet.lower()} is the {ordinal} planet from the sun.",
                )
                self.assertIn("planet_orbit_context", reasons)
                scores.append(score)
        self.assertEqual(len(set(scores)), 1)

    def test_moon_count_claim_accepts_descriptive_words_between_count_and_moons(self) -> None:
        self.assertTrue(re_moon_count_claim("Mars has two relatively small natural moons, Phobos and Deimos."))
        self.assertTrue(re_moon_count_claim("Neptune has 16 known moons."))
        self.assertTrue(re_moon_count_claim("Saturn has 292 known moons."))
        self.assertTrue(re_moon_count_claim("Jupiter has at least 115 moons."))
        self.assertTrue(re_moon_count_claim("Uranus has twenty-nine known natural satellites."))
        self.assertTrue(re_moon_count_claim("The planet has one hundred and fifteen confirmed moons."))

    def test_moon_count_claim_rejects_nearby_non_count_numbers(self) -> None:
        self.assertFalse(re_moon_count_claim("Phobos rises in the west, sets in the east, and rises again in 11 hours."))
        self.assertFalse(re_moon_count_claim("Mars orbits the Sun every 687 days."))


class SourceRoutingTests(unittest.TestCase):
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

    def test_catalog_route_is_capped_prior_not_primary_content_hit(self) -> None:
        retriever = object.__new__(RagRetriever)
        retriever.chunks = []
        retriever.question_classifier = QuestionClassifier()
        route_items = [
            self.route("broad", "broad__catalog", "source_catalog", 100.0),
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
        self.assertIn("catalog_prior:broad__catalog:2.0000", broad_score.reasons)

    def test_precise_identity_route_can_recover_a_source_with_weaker_content(self) -> None:
        retriever = object.__new__(RagRetriever)
        retriever.chunks = []
        retriever.question_classifier = QuestionClassifier()
        route_items = [
            self.route("precise", "precise__identity", "source_identity", 20.0),
            self.route("broad", "broad__section_0000", "section_window", 8.0),
            self.route("precise", "precise__section_0000", "section_window", 2.0),
        ]

        selection, _ = retriever._select_sources_from_routes(
            "How many moons does Pluto have?",
            route_items=route_items,
            top_n_sources=2,
        )

        self.assertEqual(selection.scores[0].source_id, "precise")
        precise_score = next(score for score in selection.scores if score.source_id == "precise")
        self.assertIn("identity_prior:precise__identity:8.0000", precise_score.reasons)

    @staticmethod
    def profile(source_id: str, title: str, text: str, entities: set[str]) -> SourceProfile:
        return SourceProfile(
            source_id=source_id,
            title=title,
            url="",
            cleaner="",
            trust_level="reference",
            char_count=len(text),
            chunk_count=1,
            text=text,
            tokens=Counter(text.split()),
            entities=entities,
            predicates={"moon_count", "distance_from_sun"},
        )

    def test_relational_candidate_coverage_uses_class_evidence_not_planet_whitelist(self) -> None:
        profiles = {
            "summary": self.profile(
                "summary",
                "Solar System",
                "the solar system contains planets and orbit data with moons.",
                {"Solar System"},
            ),
            "mars": self.profile("mars", "Mars", "mars is a planet and has moons in its orbit.", {"Mars"}),
            "saturn": self.profile("saturn", "Saturn", "saturn is a planet and has moons in its orbit.", {"Saturn"}),
            "kepler": self.profile("kepler", "Kepler-186f", "kepler-186f is a planet and has moons in its orbit.", {"Kepler-186f"}),
        }
        intent = build_retrieval_intent(
            "Which planets orbit beyond Mars yet have fewer moons than Saturn?"
        )
        expanded, added, reasons = apply_constraint_coverage(["summary"], profiles, intent)

        self.assertIn("mars", expanded)
        self.assertIn("saturn", expanded)
        self.assertIn("kepler", expanded)
        self.assertNotIn("summary", added)
        self.assertIn("reference_coverage:distance_from_sun:mars", reasons)
        self.assertIn("reference_coverage:moon_count:saturn", reasons)
        self.assertTrue(any(reason.startswith("candidate_coverage:kepler-186f:") for reason in reasons))

    def test_constraint_coverage_tracks_already_ranked_evidence_sources(self) -> None:
        profiles = {
            "mars": self.profile("mars", "Mars", "mars is a planet and has moons in its orbit.", {"Mars"}),
            "saturn": self.profile("saturn", "Saturn", "saturn is a planet and has moons in its orbit.", {"Saturn"}),
        }
        intent = build_retrieval_intent(
            "Which planets orbit beyond Mars yet have fewer moons than Saturn?"
        )
        _, coverage_sources, _ = apply_constraint_coverage(["mars", "saturn"], profiles, intent)
        self.assertEqual(set(coverage_sources), {"mars", "saturn"})

    def test_final_chunk_selection_preserves_required_evidence_sources(self) -> None:
        items = [
            self.route("summary", "summary_1", "section_window", 12.0),
            self.route("summary", "summary_2", "section_window", 11.0),
            self.route("candidate", "candidate_1", "section_window", 1.0),
        ]
        selected = cap_per_source(
            items,
            top_k=2,
            per_source_limit=2,
            required_source_ids=["candidate"],
        )
        self.assertEqual([item.chunk["source_id"] for item in selected], ["summary", "candidate"])

    def test_duplicate_suppression_preserves_required_evidence_sources(self) -> None:
        items = [
            self.route("summary", "summary_1", "section_window", 12.0),
            self.route("summary", "summary_2", "section_window", 11.0),
            self.route("candidate", "candidate_1", "section_window", 1.0),
        ]
        selected = suppress_near_duplicate_chunks(
            items,
            top_k=2,
            required_source_ids=["candidate"],
        )
        self.assertEqual([item.chunk["source_id"] for item in selected], ["summary", "candidate"])


if __name__ == "__main__":
    unittest.main()

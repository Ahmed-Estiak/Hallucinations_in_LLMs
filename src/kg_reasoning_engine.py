"""
KG Reasoning Engine: Advanced fact retrieval and reasoning based on question type
Handles time semantics, logic operators, multi-field queries, and computational reasoning
"""
import json
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from src.question_classifier import (
    ClassifiedQuestion, QuestionType, TimeSemantic, LogicOperator, LogicalModifier
)
from src.question_parser import TIME_RANGE_SEPARATOR
from src.time_utils import fact_matches_time, select_best_temporal_fact, latest_fact, time_window


class KGReasoningEngine:
    """
    Applies sophisticated reasoning strategies to KG based on question classification.
    Supports time filtering, filtering, sorting, comparison, and multi-field reasoning.
    """
    
    def __init__(self, kg_path: str = "data/astronomy_kg1.json"):
        """Load KG and initialize reasoning engine."""
        with open(kg_path, "r", encoding="utf-8") as f:
            self.kg = json.load(f)
        self.kg_indexed = self._index_kg()
    
    def _index_kg(self) -> Dict[str, List[Dict]]:
        """Index KG by subject for faster lookup."""
        indexed = {}
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            if subject not in indexed:
                indexed[subject] = []
            indexed[subject].append(fact)
        return indexed

    def _latest_fact_for(self, subject: str, predicate: str, cq: ClassifiedQuestion) -> Optional[Dict]:
        """Get the most relevant fact for one subject/predicate pair."""
        subject_norm = subject.lower()
        predicate_norm = predicate.lower()
        candidates = []
        for fact in self.kg_indexed.get(subject_norm, []):
            if fact.get("predicate", "").lower() != predicate_norm:
                continue
            candidates.append(fact)

        if not candidates:
            return None

        return select_best_temporal_fact(
            candidates,
            cq.time_value if cq.has_time_constraint else None,
            cq.time_semantic.name if cq.has_time_constraint and cq.time_semantic else None,
        )

    def _dedupe_fact_list(self, facts: List[Dict]) -> List[Dict]:
        """Remove duplicate facts while preserving order."""
        deduped: List[Dict] = []
        seen = set()
        for fact in facts:
            key = (
                fact.get("subject"),
                fact.get("predicate"),
                fact.get("object"),
                fact.get("time"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    def _get_candidate_entities(self, cq: ClassifiedQuestion) -> List[str]:
        """Resolve candidate pool for list and entity-selection questions."""
        pool_target = cq.target_entity_class or cq.list_target
        if pool_target == "planets":
            return self._candidate_planets()
        if pool_target == "dwarf_planets":
            return self._candidate_dwarf_planets()
        if pool_target == "moons":
            return self._candidate_moons()
        return []

    def _candidate_planets(self) -> List[str]:
        """
        Candidate planets inferred from distance facts.

        Assumption:
        - In this KG, major planets consistently have distance_from_sun facts.
        """
        planet_facts = {}
        for fact in self.kg:
            if fact.get("predicate", "").lower() != "distance_from_sun":
                continue
            subject = fact.get("subject")
            if not subject:
                continue
            current = planet_facts.get(subject)
            if current is None or self._extract_numeric_value(fact.get("object")) < self._extract_numeric_value(current.get("object")):
                planet_facts[subject] = fact
        return [
            subject
            for subject, _fact in sorted(
                planet_facts.items(),
                key=lambda item: self._extract_numeric_value(item[1].get("object"))
            )
        ]

    def _candidate_dwarf_planets(self) -> List[str]:
        """
        Candidate dwarf planets inferred from explicit classification facts.
        """
        return sorted({
            fact.get("subject")
            for fact in self.kg
            if fact.get("predicate", "").lower() == "classification" and str(fact.get("object", "")).lower() == "dwarf planet"
        })

    def _candidate_moons(self) -> List[str]:
        """
        Candidate moon entities.

        Current safety policy:
        - Only return subjects that look like actual moon entities via moon-specific classification/type facts.
        - Do not infer moons from moon_count on planets, which would produce a misleading candidate pool.
        - If the KG does not expose moon entities directly, fall back to names stored in
          `latest_known_moon` objects.
        """
        moon_candidates = sorted({
            fact.get("subject")
            for fact in self.kg
            if fact.get("subject") and (
                (
                    fact.get("predicate", "").lower() in {"classification", "planet_type"} and
                    str(fact.get("object", "")).lower() in {"moon", "natural satellite", "satellite"}
                ) or
                fact.get("predicate", "").lower() == "orbits" and str(fact.get("object", "")).lower() in {
                    "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"
                }
            )
        })
        if moon_candidates:
            return moon_candidates

        return sorted({
            str(fact.get("object", "")).strip()
            for fact in self.kg
            if fact.get("predicate", "").lower() == "latest_known_moon" and str(fact.get("object", "")).strip()
        })

    def _normalize_attribute_predicate(self, attribute: Optional[str]) -> Optional[str]:
        mapping = {
            "moons": "moon_count",
            "distance": "distance_from_sun",
            "discovered": "discovered_on",
            "size": "diameter",
            "mass": "mass",
            "classification": "classification",
            "planet_type": "planet_type",
            "surface_gravity": "surface_gravity",
            "location": "location",
            "diameter": "diameter",
        }
        if attribute is None:
            return None
        return mapping.get(attribute, attribute)

    def _build_derived_result_fact(self, title: str, derived_entities: List[str], supporting_facts: List[Dict], strategy: str) -> List[Dict]:
        return [{
            "_derived_result": True,
            "title": title,
            "entities": derived_entities,
            "supporting_facts": supporting_facts,
            "strategy": strategy,
        }]

    def _describe_list_target(self, cq: ClassifiedQuestion) -> str:
        pool_target = cq.target_entity_class or cq.list_target
        if pool_target == "planets":
            for condition in cq.entity_filter_conditions:
                if condition.get("attribute") == "planet_type" and condition.get("operator") == "==":
                    return f"{str(condition.get('value')).capitalize()} planets"
            return "Planets"
        if pool_target == "dwarf_planets":
            return "Dwarf planets"
        if pool_target == "moons":
            return "Moons"
        return "Entities"

    def _describe_ordering_attribute(self, predicate: str) -> str:
        labels = {
            "moon_count": "number of moons",
            "distance_from_sun": "distance from the Sun",
            "discovered_on": "discovery year",
            "diameter": "diameter",
            "mass": "mass",
            "surface_gravity": "surface gravity",
        }
        return labels.get(predicate, predicate.replace("_", " "))

    def _extract_numeric_value_or_none(self, value: Any) -> Optional[float]:
        """Return a numeric value when possible, otherwise None."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except:
                return None
        return None

    def _extract_ordering_value(self, predicate: str, fact: Dict) -> Optional[float]:
        """Return a comparable ordering value for numeric and date-like predicates."""
        if predicate == "discovered_on":
            window = time_window(fact.get("object"))
            if window is None:
                return None
            return float(window[0])
        return self._extract_numeric_value_or_none(fact.get("object"))
    
    def reason(self, classified_question: ClassifiedQuestion, 
               entities: List[str], 
               predicates: List[str]) -> Tuple[List[Dict], str]:
        """
        Apply reasoning strategy based on question classification.
        
        Returns:
            (list of facts, reasoning_strategy_used)
        """
        primary_type = classified_question.primary_type
        
        if primary_type == QuestionType.BOOLEAN:
            return self._reasoning_boolean(classified_question, entities, predicates)
        elif primary_type == QuestionType.ENTITY:
            if LogicalModifier.COMPARISON in classified_question.logical_modifiers:
                return self._reasoning_comparison(classified_question, entities, predicates)
            if classified_question.target_entity_class:
                return self._reasoning_entity_from_candidate_pool(classified_question, entities, predicates)
            return self._reasoning_entity(classified_question, entities, predicates)
        elif primary_type == QuestionType.LIST:
            if LogicalModifier.ORDERING in classified_question.logical_modifiers and LogicalModifier.FILTER in classified_question.logical_modifiers:
                return self._reasoning_list_filter_and_order(classified_question, entities, predicates)
            if LogicalModifier.ORDERING in classified_question.logical_modifiers:
                return self._reasoning_ordered_list(classified_question, entities, predicates)
            return self._reasoning_entity_list(classified_question, entities, predicates)
        elif primary_type == QuestionType.COUNT:
            return self._reasoning_count(classified_question, entities, predicates)
        elif primary_type == QuestionType.MULTI_FIELD:
            return self._reasoning_multi_field(classified_question, entities, predicates)
        else:
            # Default: generic retrieval
            return self._reasoning_generic(classified_question, entities, predicates)
    
    def _reasoning_boolean(self, cq: ClassifiedQuestion, 
                          entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Boolean reasoning: find yes/no answer facts."""
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates]

        selected = []
        for entity in entities_norm:
            for predicate in predicates_norm:
                fact = self._latest_fact_for(entity, predicate, cq)
                if fact:
                    selected.append(fact)

        if selected:
            return self._dedupe_fact_list(selected), "boolean_direct_lookup"

        fallback = []
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            predicate = fact.get("predicate", "").lower()
            if subject in entities_norm and predicate in predicates_norm:
                fallback.append(fact)
        return self._dedupe_fact_list(fallback[:3]), "boolean_direct_lookup"
    
    def _reasoning_entity(self, cq: ClassifiedQuestion,
                         entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Entity reasoning: find single entity facts."""
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates]

        facts = []
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            predicate = fact.get("predicate", "").lower()
            if subject not in entities_norm:
                continue
            if predicates_norm and predicate not in predicates_norm:
                continue
            facts.append(fact)

        if predicates_norm:
            selected = []
            for entity in entities_norm:
                for predicate in predicates_norm:
                    fact = self._latest_fact_for(entity, predicate, cq)
                    if fact:
                        selected.append(fact)
            selected = self._dedupe_fact_list(selected)
            if selected:
                return selected[:1], "entity_lookup"

        if cq.has_time_constraint:
            best = select_best_temporal_fact(
                facts,
                cq.time_value,
                cq.time_semantic.name if cq.time_semantic else None,
            )
            return ([best] if best else []), "entity_lookup"

        best = latest_fact(facts)
        return ([best] if best else facts[:1]), "entity_lookup"

    def _reasoning_entity_from_candidate_pool(self, cq: ClassifiedQuestion,
                                              entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Select one entity from a target class using shared filter/order candidate-pool logic."""
        if LogicalModifier.ORDERING in cq.logical_modifiers and LogicalModifier.FILTER in cq.logical_modifiers:
            derived_result, strategy = self._reasoning_list_filter_and_order(cq, entities, predicates)
        elif LogicalModifier.ORDERING in cq.logical_modifiers or cq.ordering_attribute:
            derived_result, strategy = self._reasoning_ordered_list(cq, entities, predicates)
        elif cq.entity_filter_conditions:
            derived_result, strategy = self._reasoning_entity_list(cq, entities, predicates)
        else:
            return self._reasoning_entity(cq, entities, predicates)

        if not derived_result or not derived_result[0].get("_derived_result"):
            return derived_result, strategy

        result = derived_result[0]
        derived_entities = result.get("entities", [])
        if not derived_entities:
            return [], strategy

        selected_entity = derived_entities[0]
        title = f"Selected {self._describe_list_target(cq).rstrip('s').lower()}"
        return self._build_derived_result_fact(
            title,
            [selected_entity],
            result.get("supporting_facts", []),
            f"{strategy}_entity_selection",
        ), f"{strategy}_entity_selection"
    
    def _reasoning_entity_list(self, cq: ClassifiedQuestion,
                              entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Entity list reasoning: compute filtered list from KG where possible."""
        candidates = self._get_candidate_entities(cq)
        if not candidates:
            return self._reasoning_generic(cq, entities, predicates)

        derived_entities = candidates[:]
        supporting_facts: List[Dict] = []

        for condition in cq.entity_filter_conditions:
            attribute = condition.get("attribute")
            operator = condition.get("operator")
            predicate = self._normalize_attribute_predicate(attribute)

            if operator == "==" and predicate:
                condition_value = condition.get("value")
                if condition_value is not None:
                    next_entities = []
                    for entity in derived_entities:
                        fact = self._latest_fact_for(entity, predicate, cq)
                        if fact and str(fact.get("object", "")).lower() == str(condition_value).lower():
                            next_entities.append(entity)
                            supporting_facts.append(fact)
                    derived_entities = next_entities
                    continue

            if operator in {"<", ">", "=="} and predicate:
                reference_entity = condition.get("reference_entity")
                reference_value = condition.get("value")
                if reference_entity is None and reference_value is None:
                    if cq.helper_entities:
                        reference_entity = cq.helper_entities[-1]
                    elif len(entities) == 1:
                        reference_entity = entities[0]
                if reference_entity:
                    reference_fact = self._latest_fact_for(reference_entity, predicate, cq)
                    if reference_fact:
                        reference_value = reference_fact.get("object")
                        supporting_facts.append(reference_fact)
                if reference_value is None:
                    continue

                reference_numeric = self._extract_numeric_value_or_none(reference_value)
                if reference_numeric is None:
                    continue
                next_entities = []
                for entity in derived_entities:
                    if reference_entity and entity.lower() == str(reference_entity).lower():
                        continue
                    fact = self._latest_fact_for(entity, predicate, cq)
                    if not fact:
                        continue
                    numeric_value = self._extract_numeric_value_or_none(fact.get("object"))
                    if numeric_value is None:
                        continue
                    passed = (
                        operator == "<" and numeric_value < reference_numeric or
                        operator == ">" and numeric_value > reference_numeric or
                        operator == "==" and numeric_value == reference_numeric
                    )
                    if passed:
                        next_entities.append(entity)
                        supporting_facts.append(fact)
                derived_entities = next_entities

        title = f"Filtered {self._describe_list_target(cq).lower()} matching all conditions"
        return self._build_derived_result_fact(
            title,
            derived_entities,
            self._dedupe_fact_list(supporting_facts),
            "list_filtering",
        ), "list_filtering"
    
    def _reasoning_ordered_list(self, cq: ClassifiedQuestion,
                               entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Ordered list reasoning: sort by attribute (computational)."""
        ordering_attr = self._normalize_attribute_predicate(cq.ordering_attribute) or "mass"
        candidates = self._get_candidate_entities(cq)
        if not candidates:
            candidates = sorted({
                fact.get("subject")
                for fact in self.kg
                if fact.get("subject") and fact.get("predicate", "").lower() == ordering_attr
            })

        filtered_candidates = candidates[:]
        supporting_facts: List[Dict] = []

        for condition in cq.entity_filter_conditions:
            if condition.get("operator") == "==" and condition.get("attribute") == "planet_type":
                next_entities = []
                for entity in filtered_candidates:
                    fact = self._latest_fact_for(entity, "planet_type", cq)
                    if fact and str(fact.get("object", "")).lower() == str(condition.get("value", "")).lower():
                        next_entities.append(entity)
                        supporting_facts.append(fact)
                filtered_candidates = next_entities

        sortable_rows = []
        for entity in filtered_candidates:
            fact = self._latest_fact_for(entity, ordering_attr, cq)
            if not fact:
                continue
            ordering_value = self._extract_ordering_value(ordering_attr, fact)
            if ordering_value is None:
                continue
            supporting_facts.append(fact)
            sortable_rows.append((entity, fact, ordering_value))

        sortable_rows.sort(
            key=lambda item: item[2],
            reverse=(cq.order_direction == "descending")
        )
        derived_entities = [entity for entity, _, _ in sortable_rows]
        order_text = "decreasing" if cq.order_direction == "descending" else "increasing"
        title = f"{self._describe_list_target(cq)} ordered by {order_text} {self._describe_ordering_attribute(ordering_attr)}"
        strategy = f"ordered_list_by_{ordering_attr}_{cq.order_direction}"
        return self._build_derived_result_fact(
            title,
            derived_entities,
            self._dedupe_fact_list(supporting_facts),
            strategy,
        ), strategy

    def _reasoning_list_filter_and_order(self, cq: ClassifiedQuestion,
                                         entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Filter candidates first, then order them."""
        filtered_result, _strategy = self._reasoning_entity_list(cq, entities, predicates)
        if not filtered_result or not filtered_result[0].get("_derived_result"):
            return self._reasoning_ordered_list(cq, entities, predicates)

        ordering_attr = self._normalize_attribute_predicate(cq.ordering_attribute) or "mass"
        derived_entities = filtered_result[0].get("entities", [])
        supporting_facts = list(filtered_result[0].get("supporting_facts", []))
        sortable_rows = []
        for entity in derived_entities:
            fact = self._latest_fact_for(entity, ordering_attr, cq)
            if not fact:
                continue
            ordering_value = self._extract_ordering_value(ordering_attr, fact)
            if ordering_value is None:
                continue
            supporting_facts.append(fact)
            sortable_rows.append((entity, fact, ordering_value))
        sortable_rows.sort(
            key=lambda item: item[2],
            reverse=(cq.order_direction == "descending")
        )
        ordered_entities = [entity for entity, _, _ in sortable_rows]
        order_text = "decreasing" if cq.order_direction == "descending" else "increasing"
        title = f"Filtered {self._describe_list_target(cq).lower()} ordered by {order_text} {self._describe_ordering_attribute(ordering_attr)}"
        strategy = f"list_filter_and_order_by_{ordering_attr}"
        return self._build_derived_result_fact(
            title,
            ordered_entities,
            self._dedupe_fact_list(supporting_facts),
            strategy,
        ), strategy
    
    def _reasoning_comparison(self, cq: ClassifiedQuestion,
                             entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Comparison reasoning: compare entities by attribute."""
        comparison_attr = self._normalize_attribute_predicate(cq.ordering_attribute) or "mass"
        entities_norm = [e.lower() for e in entities]

        facts = []
        for entity in entities_norm:
            fact = self._latest_fact_for(entity, comparison_attr, cq)
            if fact:
                facts.append(fact)

        facts = self._dedupe_fact_list(facts)
        return facts, f"comparison_by_{comparison_attr}"
    
    def _reasoning_count(self, cq: ClassifiedQuestion,
                        entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Count reasoning: aggregate numbers."""
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates]

        facts = []
        for entity in entities_norm:
            for predicate in predicates_norm:
                fact = self._latest_fact_for(entity, predicate, cq)
                if fact:
                    facts.append(fact)

        facts = self._dedupe_fact_list(facts)
        if facts:
            return facts[:1], "count_aggregation"
        return facts, "count_aggregation"
    
    def _reasoning_time_lookup(self, cq: ClassifiedQuestion,
                              entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Time lookup reasoning: find facts for specific time."""
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates] or [p.lower() for p in cq.major_predicates]
        facts = []

        if entities_norm and predicates_norm:
            for entity in entities_norm:
                for predicate in predicates_norm:
                    fact = self._latest_fact_for(entity, predicate, cq)
                    if fact:
                        facts.append(fact)
        else:
            for fact in self.kg:
                subject = fact.get("subject", "").lower()
                predicate = fact.get("predicate", "").lower()
                if entities_norm and subject not in entities_norm:
                    continue
                if predicates_norm and predicate not in predicates_norm:
                    continue
                facts.append(fact)

        facts = self._dedupe_fact_list(facts)
        if facts:
            best = select_best_temporal_fact(
                facts,
                cq.time_value if cq.has_time_constraint else None,
                cq.time_semantic.name if cq.has_time_constraint and cq.time_semantic else None,
            )
            return ([best] if best else []), f"time_lookup_{cq.time_semantic.value}_{cq.time_value}"
        return facts, f"time_lookup_{cq.time_semantic.value}_{cq.time_value}"
    
    def _reasoning_multi_field(self, cq: ClassifiedQuestion,
                              entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Multi-field reasoning: combine multiple predicates."""
        all_facts = []

        if cq.fields:
            for field_spec in cq.fields:
                if not field_spec.entity or not field_spec.predicate:
                    continue
                fact = self._latest_fact_for(field_spec.entity, field_spec.predicate, cq)
                if fact:
                    all_facts.append(fact)
            return self._dedupe_fact_list(all_facts), "multi_field_combination"

        entities_norm = [e.lower() for e in entities]
        multi_predicates = [p.lower() for p in cq.multi_field_predicates]
        for entity in entities_norm:
            for predicate in multi_predicates:
                fact = self._latest_fact_for(entity, predicate, cq)
                if fact:
                    all_facts.append(fact)

        return self._dedupe_fact_list(all_facts), "multi_field_combination"
    
    def _reasoning_generic(self, cq: ClassifiedQuestion,
                          entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Generic reasoning: fallback for uncertain types."""
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates] or [p.lower() for p in cq.major_predicates]

        direct_facts = []
        if entities_norm and predicates_norm:
            for entity in entities_norm:
                for predicate in predicates_norm:
                    fact = self._latest_fact_for(entity, predicate, cq)
                    if fact:
                        direct_facts.append(fact)
            direct_facts = self._dedupe_fact_list(direct_facts)
            if direct_facts:
                return direct_facts[:1], "generic_fallback"

        facts = []
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            predicate = fact.get("predicate", "").lower()
            
            entity_match = not entities_norm or subject in entities_norm
            predicate_match = not predicates_norm or predicate in predicates_norm
            
            if entity_match and predicate_match:
                facts.append(fact)

        if cq.has_time_constraint and facts:
            best = select_best_temporal_fact(
                facts,
                cq.time_value,
                cq.time_semantic.name if cq.time_semantic else None,
            )
            return ([best] if best else []), "generic_fallback"

        latest = latest_fact(facts)
        deduped_facts = self._dedupe_fact_list(facts)
        return ([latest] if latest else deduped_facts[:1]), "generic_fallback"
    
    # Helper methods
    
    def _time_matches(self, fact_time: Optional[str], cq: ClassifiedQuestion) -> bool:
        """Check if fact time matches the question's time constraint."""
        if not cq.has_time_constraint:
            return True
        return fact_matches_time(fact_time, cq.time_value, cq.time_semantic.name if cq.time_semantic else None)
    
    def _extract_numeric_value(self, value: Any) -> float:
        """Extract numeric value from fact object (handles scientific notation)."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except:
                return 0.0
        return 0.0


def format_reasoned_facts(facts: List[Dict], reasoning_strategy: str) -> str:
    """Format reasoned facts for prompt."""
    if not facts:
        return "No relevant KG facts found."

    if facts and facts[0].get("_derived_result"):
        result = facts[0]
        entities = result.get("entities", [])
        title = result.get("title", "Derived KG Result")
        formatted = "Derived KG Result:\n"
        formatted += f"{title}:\n"
        formatted += ", ".join(entities) if entities else "No matching entities found."
        return formatted
    
    formatted = "Knowledge Graph Facts:\n"
    for i, fact in enumerate(facts, 1):
        subject = fact.get("subject", "?")
        predicate = fact.get("predicate", "?")
        obj = fact.get("object", "?")
        time = fact.get("time")

        formatted += f"{i}. {subject} | {predicate} = {obj}"
        if time:
            formatted += f" | time: {time}"
        formatted += "\n"
    
    return formatted

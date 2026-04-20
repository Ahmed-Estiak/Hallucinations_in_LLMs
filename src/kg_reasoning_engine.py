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
from src.time_utils import fact_matches_time, select_best_temporal_fact, latest_fact


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

    def _get_candidate_entities(self, cq: ClassifiedQuestion) -> List[str]:
        """Resolve candidate pool for list questions."""
        if cq.list_target == "planets":
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
        if cq.list_target == "dwarf_planets":
            return sorted({
                fact.get("subject")
                for fact in self.kg
                if fact.get("predicate", "").lower() == "classification" and str(fact.get("object", "")).lower() == "dwarf planet"
            })
        if cq.list_target == "moons":
            return sorted({
                fact.get("subject")
                for fact in self.kg
                if fact.get("predicate", "").lower() in {"moon_count", "latest_known_moon"}
            })
        return []

    def _normalize_attribute_predicate(self, attribute: Optional[str]) -> Optional[str]:
        mapping = {
            "moons": "moon_count",
            "distance": "distance_from_sun",
            "discovered": "discovered_on",
            "size": "diameter",
            "mass": "mass",
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
        if cq.list_target == "planets":
            for condition in cq.entity_filter_conditions:
                if condition.get("attribute") == "planet_type" and condition.get("operator") == "==":
                    return f"{str(condition.get('value')).capitalize()} planets"
            return "Planets"
        if cq.list_target == "dwarf_planets":
            return "Dwarf planets"
        if cq.list_target == "moons":
            return "Moons"
        return "Entities"
    
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
        facts = []
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates]
        
        # Find matching classification or status facts
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            predicate = fact.get("predicate", "").lower()
            
            if subject in entities_norm and predicate in predicates_norm:
                facts.append(fact)

        selected = []
        for entity in entities_norm:
            for predicate in predicates_norm:
                fact = self._latest_fact_for(entity, predicate, cq)
                if fact:
                    selected.append(fact)
        return selected or facts, "boolean_direct_lookup"
    
    def _reasoning_entity(self, cq: ClassifiedQuestion,
                         entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Entity reasoning: find single entity facts."""
        facts = []
        entities_norm = [e.lower() for e in entities]
        
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            if subject in entities_norm:
                facts.append(fact)

        if cq.has_time_constraint:
            best = select_best_temporal_fact(
                facts,
                cq.time_value,
                cq.time_semantic.name if cq.time_semantic else None,
            )
            return ([best] if best else []), "entity_lookup"

        best = latest_fact(facts)
        return ([best] if best else facts[:1]), "entity_lookup"
    
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
                next_entities = []
                for entity in derived_entities:
                    fact = self._latest_fact_for(entity, predicate, cq)
                    if fact and str(fact.get("object", "")).lower() == str(condition.get("value", "")).lower():
                        next_entities.append(entity)
                        supporting_facts.append(fact)
                derived_entities = next_entities

            elif operator in {"<", ">", "=="} and predicate:
                reference_entity = condition.get("reference_entity")
                reference_value = condition.get("value")
                if reference_entity is None and entities:
                    reference_entity = entities[-1]
                if reference_entity:
                    reference_fact = self._latest_fact_for(reference_entity, predicate, cq)
                    if reference_fact:
                        reference_value = reference_fact.get("object")
                        supporting_facts.append(reference_fact)
                if reference_value is None:
                    continue

                reference_numeric = self._extract_numeric_value(reference_value)
                next_entities = []
                for entity in derived_entities:
                    if reference_entity and entity.lower() == str(reference_entity).lower():
                        continue
                    fact = self._latest_fact_for(entity, predicate, cq)
                    if not fact:
                        continue
                    numeric_value = self._extract_numeric_value(fact.get("object"))
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
        return self._build_derived_result_fact(title, derived_entities, supporting_facts, "list_filtering"), "list_filtering"
    
    def _reasoning_ordered_list(self, cq: ClassifiedQuestion,
                               entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Ordered list reasoning: sort by attribute (computational)."""
        ordering_attr = self._normalize_attribute_predicate(cq.ordering_attribute) or "mass"
        candidates = self._get_candidate_entities(cq)
        if not candidates:
            candidates = sorted({fact.get("subject") for fact in self.kg if fact.get("predicate", "").lower() == ordering_attr})

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
            if fact:
                supporting_facts.append(fact)
                sortable_rows.append((entity, fact))

        sortable_rows.sort(
            key=lambda item: self._extract_numeric_value(item[1].get("object")),
            reverse=(cq.order_direction == "descending")
        )
        derived_entities = [entity for entity, _ in sortable_rows]
        order_text = "decreasing" if cq.order_direction == "descending" else "increasing"
        title = f"{self._describe_list_target(cq)} ordered by {order_text} {ordering_attr}"
        return self._build_derived_result_fact(title, derived_entities, supporting_facts, f"ordered_list_by_{ordering_attr}_{cq.order_direction}"), f"ordered_list_by_{ordering_attr}_{cq.order_direction}"

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
            if fact:
                supporting_facts.append(fact)
                sortable_rows.append((entity, fact))
        sortable_rows.sort(
            key=lambda item: self._extract_numeric_value(item[1].get("object")),
            reverse=(cq.order_direction == "descending")
        )
        ordered_entities = [entity for entity, _ in sortable_rows]
        order_text = "decreasing" if cq.order_direction == "descending" else "increasing"
        title = f"Filtered {self._describe_list_target(cq).lower()} ordered by {order_text} {ordering_attr}"
        return self._build_derived_result_fact(title, ordered_entities, supporting_facts, f"list_filter_and_order_by_{ordering_attr}"), f"list_filter_and_order_by_{ordering_attr}"
    
    def _reasoning_comparison(self, cq: ClassifiedQuestion,
                             entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Comparison reasoning: compare entities by attribute."""
        facts = []
        comparison_attr = cq.ordering_attribute or "mass"
        entities_norm = [e.lower() for e in entities]
        
        # Get attribute facts for comparison entities
        entity_values = {}
        for fact in self.kg:
            if fact.get("predicate", "").lower() == comparison_attr.lower():
                subject = fact.get("subject", "").lower()
                if subject in entities_norm:
                    if cq.has_time_constraint:
                        if not fact_matches_time(fact.get("time"), cq.time_value, cq.time_semantic.name if cq.time_semantic else None):
                            continue
                    
                    if subject not in entity_values:
                        entity_values[subject] = fact
                    elif self._is_more_recent(fact, entity_values[subject]):
                        entity_values[subject] = fact
        
        facts = list(entity_values.values())
        return facts, f"comparison_by_{comparison_attr}"
    
    def _reasoning_count(self, cq: ClassifiedQuestion,
                        entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Count reasoning: aggregate numbers."""
        facts = []
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates]
        
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            predicate = fact.get("predicate", "").lower()
            
            if subject in entities_norm and predicate in predicates_norm:
                facts.append(fact)

        if facts:
            best = select_best_temporal_fact(
                facts,
                cq.time_value if cq.has_time_constraint else None,
                cq.time_semantic.name if cq.has_time_constraint and cq.time_semantic else None,
            )
            return ([best] if best else []), "count_aggregation"
        return facts, "count_aggregation"
    
    def _reasoning_time_lookup(self, cq: ClassifiedQuestion,
                              entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Time lookup reasoning: find facts for specific time."""
        facts = []
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates] or ["discovered_on", "discovered_by"]
        
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            predicate = fact.get("predicate", "").lower()
            
            if (not entities_norm or subject in entities_norm) and \
               (not predicates_norm or predicate in predicates_norm):
                facts.append(fact)

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
        entities_norm = [e.lower() for e in entities]
        
        # For each entity, get all matching predicates
        for entity in entities_norm:
            entity_facts = {}
            for fact in self.kg:
                if fact.get("subject", "").lower() == entity:
                    predicate = fact.get("predicate", "").lower()

                    # Match any multi-field predicate
                    if predicate in [p.lower() for p in cq.multi_field_predicates]:
                        entity_facts.setdefault(predicate, []).append(fact)

            for predicate_facts in entity_facts.values():
                best = select_best_temporal_fact(
                    predicate_facts,
                    cq.time_value if cq.has_time_constraint else None,
                    cq.time_semantic.name if cq.has_time_constraint and cq.time_semantic else None,
                )
                if best:
                    all_facts.append(best)
        
        return all_facts, "multi_field_combination"
    
    def _reasoning_generic(self, cq: ClassifiedQuestion,
                          entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Generic reasoning: fallback for uncertain types."""
        entities_norm = [e.lower() for e in entities]
        predicates_norm = [p.lower() for p in predicates]
        
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
        return ([latest] if latest else facts[:1]), "generic_fallback"
    
    # Helper methods
    
    def _time_matches(self, fact_time: Optional[str], cq: ClassifiedQuestion) -> bool:
        """Check if fact time matches the question's time constraint."""
        if not cq.has_time_constraint:
            return True
        return fact_matches_time(fact_time, cq.time_value, cq.time_semantic.name if cq.time_semantic else None)
    
    def _is_more_recent(self, fact1: Dict, fact2: Dict) -> bool:
        """Compare two facts by time recency."""
        time1 = fact1.get("time", "0000")
        time2 = fact2.get("time", "0000")
        
        try:
            year1 = int(str(time1).split("-")[0])
            year2 = int(str(time2).split("-")[0])
            return year1 > year2
        except (ValueError, TypeError):
            return False
    
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
        return f"No facts found using strategy: {reasoning_strategy}"

    if facts and facts[0].get("_derived_result"):
        result = facts[0]
        entities = result.get("entities", [])
        title = result.get("title", "Derived KG Result")
        formatted = "Derived KG Result:\n"
        formatted += f"{title}:\n"
        formatted += ", ".join(entities) if entities else "No matching entities found."
        return formatted
    
    formatted = f"Knowledge Graph Facts (Strategy: {reasoning_strategy}):\n"
    for i, fact in enumerate(facts, 1):
        subject = fact.get("subject", "?")
        predicate = fact.get("predicate", "?")
        obj = fact.get("object", "?")
        time = fact.get("time", "unknown")
        
        formatted += f"{i}. {subject} | {predicate} = {obj} (as of {time})\n"
    
    return formatted

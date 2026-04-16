"""
KG Reasoning Engine: Advanced fact retrieval and reasoning based on question type
Handles time semantics, logic operators, multi-field queries, and computational reasoning
"""
import json
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from src.question_classifier import (
    ClassifiedQuestion, QuestionType, TimeSemantic, LogicOperator
)


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
            return self._reasoning_entity(classified_question, entities, predicates)
        elif primary_type == QuestionType.ENTITY_LIST:
            return self._reasoning_entity_list(classified_question, entities, predicates)
        elif primary_type == QuestionType.ORDERED_LIST:
            return self._reasoning_ordered_list(classified_question, entities, predicates)
        elif primary_type == QuestionType.COMPARISON:
            return self._reasoning_comparison(classified_question, entities, predicates)
        elif primary_type == QuestionType.COUNT:
            return self._reasoning_count(classified_question, entities, predicates)
        elif primary_type == QuestionType.TIME_LOOKUP:
            return self._reasoning_time_lookup(classified_question, entities, predicates)
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
                # Apply time filtering if needed
                if cq.has_time_constraint:
                    if self._time_matches(fact.get("time"), cq):
                        facts.append(fact)
                else:
                    facts.append(fact)
        
        return facts, "boolean_direct_lookup"
    
    def _reasoning_entity(self, cq: ClassifiedQuestion,
                         entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Entity reasoning: find single entity facts."""
        facts = []
        entities_norm = [e.lower() for e in entities]
        
        for fact in self.kg:
            subject = fact.get("subject", "").lower()
            if subject in entities_norm:
                if cq.has_time_constraint:
                    if self._time_matches(fact.get("time"), cq):
                        facts.append(fact)
                else:
                    # Take most recent if no time constraint
                    if not facts or self._is_more_recent(fact, facts[0]):
                        facts = [fact]
        
        return facts, "entity_lookup"
    
    def _reasoning_entity_list(self, cq: ClassifiedQuestion,
                              entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Entity list reasoning: find multiple entities with filtering."""
        facts = []
        predicates_norm = [p.lower() for p in predicates]
        
        # Group facts by predicate
        predicate_facts = {}
        for fact in self.kg:
            predicate = fact.get("predicate", "").lower()
            if predicate in predicates_norm:
                if predicate not in predicate_facts:
                    predicate_facts[predicate] = []
                
                # Apply time filtering
                if cq.has_time_constraint:
                    if self._time_matches(fact.get("time"), cq):
                        predicate_facts[predicate].append(fact)
                else:
                    predicate_facts[predicate].append(fact)
        
        # Combine facts (keep latest per entity)
        combined = {}
        for pred, pred_facts in predicate_facts.items():
            for fact in pred_facts:
                subject = fact.get("subject", "").lower()
                if subject not in combined:
                    combined[subject] = fact
                elif self._is_more_recent(fact, combined[subject]):
                    combined[subject] = fact
        
        facts = list(combined.values())
        return facts, "entity_list_filtering"
    
    def _reasoning_ordered_list(self, cq: ClassifiedQuestion,
                               entities: List[str], predicates: List[str]) -> Tuple[List[Dict], str]:
        """Ordered list reasoning: sort by attribute (computational)."""
        facts = []
        predicates_norm = [p.lower() for p in predicates]
        ordering_attr = cq.ordering_attribute or "mass"
        
        # Get all facts with the ordering attribute
        attr_facts = {}
        for fact in self.kg:
            if fact.get("predicate", "").lower() == ordering_attr.lower():
                subject = fact.get("subject", "")
                
                if cq.has_time_constraint:
                    if not self._time_matches(fact.get("time"), cq):
                        continue
                
                if subject not in attr_facts:
                    attr_facts[subject] = fact
                elif self._is_more_recent(fact, attr_facts[subject]):
                    attr_facts[subject] = fact
        
        # Sort by object value
        sorted_facts = sorted(
            attr_facts.values(),
            key=lambda x: self._extract_numeric_value(x.get("object")),
            reverse=(cq.order_direction == "descending")
        )
        
        facts = sorted_facts
        return facts, f"ordered_list_by_{ordering_attr}_{cq.order_direction}"
    
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
                        if not self._time_matches(fact.get("time"), cq):
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
                if cq.has_time_constraint:
                    if self._time_matches(fact.get("time"), cq):
                        facts.append(fact)
                else:
                    # Take most recent for count
                    if not facts or self._is_more_recent(fact, facts[0]):
                        facts = [fact]
        
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
                if cq.has_time_constraint:
                    if self._time_matches(fact.get("time"), cq):
                        facts.append(fact)
        
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
                        if cq.has_time_constraint:
                            if self._time_matches(fact.get("time"), cq):
                                entity_facts[predicate] = fact
                        else:
                            if predicate not in entity_facts or \
                               self._is_more_recent(fact, entity_facts[predicate]):
                                entity_facts[predicate] = fact
            
            all_facts.extend(entity_facts.values())
        
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
                if cq.has_time_constraint:
                    if self._time_matches(fact.get("time"), cq):
                        facts.append(fact)
                else:
                    facts.append(fact)
        
        return facts[:3], "generic_fallback"
    
    # Helper methods
    
    def _time_matches(self, fact_time: Optional[str], cq: ClassifiedQuestion) -> bool:
        """Check if fact time matches the question's time constraint."""
        if not fact_time or not cq.has_time_constraint:
            return True
        
        try:
            fact_year = int(str(fact_time).split("-")[0])
            constraint_year = int(cq.time_value.split("-")[0])
            
            if cq.time_semantic == TimeSemantic.EXACT:
                return fact_year == constraint_year
            elif cq.time_semantic == TimeSemantic.BEFORE:
                return fact_year <= constraint_year
            elif cq.time_semantic == TimeSemantic.AFTER:
                return fact_year >= constraint_year
            elif cq.time_semantic == TimeSemantic.BETWEEN:
                start, end = cq.time_value.split("-")
                return int(start) <= fact_year <= int(end)
        except (ValueError, TypeError, AttributeError):
            return False
        
        return True
    
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
    
    formatted = f"Knowledge Graph Facts (Strategy: {reasoning_strategy}):\n"
    for i, fact in enumerate(facts, 1):
        subject = fact.get("subject", "?")
        predicate = fact.get("predicate", "?")
        obj = fact.get("object", "?")
        time = fact.get("time", "unknown")
        
        formatted += f"{i}. {subject} | {predicate} = {obj} (as of {time})\n"
    
    return formatted

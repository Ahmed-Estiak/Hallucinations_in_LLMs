"""
KG Retriever: Fetch relevant facts from knowledge graph based on parsed question
"""
import json
from typing import List, Dict, Optional
from pathlib import Path
from src.time_utils import TIME_RANGE_SEPARATOR, fact_matches_time, select_best_temporal_fact


class KGRetriever:
    """
    Retrieves relevant knowledge graph facts for a parsed question.
    Supports entity matching, predicate matching, and time-based ranking.
    """
    
    def __init__(self, kg_path: str = "data/astronomy_kg1.json"):
        """Load knowledge graph from JSON file."""
        with open(kg_path, "r", encoding="utf-8") as f:
            self.kg = json.load(f)
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity name for matching (case-insensitive, etc.)."""
        return entity.strip().lower()
    
    def _normalize_predicate(self, pred: str) -> str:
        """Normalize predicate name."""
        return pred.strip().lower()
    
    def retrieve(
        self,
        entities: List[str],
        predicates: List[str],
        time_constraint: Optional[str] = None,
        time_semantic: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant facts from KG with deterministic temporal selection.
        
        TIME POLICY:
        - If no time constraint: return most recent fact per entity-predicate pair
        - EXACT/as-of: return latest fact at or before the constraint
        - BEFORE: return latest fact strictly before the constraint
        - AFTER: return earliest fact strictly after the constraint
        - BETWEEN: return latest fact fully inside the range
        
        Args:
            entities: List of entity names to search for
            predicates: List of predicates to search for
            time_constraint: Optional time constraint (e.g., "2022-08", "2022") - ONLY facts matching this
            limit: Max number of facts to return (default 3 for conciseness)
        
        Returns:
            List of highly filtered relevant facts ranked by relevance
        """
        grouped_candidates: dict[tuple[str, str], List[Dict]] = {}
        
        # Normalize inputs
        entities_norm = [self._normalize_entity(e) for e in entities]
        predicates_norm = [self._normalize_predicate(p) for p in predicates]
        
        # Iterate through KG and find matches
        for fact in self.kg:
            subject = self._normalize_entity(fact.get("subject", ""))
            predicate = self._normalize_predicate(fact.get("predicate", ""))
            fact_time = fact.get("time")
            
            entity_match = subject in entities_norm
            pred_match = predicate in predicates_norm
            
            # Prefer precise matches. If both sides exist in the query, require both.
            if entities_norm and predicates_norm:
                is_candidate = entity_match and pred_match
            elif entities_norm:
                is_candidate = entity_match
            elif predicates_norm:
                is_candidate = pred_match
            else:
                is_candidate = False

            if is_candidate:
                if time_constraint and not fact_matches_time(fact_time, time_constraint, time_semantic):
                    continue
                key = (fact.get("subject", ""), fact.get("predicate", ""))
                grouped_candidates.setdefault(key, []).append(fact)

        selected_facts: List[Dict] = []
        for facts in grouped_candidates.values():
            best_fact = select_best_temporal_fact(facts, time_constraint, time_semantic)
            if best_fact is not None:
                selected_facts.append(best_fact)

        selected_facts.sort(
            key=lambda fact: (
                1 if self._normalize_entity(fact.get("subject", "")) in entities_norm else 0,
                1 if self._normalize_predicate(fact.get("predicate", "")) in predicates_norm else 0,
                fact.get("time") or "",
            ),
            reverse=True,
        )

        return selected_facts[:limit]
    
    def format_facts_for_prompt(self, facts: List[Dict]) -> str:
        """
        Format retrieved facts as a readable string for LLM prompt.
        
        Returns:
            Formatted string with all facts
        """
        if not facts:
            return "No relevant facts found in the knowledge graph."
        
        formatted = "Knowledge Graph Facts:\n"
        for i, fact in enumerate(facts, 1):
            subject = fact.get("subject", "?")
            predicate = fact.get("predicate", "?")
            obj = fact.get("object", "?")
            time = fact.get("time", "unknown time")
            source = fact.get("source", "unknown source")
            
            formatted += f"{i}. {subject} | {predicate} = {obj} (as of {time})\n"
            formatted += f"   Source: {source}\n"
        
        return formatted


# Test
if __name__ == "__main__":
    retriever = KGRetriever()
    
    # Test case 1: Jupiter moons as of 2022
    facts = retriever.retrieve(
        entities=["Jupiter"],
        predicates=["moon_count"],
        time_constraint="2022-08"
    )
    print("Test 1: Jupiter moon_count as of 2022-08")
    print(retriever.format_facts_for_prompt(facts))
    print()
    
    # Test case 2: Neptune discovery
    facts = retriever.retrieve(
        entities=["Neptune"],
        predicates=["discovered_on", "discovered_by"],
        time_constraint="1846"
    )
    print("Test 2: Neptune discovery info")
    print(retriever.format_facts_for_prompt(facts))
    print()
    
    # Test case 3: Uranus moons
    facts = retriever.retrieve(
        entities=["Uranus"],
        predicates=["moon_count"],
        time_constraint="2025-02"
    )
    print("Test 3: Uranus moon_count as of 2025-02")
    print(retriever.format_facts_for_prompt(facts))

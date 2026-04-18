"""
KG Retriever: Fetch relevant facts from knowledge graph based on parsed question
"""
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

TIME_RANGE_SEPARATOR = ".."


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
    
    def _time_relevance_score(self, fact_time: Optional[str], constraint_time: Optional[str]) -> float:
        """
        Score how relevant a fact is based on time constraint.
        Returns score between 0 and 1.
        
        Rules:
        - Exact match: 1.0
        - Close match (within 1 year): 0.9
        - Same year: 0.8
        - Closest to constraint: 0.7
        - No constraint or no time in fact: 0.5
        """
        if not constraint_time or not fact_time:
            return 0.5
        
        try:
            constraint_key = self._parse_time_key(constraint_time)
            fact_key = self._parse_time_key(fact_time)
            
            if constraint_key is None or fact_key is None:
                return 0.5

            constraint_year, constraint_month, constraint_day = constraint_key
            fact_year, fact_month, fact_day = fact_key

            if constraint_month is not None and fact_month is not None:
                if (
                    constraint_year == fact_year and
                    constraint_month == fact_month and
                    (constraint_day is None or fact_day == constraint_day)
                ):
                    return 1.0
                month_diff = abs((constraint_year - fact_year) * 12 + (constraint_month - fact_month))
                if month_diff == 1:
                    return 0.95
                if month_diff <= 12:
                    return 0.8
                diff = abs(constraint_year - fact_year)
                return max(0.5, 1.0 - (diff * 0.02))

            diff = abs(constraint_year - fact_year)
            
            if diff == 0:
                return 1.0
            elif diff == 1:
                return 0.9
            elif diff <= 2:
                return 0.8
            else:
                return max(0.5, 1.0 - (diff * 0.02))  # Decay with distance
        except:
            return 0.5

    def _parse_time_key(self, value: Optional[str]) -> tuple[Optional[int], Optional[int], Optional[int]] | None:
        """Parse YYYY / YYYY-MM / YYYY-MM-DD into comparable parts."""
        if value is None:
            return None

        text = str(value).strip()
        if len(text) == 4 and text.isdigit():
            return int(text), None, None

        if re.match(r"^\d{4}-\d{2}$", text):
            year, month = text.split("-")
            return int(year), int(month), None

        if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            year, month, day = text.split("-")
            return int(year), int(month), int(day)

        return None
    
    def retrieve(
        self,
        entities: List[str],
        predicates: List[str],
        time_constraint: Optional[str] = None,
        time_semantic: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant facts from KG with AGGRESSIVE TIME FILTERING.
        
        TIME-CRITICAL FILTERING:
        - If time_constraint specified, ONLY return facts matching that time (reject mismatches)
        - If no time_constraint, return only the MOST RECENT facts per entity-predicate pair
        
        Args:
            entities: List of entity names to search for
            predicates: List of predicates to search for
            time_constraint: Optional time constraint (e.g., "2022-08", "2022") - ONLY facts matching this
            limit: Max number of facts to return (default 3 for conciseness)
        
        Returns:
            List of highly filtered relevant facts ranked by relevance
        """
        candidates = []
        
        # Normalize inputs
        entities_norm = [self._normalize_entity(e) for e in entities]
        predicates_norm = [self._normalize_predicate(p) for p in predicates]
        
        # Iterate through KG and find matches
        for fact in self.kg:
            subject = self._normalize_entity(fact.get("subject", ""))
            predicate = self._normalize_predicate(fact.get("predicate", ""))
            fact_time = fact.get("time")
            
            # Check if subject matches
            entity_match = subject in entities_norm
            
            # Check if predicate matches
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
                semantic_match = True
                if time_constraint:
                    semantic_match = self._time_matches_semantic(fact_time, time_constraint, time_semantic)
                if time_constraint and not semantic_match:
                    continue

                if time_constraint and (time_semantic or "").upper() in {"BEFORE", "AFTER", "BETWEEN"}:
                    time_score = 1.0
                else:
                    time_score = self._time_relevance_score(fact_time, time_constraint)
                
                # CRITICAL TIME FILTERING: If time constraint exists and fact doesn't match closely, REJECT it
                if time_constraint and time_score < 0.85:  # Only very close matches allowed
                    continue
                
                # Combined score: prioritize exact entity+predicate matches with time relevance
                if entity_match and pred_match:
                    combined_score = 1.0 * time_score  # Exact match
                elif entity_match:
                    combined_score = 0.9 * time_score
                else:
                    combined_score = 0.7 * time_score
                
                candidates.append({
                    "fact": fact,
                    "score": combined_score,
                    "entity_match": entity_match,
                    "pred_match": pred_match,
                    "time_score": time_score
                })
        
        # If no time_constraint specified, filter to LATEST fact per entity-predicate pair
        if not time_constraint and candidates:
            candidates = self._filter_to_latest_facts(candidates)
        
        # Sort by score descending and return top N
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return [c["fact"] for c in candidates[:limit]]

    def _time_matches_semantic(
        self,
        fact_time: Optional[str],
        constraint_time: Optional[str],
        time_semantic: Optional[str],
    ) -> bool:
        """Apply exact/before/after filtering when time semantics are known."""
        if not constraint_time or not fact_time:
            return True

        if TIME_RANGE_SEPARATOR in constraint_time:
            start_text, end_text = constraint_time.split(TIME_RANGE_SEPARATOR, 1)
            start_key = self._parse_time_key(start_text)
            end_key = self._parse_time_key(end_text)
            fact_key = self._parse_time_key(fact_time)
            if start_key is None or end_key is None or fact_key is None:
                return True
            return start_key <= fact_key <= end_key

        fact_key = self._parse_time_key(fact_time)
        constraint_key = self._parse_time_key(constraint_time)
        if fact_key is None or constraint_key is None:
            return True

        semantic = (time_semantic or "EXACT").upper()
        fact_year, fact_month, fact_day = fact_key
        constraint_year, constraint_month, constraint_day = constraint_key
        fact_cmp = (fact_year, fact_month or 0, fact_day or 0)
        constraint_cmp = (constraint_year, constraint_month or 0, constraint_day or 0)

        if semantic == "BEFORE":
            return fact_cmp < constraint_cmp
        if semantic == "AFTER":
            return fact_cmp >= constraint_cmp
        if semantic == "BETWEEN":
            return fact_cmp == constraint_cmp
        if constraint_month is not None:
            if constraint_day is not None:
                return fact_cmp == constraint_cmp
            return fact_year == constraint_year and fact_month == constraint_month
        return fact_year == constraint_year
    
    def _filter_to_latest_facts(self, candidates: List[Dict]) -> List[Dict]:
        """Keep only the most recent fact for each unique entity-predicate pair."""
        grouped = {}
        
        for candidate in candidates:
            fact = candidate["fact"]
            key = (fact.get("subject"), fact.get("predicate"))
            
            # Keep fact with highest time score (most recent)
            if key not in grouped or candidate["time_score"] > grouped[key]["time_score"]:
                grouped[key] = candidate
        
        return list(grouped.values())
    
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

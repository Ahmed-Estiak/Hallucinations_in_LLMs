"""
Question Parser: Extract entities, predicates, and time constraints from questions
"""
import re
from typing import Dict, List, Tuple, Optional


# Predicate synonyms and keyword mappings
PREDICATE_KEYWORDS = {
    "moon_count": {
        "keywords": ["moon", "moons", "satellite", "satellites", "orbital bodies"],
        "pattern": r"(?:how many|count of|number of)\s+(?:confirmed\s+)?moons"
    },
    "discovered_on": {
        "keywords": ["discovered", "discovery", "when discovered"],
        "pattern": r"(?:when|what year|which year).*(?:discover|discover)"
    },
    "discovered_by": {
        "keywords": ["discovered by", "who discovered", "discoverer"],
        "pattern": r"(?:who|which)\s+(?:astronomer\s+)?discover"
    },
    "mass": {
        "keywords": ["mass", "weight", "massive"],
        "pattern": r"(?:mass|weight)"
    },
    "distance_from_sun": {
        "keywords": ["distance", "away from sun", "far from sun"],
        "pattern": r"(?:distance|far|away).*(?:sun|solar)"
    },
    "surface_gravity": {
        "keywords": ["gravity", "gravitational", "surface gravity"],
        "pattern": r"(?:surface\s+)?gravity"
    },
    "classification": {
        "keywords": ["planet", "dwarf planet", "classification", "classified"],
        "pattern": r"(?:classify|classification|is\s+.{0,20}planet)"
    },
    "planet_type": {
        "keywords": ["type of planet", "terrestrial", "ice giant", "gas giant"],
        "pattern": r"(?:type|kind).*planet"
    },
    "location": {
        "keywords": ["location", "located", "belt", "where"],
        "pattern": r"(?:location|located|where)"
    }
}

# Entity name patterns (planets and dwarf planets)
KNOWN_ENTITIES = {
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Pluto", "Ceres", "Eris", "Makemake", "Haumea",
    "Moon", "Charon", "Triton", "Europa", "Ganymede", "Callisto", "Io",
    "Johann Galle", "Clyde Tombaugh", "Johann Gottfried Galle"
}


def extract_entities(question: str) -> List[str]:
    """
    Extract entity names from question using keyword matching.
    Returns list of entity names found.
    """
    entities = []
    question_lower = question.lower()
    
    for entity in KNOWN_ENTITIES:
        if entity.lower() in question_lower:
            entities.append(entity)
    
    return list(set(entities))  # Remove duplicates


def extract_time_constraint(question: str) -> Optional[str]:
    """
    Extract time constraint from question.
    Returns time string (e.g., "2022-08", "2024-01", year like "1981") or None.
    """
    # Pattern: "As of [month year]" or "By [month year]" or "Before [year]"
    patterns = [
        r"(?:as of|by)\s+([a-z]+\s+\d{4})",  # "As of January 2022"
        r"(?:as of|by)\s+(\d{4})",             # "As of 2022"
        r"in\s+(\d{4})",                       # "In 1981"
        r"(?:before|prior to)\s+(\d{4})",     # "Before 1980"
        r"as\s+of\s+([a-z]+\s+\d{4})"          # "as of February 2025"
    ]
    
    question_lower = question.lower()
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            time_str = match.group(1)
            # Normalize to YYYY-MM format if possible
            if len(time_str) == 4 and time_str.isdigit():
                return time_str  # Just year
            return time_str
    
    return None


def infer_predicates(question: str) -> List[str]:
    """
    Infer which predicates the question is asking about.
    Returns list of predicate names.
    """
    predicates = []
    question_lower = question.lower()
    
    for predicate, info in PREDICATE_KEYWORDS.items():
        # Check pattern first for higher confidence
        if re.search(info["pattern"], question_lower):
            predicates.append(predicate)
        else:
            # Fall back to keyword matching
            for keyword in info["keywords"]:
                if keyword.lower() in question_lower:
                    predicates.append(predicate)
                    break
    
    return list(set(predicates))  # Remove duplicates


def parse_question(question: str) -> Dict[str, any]:
    """
    Parse a question and extract structured information.
    
    Returns:
        {
            "question": str,
            "entities": List[str],
            "predicates": List[str],
            "time_constraint": Optional[str]
        }
    """
    return {
        "question": question,
        "entities": extract_entities(question),
        "predicates": infer_predicates(question),
        "time_constraint": extract_time_constraint(question)
    }


# Test
if __name__ == "__main__":
    test_questions = [
        "As of November 2022, how many confirmed moons orbit Jupiter?",
        "In what year was Neptune discovered, and who discovered it?",
        "As of February 2025, what is the total number of confirmed moons of Uranus?",
        "Which dwarf planet located in the Kuiper Belt was discovered first?",
    ]
    
    for q in test_questions:
        result = parse_question(q)
        print(f"Q: {q}")
        print(f"  Entities: {result['entities']}")
        print(f"  Predicates: {result['predicates']}")
        print(f"  Time: {result['time_constraint']}")
        print()

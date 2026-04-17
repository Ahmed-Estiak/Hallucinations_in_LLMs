"""
Question Parser: Extract entities, predicates, and time constraints from questions
"""
import re
from typing import Dict, List, Tuple, Optional


MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

MOON_LIKE_BODY_PATTERN = r"(?:moons|satellites|orbital bodies)"


# Predicate synonyms and keyword mappings with COMPLEX question patterns
PREDICATE_KEYWORDS = {
    "moon_count": {
        "keywords": [],
        "pattern": rf"(?:how many|count of|number of|total number of)\s+(?:officially\s+)?(?:confirmed\s+)?(?:known\s+)?{MOON_LIKE_BODY_PATTERN}|how many\s+(?:officially\s+)?(?:confirmed\s+)?(?:known\s+)?{MOON_LIKE_BODY_PATTERN}\s+.*(?:did|does)\s+.*have"
    },
    "discovered_on": {
        "keywords": ["discovered", "discovery", "when discovered", "found", "when found", "discovery year", "found in"],
        "pattern": r"(?:when|what year|which year|in what year)\s+(?:was\s+)?(?:.*\s+)?(?:discovered|found)|\b(?:discovered|found)\b\s+.*\bwhen\b|\bdiscovery year\b"
    },
    "discovered_by": {
        "keywords": ["discovered by", "who discovered", "discoverer", "found by", "who found", "finder"],
        "pattern": r"(?:who|which)\s+(?:astronomer\s+)?(?:discovered|found)|(?:discovered|found)\s+by|who\s+was\s+.*(?:discovered|found)\s+by|name\s+the\s+(?:astronomer|discoverer|finder)"
    },
    "mass": {
        "keywords": ["mass", "weight", "massive", "heavier", "lighter", "heaviest", "lightest"],
        "pattern": r"(?:what\s+is\s+the\s+mass|mass\s+of|greater\s+mass|more\s+massive|less\s+massive|heaviest|lightest|heavier|lighter)"
    },
    "distance_from_sun": {
        "keywords": ["distance from sun", "distance to sun", "away from sun", "far from sun", "farther from sun", "closer to sun"],
        "pattern": r"(?:distance\s+(?:from|to)\s+the?\s*sun|distance\s+from\s+sun|far(?:ther)?\s+from\s+the?\s*sun|closer\s+to\s+the?\s*sun|which\s+.*(?:farther\s+from|closer\s+to)\s+the?\s*sun)"
    },
    "surface_gravity": {
        "keywords": ["surface gravity", "gravitational pull", "gravity of", "stronger gravity", "weaker gravity", "higher gravity", "lower gravity", "stronger surface gravity", "weaker surface gravity", "higher surface gravity", "lower surface gravity"],
        "pattern": r"(?:surface\s+gravity|gravity\s+of|gravitational\s+pull|stronger\s+(?:surface\s+)?gravity|weaker\s+(?:surface\s+)?gravity|higher\s+(?:surface\s+)?gravity|lower\s+(?:surface\s+)?gravity)"
    },
    "classification": {
        "keywords": ["classification", "classified", "recognized as", "classify as"],
        "pattern": r"(?:classification\s+of|how\s+is\s+.*classified|classif(?:y|ied)\s+as|is\s+.*(?:a|an)\s+(?:dwarf\s+)?planet|recogniz(?:e|ed)\s+.*(?:as\s+)?(?:a\s+)?(?:dwarf\s+)?planet|recognized\s+as)"
    },
    "planet_type": {
        "keywords": ["type of planet", "terrestrial", "ice giant", "gas giant"],
        "pattern": r"(?:terrestrial|gas giant|ice giant|type.*planet)"
    },
    "location": {
        "keywords": ["location", "located", "belt", "where", "orbit"],
        "pattern": r"(?:location|located|where.*found|kuiper belt|asteroid belt|orbit)"
    },
    "ordering": {
        "keywords": ["order", "list", "decreasing", "increasing", "smallest", "largest"],
        "pattern": r"(?:order.*decreasing|order.*increasing|list.*order|rank|smallest|largest|greater than|less than)"
    },
    "comparison": {
        "keywords": ["between", "compared to", "than", "more", "less"],
        "pattern": r"(?:between.*which|compared|greater|more|fewer|less)"
    }
}

# Entity name patterns (planets, dwarf planets, AND other astronomical features)
KNOWN_ENTITIES = {
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Pluto", "Ceres", "Eris", "Makemake", "Haumea",
    "Moon", "Charon", "Triton", "Europa", "Ganymede", "Callisto", "Io",
    "Johann Galle", "Clyde Tombaugh", "Johann Gottfried Galle",
    "Kuiper Belt", "Asteroid Belt", "Solar System", "Oort Cloud"  # Astronomical regions
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
        r"(?:as of|by)\s+([a-z]+\s+\d{4})",   # "As of January 2022"
        r"(?:as of|by)\s+(\d{4})",            # "As of 2022"
        r"in\s+(\d{4})",                      # "In 1981"
        r"(?:before|prior to)\s+(\d{4})",     # "Before 1980"
        r"as\s+of\s+([a-z]+\s+\d{4})"         # "as of February 2025"
    ]
    
    question_lower = question.lower()
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            time_str = match.group(1)
            month_year_match = re.fullmatch(r"([a-z]+)\s+(\d{4})", time_str)
            if month_year_match:
                month_name, year = month_year_match.groups()
                month = MONTH_MAP.get(month_name)
                if month:
                    return f"{year}-{month}"

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

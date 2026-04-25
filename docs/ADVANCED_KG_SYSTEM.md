# Advanced KG Reasoning System - Implementation Guide

## Overview
আপনার KG System এখন একটি sophisticated reasoning engine যা প্রশ্নের ধরন বুঝে সেই অনুযায়ী আলগোরিদম চালায়।

---

## System Architecture

### 1. **Question Classifier** (`question_classifier.py`)
প্রশ্নকে বিশ্লেষণ করে multiple dimensions-এ classify করে:

#### Question Types (Priority Order)
1. **BOOLEAN** (সর্বোচ্চ প্রায়োরিটি)
   - "Is Pluto a dwarf planet?"
   - KG Strategy: Direct fact matching

2. **ENTITY**
   - "Which planet was discovered first?"
   - KG Strategy: Direct entity lookup

3. **ENTITY_LIST**
   - "Which planets have fewer moons than Jupiter?"
   - KG Strategy: Filtering with AND/OR conditions

4. **ORDERED_LIST**
   - "List planets in decreasing mass"
   - KG Strategy: Sort by attribute (computational)

5. **COMPARISON**
   - "Which is farther, Neptune or Uranus?"
   - KG Strategy: Retrieve values, compare numerically

6. **COUNT**
   - "How many moons does Jupiter have?"
   - KG Strategy: Aggregation

7. **TIME_LOOKUP**
   - "In what year was Neptune discovered?"
   - KG Strategy: Time-based fact retrieval

8. **MULTI_FIELD**
   - "Year AND discoverer of Neptune?"
   - KG Strategy: Combine multiple predicates

#### Time Semantics
- **EXACT**: "As of 2022" = সেই বছরের ডাটা
- **BEFORE**: "Before 1980" = কোনো সময় ≤ 1980
- **AFTER**: "After 2000" = কোনো সময় ≥ 2000
- **BETWEEN**: Range-based

---

## 2. **KG Reasoning Engine** (`kg_reasoning_engine.py`)
প্রতিটি question type এর জন্য specialized reasoning:

### Reasoning Strategies

#### Boolean Reasoning
```
প্রশ্ন: "Is Pluto dwarf planet with largest diameter?"
KG Strategy:
  1. Find facts with subject="Pluto", predicate="classification"
  2. Filter by time constraint (if any)
  3. Return direct match
```

#### Entity List & Filtering
```
প্রশ্ন: "Which planets have fewer moons than Jupiter?"
KG Strategy:
  1. Get Jupiter's moon_count from KG
  2. Get all planets' moon_count
  3. Filter: planets with moon_count < Jupiter.moon_count
  4. Use AND/OR logic from question
```

#### Ordered List (Computational)
```
প্রশ্ন: "List terrestrial planets by decreasing mass"
KG Strategy:
  1. Retrieve mass facts for all planets
  2. Extract numeric values
  3. Sort by mass (descending)
  4. Return ordered list
```

#### Comparison
```
প্রশ্ন: "Between Neptune and Uranus, which is farther?"
KG Strategy:
  1. Get distance_from_sun for both
  2. Compare numeric values
  3. Return entity with greater/lesser value
```

#### Count with Time
```
প্রশ্ন: "As of Nov 2022, how many moons orbit Jupiter?"
KG Strategy:
  1. Get Jupiter moon_count facts
  2. Filter: fact.time matches November 2022
  3. Return count
```

#### Multi-field Combination
```
প্রশ্ন: "Year discovered AND who discovered Neptune?"
KG Strategy:
  1. Get Neptune.discovered_on facts
  2. Get Neptune.discovered_by facts
  3. Combine: "1846, Johann Galle"
```

---

## 3. **Time Awareness**

### Critical Time Logic
- যদি প্রশ্নে নির্দিষ্ট সময় থাকে: কেবল সেই সময়ের fact ব্যবহার করো
- যদি সময় না থাকে: সর্বশেষ fact ব্যবহার করো
- "Before 1980" = পূর্ববর্তী সময়ের data খুঁজো

### উদাহরণ
- "Before 1980 Saturn had X moons" → খোঁজ 1980-এর আগের KG entry
- "As of 2022 Jupiter had X moons" → সমান বা পূর্ববর্তী 2022 ডাটা
- "Jupiter moons" (কোনো সময় নেই) → সর্বশেষ ডাটা নাও

---

## 4. **Multi-field Questions**

### Detection Patterns
- "Year AND Who" → multi-field with AND
- "Year OR Classification" → multi-field with OR
- "Between X and Y, which" → comparison + list

### Processing
```
প্রশ্ন: "In what year was Neptune discovered, and who discovered it?"

1. Detect: multi-field=TRUE, predicates=[discovered_on, discovered_by]
2. Get all Neptune facts with these predicates
3. Combine: "1846, Johann Galle"
4. Format for LLM
```

---

## 5. **Boolean Priority**

Boolean questions **সর্বদা প্রথম priority** পায় কারণ:
- সরাসরি সত্য/মিথ্যা উত্তর
- কোনো computation/sorting দরকার নেই
- দ্রুত এবং নির্ভুল

---

## 6. **Logic Operators (AND/OR)**

### Implementation
```
AND Logic:
  Entity ∈ [planets with moons > 10] AND 
  Entity ∈ [planets discovered before 1900]

OR Logic:
  Result = Entity_set_1 ∪ Entity_set_2
```

---

## 7. **Local NLP Tools (No Token Cost)**

আপনার system-এ ব্যবহার করার জন্য free tools:

### Already Integrated
- ✅ Regex patterns (no tools needed)
- ✅ Rule-based classification

### Can Add (Optional, free)
- spaCy: POS tagging, NER
- NLTK: Tokenization
- sentence-transformers: Semantic similarity

---

## Example: Complete Flow

```
প্রশ্ন: "Before 1980, how many of Saturn's moons had been officially confirmed?"

Step 1 [Classifier]:
  - Detect: primary_type = COUNT
  - secondary = TIME_LOOKUP
  - time_semantic = BEFORE
  - time_value = 1980

Step 2 [Question Parser]:
  - entities = ["Saturn", "Moon"]
  - predicates = ["moon_count"]
  - time_constraint = "before 1980"

Step 3 [Reasoning Engine]:
  - Strategy: count + time_filtering (BEFORE semantics)
  - Search KG: Saturn.moon_count facts where time < 1980
  - Find: Saturn had 11 moons as of 1979
  
  Actually KG had: 17 (এটা correct, but need older entry)
  Problem: KG শুধু recent data আছে
  
  Solution: Fallback to LLM with KG guidance

Step 4 [LLM Getting]:
  facts = "Saturn moon_count (1979): 11"
  time_constraint = "before 1980"
  
  LLM: "According to KG, 11 moons before 1980"
```

---

## Next Steps to Integrate

1. **Update kg_runner.py** to use new classifier and reasoning engine
2. **Update kg_models.py** to accept reasoning strategy metadata
3. **Test** with all 8 question types
4. **Evaluate** accuracy improvements

---

## Key Advantages

✅ **Type-specific reasoning** - প্রতিটি প্রশ্ন সঠিকভাবে হ্যান্ডেল হয়
✅ **Time-aware** - historical queries সঠিকভাবে filter
✅ **Multi-field support** - complex questions tackle করতে পারে
✅ **Computational** - ordering, comparison programming দ্বারা হয়
✅ **Priority-based** - boolean প্রথমে, fallback systematic
✅ **No token cost** - সব NLP locally হয়

---

## Metrics to Track

- Type detection accuracy
- Time filtering correctness
- Multi-field extraction accuracy
- Reasoning strategy usage frequency
- Final answer correctness per type

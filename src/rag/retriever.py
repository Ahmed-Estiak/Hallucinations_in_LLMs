"""Lexical retriever for the first RAG+LLM benchmark slice."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.question_classifier import LogicalModifier, QuestionClassifier
from src.question_parser import parse_question
from src.rag.embeddings import DEFAULT_EMBEDDINGS_PATH, EmbeddingIndex, cosine_similarity
from src.rag.retriever_terms import build_query_terms, infer_query_predicates, tokenize
from src.rag.source_selector import SourceScore, SourceSelection, SourceSelector, append_major_planet_sources


RETRIEVAL_MODES = {"global", "auto-source", "vector", "hybrid"}
HYBRID_SOURCE_LEXICAL_WEIGHT = 0.60
HYBRID_SOURCE_VECTOR_WEIGHT = 0.40
HYBRID_CHUNK_LEXICAL_WEIGHT = 0.50
HYBRID_CHUNK_VECTOR_WEIGHT = 0.50


@dataclass
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.chunk)
        data["score"] = round(self.score, 4)
        data["score_reasons"] = self.reasons
        return data


@dataclass
class RagRetrievalResult:
    retrieved_chunks: list[RetrievedChunk]
    retrieval_mode: str
    source_selection: SourceSelection | None = None
    fallback_used: bool = False
    fallback_reason: str = ""


class RagRetriever:
    def __init__(
        self,
        chunks_path: str | Path = "data/rag_sources/rag_index/chunks.jsonl",
        documents_path: str | Path = "data/rag_sources/rag_index/documents.jsonl",
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks(self.chunks_path)
        self.documents_path = Path(documents_path)
        self.embeddings_path = Path(embeddings_path)
        self._embedding_index: EmbeddingIndex | None = None
        self.question_classifier = QuestionClassifier()
        self.source_selector = SourceSelector(self.chunks, documents_path=self.documents_path)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 12,
        per_source_limit: int = 4,
        mode: str = "global",
        top_n_sources: int = 5,
    ) -> list[RetrievedChunk]:
        return self.retrieve_with_details(
            question,
            top_k=top_k,
            per_source_limit=per_source_limit,
            mode=mode,
            top_n_sources=top_n_sources,
        ).retrieved_chunks

    def retrieve_with_details(
        self,
        question: str,
        *,
        top_k: int = 12,
        per_source_limit: int = 4,
        mode: str = "global",
        top_n_sources: int = 5,
    ) -> RagRetrievalResult:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(RETRIEVAL_MODES))}")

        source_selection = None
        selected_source_ids = None
        fallback_used = False
        fallback_reason = ""
        query_embedding = None
        vector_scores = None
        if mode == "auto-source":
            source_selection = self.source_selector.select(question, top_n_sources=top_n_sources)
            selected_source_ids = set(source_selection.selected_source_ids)
            if not selected_source_ids:
                fallback_used = True
                fallback_reason = "no_sources_selected"
                selected_source_ids = None
        elif mode in {"vector", "hybrid"}:
            query_embedding = self.embedding_index.embed_query(question)
            vector_scores = self._vector_scores_by_chunk(query_embedding)
            if mode == "vector":
                source_selection = self._select_sources_vector(
                    question,
                    vector_scores,
                    top_n_sources=top_n_sources,
                )
            else:
                source_selection = self._select_sources_hybrid(
                    question,
                    vector_scores,
                    top_n_sources=top_n_sources,
                )
            selected_source_ids = set(source_selection.selected_source_ids)
            if not selected_source_ids:
                fallback_used = True
                fallback_reason = "no_sources_selected"
                selected_source_ids = None

        effective_per_source_limit = self._effective_per_source_limit(question, per_source_limit)
        candidate_chunks = [
            chunk for chunk in self.chunks
            if selected_source_ids is None or chunk.get("source_id") in selected_source_ids
        ]
        if mode == "vector":
            retrieved = self._retrieve_from_chunks_vector(
                candidate_chunks,
                vector_scores=vector_scores or {},
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
            )
        elif mode == "hybrid":
            retrieved = self._retrieve_from_chunks_hybrid(
                question,
                chunks=candidate_chunks,
                vector_scores=vector_scores or {},
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
            )
        else:
            retrieved = self._retrieve_from_chunks(
                question,
                chunks=candidate_chunks,
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
            )

        if mode == "auto-source" and is_weak_retrieval(retrieved):
            fallback_used = True
            fallback_reason = "weak_auto_source_chunks"
            retrieved = self._retrieve_from_chunks(
                question,
                chunks=self.chunks,
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
            )
        if mode in {"vector", "hybrid"} and is_weak_retrieval(retrieved):
            fallback_used = True
            fallback_reason = "weak_vector_chunks"
            if mode == "vector":
                retrieved = self._retrieve_from_chunks_vector(
                    self.chunks,
                    vector_scores=vector_scores or {},
                    top_k=top_k,
                    per_source_limit=effective_per_source_limit,
                )
            else:
                retrieved = self._retrieve_from_chunks_hybrid(
                    question,
                    chunks=self.chunks,
                    vector_scores=vector_scores or {},
                    top_k=top_k,
                    per_source_limit=effective_per_source_limit,
                )

        if source_selection and source_selection.fallback_used:
            fallback_used = True
            fallback_reason = fallback_reason or source_selection.fallback_reason

        return RagRetrievalResult(
            retrieved_chunks=retrieved,
            retrieval_mode=mode,
            source_selection=source_selection,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def _effective_per_source_limit(self, question: str, per_source_limit: int) -> int:
        classified = self.question_classifier.classify(question)
        if str(classified.primary_type.name) == "LIST":
            return min(per_source_limit, 2)
        return per_source_limit

    def _retrieve_from_chunks(
        self,
        question: str,
        *,
        chunks: list[dict[str, Any]],
        top_k: int,
        per_source_limit: int,
    ) -> list[RetrievedChunk]:
        parsed = parse_question(question)
        classified = self.question_classifier.classify(question)
        query_terms = build_query_terms(question)
        entity_terms = [entity.lower() for entity in parsed["entities"] + classified.major_entities]
        predicate_terms = list(dict.fromkeys(
            parsed["predicates"] + classified.major_predicates + infer_query_predicates(question)
        ))
        time_constraints = extract_time_constraints(question)
        target_entity_class = classified.target_entity_class or classified.list_target

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            score, reasons = self._score_chunk(
                chunk,
                query_terms=query_terms,
                entity_terms=entity_terms,
                predicate_terms=predicate_terms,
                time_constraints=time_constraints,
                target_entity_class=target_entity_class,
                has_filter=LogicalModifier.FILTER in classified.logical_modifiers,
                has_ordering=LogicalModifier.ORDERING in classified.logical_modifiers,
            )
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score, reasons=reasons))

        scored.sort(key=lambda item: item.score, reverse=True)
        return cap_per_source(scored, top_k=top_k, per_source_limit=per_source_limit)

    @property
    def embedding_index(self) -> EmbeddingIndex:
        if self._embedding_index is None:
            self._embedding_index = EmbeddingIndex(self.embeddings_path)
        return self._embedding_index

    def _vector_scores_by_chunk(self, query_embedding: list[float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for chunk in self.chunks:
            embedding = self.embedding_index.get(chunk["chunk_id"])
            if embedding is None:
                continue
            scores[chunk["chunk_id"]] = cosine_similarity(query_embedding, embedding)
        return scores

    def _select_sources_vector(
        self,
        question: str,
        vector_scores: dict[str, float],
        *,
        top_n_sources: int,
    ) -> SourceSelection:
        source_scores = self._aggregate_vector_source_scores(vector_scores)
        source_scores.sort(key=lambda item: item.score, reverse=True)
        selected = [score.source_id for score in source_scores[:top_n_sources]]
        return SourceSelection(
            mode="vector",
            selected_source_ids=selected,
            scores=source_scores,
        )

    def _select_sources_hybrid(
        self,
        question: str,
        vector_scores: dict[str, float],
        *,
        top_n_sources: int,
    ) -> SourceSelection:
        lexical_selection = self.source_selector.select(
            question,
            top_n_sources=max(top_n_sources, len(self.source_selector.profiles)),
            min_score=0.0,
        )
        lexical_scores = {score.source_id: score for score in lexical_selection.scores}
        vector_source_scores = {
            score.source_id: score
            for score in self._aggregate_vector_source_scores(vector_scores)
        }
        source_ids = set(lexical_scores) | set(vector_source_scores)
        hybrid_scores = []
        for source_id in source_ids:
            lexical_score = lexical_scores.get(source_id)
            vector_score = vector_source_scores.get(source_id)
            lexical_value = lexical_score.score if lexical_score else 0.0
            vector_value = vector_score.score if vector_score else 0.0
            score = (
                HYBRID_SOURCE_LEXICAL_WEIGHT * lexical_value
                + HYBRID_SOURCE_VECTOR_WEIGHT * vector_value
            )
            profile = self.source_selector.profiles.get(source_id)
            reasons = [
                f"hybrid_lexical:{lexical_value:.2f}",
                f"hybrid_vector:{vector_value:.2f}",
            ]
            if lexical_score:
                reasons.extend(lexical_score.reasons[:4])
            if vector_score:
                reasons.extend(vector_score.reasons[:2])
            hybrid_scores.append(SourceScore(
                source_id=source_id,
                score=score,
                reasons=reasons,
                title=(profile.title if profile else ""),
                url=(profile.url if profile else ""),
                chunk_count=(profile.chunk_count if profile else 0),
                char_count=(profile.char_count if profile else 0),
            ))

        hybrid_scores.sort(key=lambda item: item.score, reverse=True)
        selected = [score.source_id for score in hybrid_scores[:top_n_sources]]
        if re.search(r"\bwhich\s+planets\b|\blist\s+(?:the\s+)?planets\b", question.lower()):
            selected = append_major_planet_sources(selected, self.source_selector.profiles)
        return SourceSelection(
            mode="hybrid",
            selected_source_ids=selected,
            scores=hybrid_scores,
        )

    def _aggregate_vector_source_scores(self, vector_scores: dict[str, float]) -> list[SourceScore]:
        scores_by_source: dict[str, list[float]] = defaultdict(list)
        for chunk in self.chunks:
            score = vector_scores.get(chunk["chunk_id"])
            if score is not None:
                scores_by_source[chunk["source_id"]].append(score)

        source_scores = []
        for source_id, scores in scores_by_source.items():
            profile = self.source_selector.profiles.get(source_id)
            top_scores = sorted(scores, reverse=True)[:3]
            if not top_scores:
                continue
            top_average = sum(top_scores) / len(top_scores)
            top_score = top_scores[0]
            score = top_average * 100.0
            source_scores.append(SourceScore(
                source_id=source_id,
                score=score,
                reasons=[
                    f"vector_top_similarity:{top_score:.4f}",
                    f"vector_top3_avg:{top_average:.4f}",
                    f"vector_chunks:{len(scores)}",
                ],
                title=(profile.title if profile else ""),
                url=(profile.url if profile else ""),
                chunk_count=(profile.chunk_count if profile else len(scores)),
                char_count=(profile.char_count if profile else 0),
            ))
        return source_scores

    def _retrieve_from_chunks_vector(
        self,
        chunks: list[dict[str, Any]],
        *,
        vector_scores: dict[str, float],
        top_k: int,
        per_source_limit: int,
    ) -> list[RetrievedChunk]:
        scored = []
        for chunk in chunks:
            similarity = vector_scores.get(chunk["chunk_id"])
            if similarity is None:
                continue
            scored.append(RetrievedChunk(
                chunk=chunk,
                score=similarity * 100.0,
                reasons=[f"vector_similarity:{similarity:.4f}"],
            ))
        scored.sort(key=lambda item: item.score, reverse=True)
        return cap_per_source(scored, top_k=top_k, per_source_limit=per_source_limit)

    def _retrieve_from_chunks_hybrid(
        self,
        question: str,
        *,
        chunks: list[dict[str, Any]],
        vector_scores: dict[str, float],
        top_k: int,
        per_source_limit: int,
    ) -> list[RetrievedChunk]:
        parsed = parse_question(question)
        classified = self.question_classifier.classify(question)
        query_terms = build_query_terms(question)
        entity_terms = [entity.lower() for entity in parsed["entities"] + classified.major_entities]
        predicate_terms = list(dict.fromkeys(
            parsed["predicates"] + classified.major_predicates + infer_query_predicates(question)
        ))
        time_constraints = extract_time_constraints(question)
        target_entity_class = classified.target_entity_class or classified.list_target

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            lexical_score, lexical_reasons = self._score_chunk(
                chunk,
                query_terms=query_terms,
                entity_terms=entity_terms,
                predicate_terms=predicate_terms,
                time_constraints=time_constraints,
                target_entity_class=target_entity_class,
                has_filter=LogicalModifier.FILTER in classified.logical_modifiers,
                has_ordering=LogicalModifier.ORDERING in classified.logical_modifiers,
            )
            vector_similarity = vector_scores.get(chunk["chunk_id"], 0.0)
            vector_score = vector_similarity * 100.0
            score = (
                HYBRID_CHUNK_LEXICAL_WEIGHT * lexical_score
                + HYBRID_CHUNK_VECTOR_WEIGHT * vector_score
            )
            if score <= 0:
                continue
            reasons = [
                f"hybrid_lexical:{lexical_score:.2f}",
                f"hybrid_vector:{vector_score:.2f}",
                f"vector_similarity:{vector_similarity:.4f}",
            ]
            reasons.extend(lexical_reasons[:8])
            scored.append(RetrievedChunk(chunk=chunk, score=score, reasons=reasons))

        scored.sort(key=lambda item: item.score, reverse=True)
        return cap_per_source(scored, top_k=top_k, per_source_limit=per_source_limit)

    def format_context(self, retrieved_chunks: list[RetrievedChunk], *, max_chars: int = 12000) -> str:
        parts = []
        current_chars = 0
        for index, item in enumerate(retrieved_chunks, start=1):
            chunk = item.chunk
            block = (
                f"[R{index} | source_id={chunk['source_id']} | section={chunk.get('section', '')} | "
                f"score={item.score:.2f}]\n"
                f"URL: {chunk.get('url', '')}\n"
                f"{chunk['text']}\n"
            )
            if current_chars + len(block) > max_chars:
                break
            parts.append(block)
            current_chars += len(block)
        return "\n".join(parts).strip()

    def _score_chunk(
        self,
        chunk: dict[str, Any],
        *,
        query_terms: list[str],
        entity_terms: list[str],
        predicate_terms: list[str],
        time_constraints: dict[str, list[str]],
        target_entity_class: str | None,
        has_filter: bool,
        has_ordering: bool,
    ) -> tuple[float, list[str]]:
        text = " ".join([
            chunk.get("title", ""),
            chunk.get("section", ""),
            chunk.get("text", ""),
        ]).lower()
        tokens = Counter(tokenize(text))
        score = 0.0
        reasons: list[str] = []

        for term in query_terms:
            if " " in term:
                if has_phrase(text, term):
                    score += 3.0
                    reasons.append(f"phrase:{term}")
            elif tokens.get(term):
                score += 1.0 + math.log(tokens[term])

        for entity in set(entity_terms):
            if entity and entity in text:
                score += 5.0
                reasons.append(f"entity:{entity}")

        predicate_hints = set(chunk.get("predicate_hints", []))
        for predicate in predicate_terms:
            if predicate in predicate_hints:
                score += 3.0
                reasons.append(f"predicate:{predicate}")

        if has_filter and any(value in text for value in ("kuiper belt", "trans-neptunian", "located")):
            score += 2.5
            reasons.append("filter_context")
        if has_filter and any(value in text for value in ("fewer", "less than", "beyond earth", "orbit beyond", "moons")):
            score += 2.5
            reasons.append("comparative_filter_context")
        entity_match_present = any(entity and entity in text for entity in set(entity_terms))

        if "moon_count" in predicate_terms:
            moon_score, moon_reasons = score_moon_count_context(chunk, text)
            score += moon_score
            reasons.extend(moon_reasons)
        time_score, time_reasons = score_time_context(
            text,
            time_constraints,
            entity_match_present=entity_match_present,
        )
        score += time_score
        reasons.extend(time_reasons)
        if "distance_from_sun" in predicate_terms:
            distance_score, distance_reasons = score_orbit_order_context(chunk, text)
            score += distance_score
            reasons.extend(distance_reasons)
        if has_ordering and any(value in text for value in ("discovered", "discovery", "first observed")):
            score += 2.5
            reasons.append("ordering_context")
        if has_ordering and "in order of discovery" in text:
            score += 5.0
            reasons.append("ordered_discovery_section")
        if has_ordering and tokens.get("discovered", 0) >= 2:
            score += 2.0 + math.log(tokens["discovered"])
            reasons.append("multiple_discovery_mentions")
        dwarf_score, dwarf_reasons = score_dwarf_planet_context(
            text,
            target_entity_class=target_entity_class,
            query_terms=query_terms,
        )
        score += dwarf_score
        reasons.extend(dwarf_reasons)

        return score, reasons

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks


def cap_per_source(items: list[RetrievedChunk], *, top_k: int, per_source_limit: int) -> list[RetrievedChunk]:
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for item in items:
        source_id = item.chunk.get("source_id", "")
        if counts[source_id] >= per_source_limit:
            continue
        selected.append(item)
        counts[source_id] += 1
        if len(selected) >= top_k:
            break
    return selected


def is_weak_retrieval(items: list[RetrievedChunk]) -> bool:
    if len(items) < 3:
        return True
    if not items:
        return True
    return items[0].score < 8.0


def score_dwarf_planet_context(
    text: str,
    *,
    target_entity_class: str | None,
    query_terms: list[str],
) -> tuple[float, list[str]]:
    if not re.search(r"\b(?:dwarf|minor)\s+planets?\b", text):
        return 0.0, []

    score = 0.0
    reasons: list[str] = []
    if has_dwarf_planet_query_intent(query_terms):
        score += 2.0
        reasons.append("dwarf_planet_query_context")
    if target_entity_class == "dwarf_planets":
        score += 1.0
        reasons.append("target_class:dwarf_planets")
    return score, reasons


def has_dwarf_planet_query_intent(query_terms: list[str]) -> bool:
    terms = set(query_terms)
    return bool({"dwarf", "dwarf planet", "dwarf planets", "minor planet", "minor planets"} & terms)


def has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def extract_time_constraints(question: str) -> dict[str, list[str]]:
    question_lower = question.lower()
    years = re.findall(r"\b(?:18|19|20)\d{2}\b", question_lower)
    months = [month for month in MONTH_NAMES if re.search(rf"\b{month}\b", question_lower)]
    phrases = []
    for month in months:
        for year in years:
            phrase = f"{month} {year}"
            if phrase in question_lower:
                phrases.append(phrase)
    return {
        "phrases": list(dict.fromkeys(phrases)),
        "years": list(dict.fromkeys(years)),
        "months": list(dict.fromkeys(months)),
    }


def score_time_context(
    text: str,
    time_constraints: dict[str, list[str]],
    *,
    entity_match_present: bool,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    phrases = time_constraints.get("phrases", [])
    years = time_constraints.get("years", [])
    months = time_constraints.get("months", [])
    if not phrases and not years and not months:
        return score, reasons

    for phrase in phrases:
        if phrase in text:
            score += 14.0 if entity_match_present else 3.0
            reasons.append(f"exact_time:{phrase}" if entity_match_present else f"exact_time_without_entity:{phrase}")
    if not entity_match_present:
        return score, reasons
    for year in years:
        if re.search(rf"\b{re.escape(year)}\b", text):
            score += 4.0
            reasons.append(f"year:{year}")
    for month in months:
        if re.search(rf"\b{re.escape(month)}\b", text):
            score += 2.0
            reasons.append(f"month:{month}")
    if phrases and "as of" in text and any(phrase in text for phrase in phrases):
        score += 3.0
        reasons.append("as_of_time_context")
    return score, reasons


def score_moon_count_context(chunk: dict[str, Any], text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    section = str(chunk.get("section", "")).lower()
    title = str(chunk.get("title", "")).lower()

    if "moon" in section or "satellite" in section:
        score += 5.0
        reasons.append("moon_section")
    if any(word in text for word in ("known moons", "confirmed moons", "natural satellites", "confirmed satellites")):
        score += 5.0
        reasons.append("moon_count_terms")
    if re_moon_count_claim(text):
        score += 8.0
        reasons.append("moon_count_claim")
    if title_is_planet(title) and any(word in text for word in ("moon", "moons", "satellite", "satellites")):
        score += 4.0
        reasons.append("planet_moon_context")
    return score, reasons


def score_orbit_order_context(chunk: dict[str, Any], text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    section = str(chunk.get("section", "")).lower()
    title = str(chunk.get("title", "")).lower()

    if section in {"inner planets", "outer planets", "orbits"}:
        score += 5.0
        reasons.append("planet_order_section")
    if title == "solar system" and any(value in text for value in ("inner planets", "outer planets", "au)", "from the sun")):
        score += 4.0
        reasons.append("solar_system_order_context")
    if title_is_planet(title) and any(value in text for value in ("from the sun", "au", "orbit", "fifth planet", "seventh planet", "eighth planet", "fourth planet")):
        score += 3.0
        reasons.append("planet_orbit_context")
    return score, reasons


def re_moon_count_claim(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:has|have|had|includes?|possesses?)\s+(?:at\s+least\s+)?(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred)\s+"
            r"(?:known\s+|confirmed\s+|natural\s+)?(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred)\s+"
            r"(?:\w+\s+){0,4}(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b(?:mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)'?s\s+"
            r"(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred)\s+"
            r"(?:\w+\s+){0,4}(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b(?:mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)\s+has\s+"
            r"(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred).{0,90}?"
            r"\b(?:moon|moons|satellite|satellites)\b",
            text,
        )
    )


def title_is_planet(title: str) -> bool:
    normalized = title.replace(" (planet)", "")
    return normalized in {
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    }

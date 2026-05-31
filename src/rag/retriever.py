"""Lexical retriever for the first RAG+LLM benchmark slice."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from number_parser import parse_number as parse_number_words
except Exception:  # pragma: no cover - optional dependency fallback
    parse_number_words = None

from src.question_classifier import LogicalModifier, QuestionClassifier
from src.rag.embeddings import (
    DEFAULT_BGE_BASE_EMBEDDINGS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_OPENAI_EMBEDDINGS_PATH,
    EmbeddingIndex,
    bge_m3_colbert_scores,
    cosine_similarity,
    embedding_text_for_chunk,
    ensure_embedding_cache,
    sparse_dot,
)
from src.rag.retrieval_intent import (
    RetrievalIntent,
    build_retrieval_intent,
    score_filter_evidence,
    score_ordering_evidence,
    score_target_class_evidence,
)
from src.rag.retriever_terms import tokenize
from src.rag.routing import DEFAULT_ROUTING_EMBEDDINGS_PATH, DEFAULT_ROUTING_UNITS_PATH
from src.rag.source_selector import (
    SourceScore,
    SourceSelection,
    SourceSelector,
    apply_constraint_coverage,
    apply_pdf_raw_frequency_safety_include,
    profiles_are_pdf_only,
)
from src.rag.temporal_evidence import (
    compatible_current_chunks,
    compatible_temporal_chunks,
    contains_numeric_moon_count_assertion,
    contains_target_count_assertion,
    current_fact_match_kind,
    current_match_score,
    is_current_count_intent,
    is_temporal_count_intent,
    temporal_fact_match_kind,
    temporal_match_score,
)


RETRIEVAL_MODES = {
    "hierarchical-bge-m3-rrf",
    "global",
    "auto-source",
    "vector",
    "hybrid",
    "bge-m3-rrf",
    "bge-base-rrf",
    "openai-embedding-rrf",
}
DEFAULT_RETRIEVAL_MODE = "hierarchical-bge-m3-rrf"
HYBRID_SOURCE_LEXICAL_WEIGHT = 0.60
HYBRID_SOURCE_VECTOR_WEIGHT = 0.40
HYBRID_CHUNK_LEXICAL_WEIGHT = 0.50
HYBRID_CHUNK_VECTOR_WEIGHT = 0.50
RRF_K = 60
BGE_M3_SOURCE_RRF_WEIGHTS = {"dense": 0.30, "sparse": 0.45, "lexical": 0.25}
BGE_M3_CHUNK_RRF_WEIGHTS = {"dense": 0.35, "sparse": 0.40, "lexical": 0.25}
BGE_M3_FINAL_RRF_WEIGHTS = {"colbert": 0.45, "sparse": 0.25, "dense": 0.20, "lexical": 0.10}
DENSE_LEXICAL_SOURCE_RRF_WEIGHTS = {"dense": 0.40, "lexical": 0.60}
DENSE_LEXICAL_CHUNK_RRF_WEIGHTS = {"dense": 0.50, "lexical": 0.50}
COLBERT_RERANK_CANDIDATES = 80
ROUTE_RRF_WEIGHTS = {"dense": 0.35, "sparse": 0.40, "lexical": 0.25}
ROUTE_SOURCE_HIT_WEIGHTS = (1.0, 0.50, 0.25)
ROUTE_IDENTITY_PRIOR_WEIGHT = 0.40
ROUTE_IDENTITY_PRIOR_CAP = 8.0
ROUTE_CATALOG_PRIOR_WEIGHT = 0.10
ROUTE_CATALOG_PRIOR_CAP = 2.0
ROUTE_IDENTITY_GUARD_SOURCES = 2
CLASS_ENTITY_MEMBERS = {
    "planets": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    "dwarf_planets": ["Ceres", "Pluto", "Eris", "Haumea", "Makemake", "Quaoar", "Orcus", "Sedna", "Gonggong"],
}
MOON_COUNT_CONTEXT_CHUNK_LIMIT = 8
MOON_COUNT_SUPPORT_LIMIT = 16
MOON_COUNT_SUPPORT_PER_ENTITY = 2
AUTOMATIC_FALLBACK_CHAIN = [
    "hierarchical-bge-m3-rrf",
    "bge-m3-rrf",
    "bge-base-rrf",
    "openai-embedding-rrf",
    "auto-source",
    "global",
]


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
    requested_mode: str = ""
    source_selection: SourceSelection | None = None
    fallback_used: bool = False
    fallback_reason: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    embeddings_path: str = ""
    routing_units_total: int = 0
    routing_units_scored: int = 0
    selected_route_ids: list[str] = field(default_factory=list)
    candidate_chunks_scored: int = 0
    full_chunk_count: int = 0
    comparison_reduction_percent: float | None = None
    colbert_candidates: int = 0
    temporal_evidence_status: str = ""
    temporal_evidence_reason: str = ""
    current_evidence_status: str = ""
    current_evidence_reason: str = ""


class RagRetriever:
    def __init__(
        self,
        chunks_path: str | Path = "data/rag_sources/rag_index/chunks.jsonl",
        documents_path: str | Path = "data/rag_sources/rag_index/documents.jsonl",
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
        bge_base_embeddings_path: str | Path = DEFAULT_BGE_BASE_EMBEDDINGS_PATH,
        openai_embeddings_path: str | Path = DEFAULT_OPENAI_EMBEDDINGS_PATH,
        routing_units_path: str | Path = DEFAULT_ROUTING_UNITS_PATH,
        routing_embeddings_path: str | Path = DEFAULT_ROUTING_EMBEDDINGS_PATH,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks(self.chunks_path)
        self.documents_path = Path(documents_path)
        self.embeddings_path = Path(embeddings_path)
        self.bge_base_embeddings_path = Path(bge_base_embeddings_path)
        self.openai_embeddings_path = Path(openai_embeddings_path)
        self.routing_units_path = Path(routing_units_path)
        self.routing_embeddings_path = Path(routing_embeddings_path)
        self._embedding_index: EmbeddingIndex | None = None
        self._embedding_indexes: dict[Path, EmbeddingIndex] = {}
        self._routing_units: list[dict[str, Any]] | None = None
        self._routing_embedding_index: EmbeddingIndex | None = None
        self.question_classifier = QuestionClassifier()
        self._source_selector: SourceSelector | None = None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 12,
        per_source_limit: int = 4,
        mode: str = DEFAULT_RETRIEVAL_MODE,
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
        mode: str = DEFAULT_RETRIEVAL_MODE,
        top_n_sources: int = 5,
    ) -> RagRetrievalResult:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(RETRIEVAL_MODES))}")

        if mode == "hierarchical-bge-m3-rrf":
            return self._retrieve_rrf_fallback_chain(
                question,
                modes=AUTOMATIC_FALLBACK_CHAIN,
                top_k=top_k,
                per_source_limit=per_source_limit,
                top_n_sources=top_n_sources,
            )
        if mode == "bge-m3-rrf":
            return self._retrieve_rrf_fallback_chain(
                question,
                modes=AUTOMATIC_FALLBACK_CHAIN[1:],
                top_k=top_k,
                per_source_limit=per_source_limit,
                top_n_sources=top_n_sources,
            )
        if mode == "bge-base-rrf":
            return self._retrieve_rrf_fallback_chain(
                question,
                modes=AUTOMATIC_FALLBACK_CHAIN[2:],
                top_k=top_k,
                per_source_limit=per_source_limit,
                top_n_sources=top_n_sources,
            )
        if mode == "openai-embedding-rrf":
            return self._retrieve_rrf_fallback_chain(
                question,
                modes=AUTOMATIC_FALLBACK_CHAIN[3:],
                top_k=top_k,
                per_source_limit=per_source_limit,
                top_n_sources=top_n_sources,
            )

        source_selection = None
        selected_source_ids = None
        fallback_used = False
        fallback_reason = ""
        embedding_provider = ""
        embedding_model = ""
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
            embedding_index = self.embedding_index
            embedding_provider = embedding_index.provider
            embedding_model = embedding_index.model
            query_embedding = embedding_index.embed_query(question)
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
        coverage_source_ids = (
            source_selection.coverage_source_ids
            if source_selection
            else []
        )
        if mode == "vector":
            retrieved = self._retrieve_from_chunks_vector(
                candidate_chunks,
                vector_scores=vector_scores or {},
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
                required_source_ids=coverage_source_ids,
            )
        elif mode == "hybrid":
            retrieved = self._retrieve_from_chunks_hybrid(
                question,
                chunks=candidate_chunks,
                vector_scores=vector_scores or {},
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
                required_source_ids=coverage_source_ids,
            )
        else:
            retrieved = self._retrieve_from_chunks(
                question,
                chunks=candidate_chunks,
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
                required_source_ids=coverage_source_ids,
            )

        if source_selection and source_selection.fallback_used:
            fallback_used = True
            fallback_reason = fallback_reason or source_selection.fallback_reason

        result = RagRetrievalResult(
            retrieved_chunks=retrieved,
            retrieval_mode=mode,
            requested_mode=mode,
            source_selection=source_selection,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embeddings_path=str(self.embeddings_path),
        )
        return self._apply_moon_count_evidence_gate(question, result, top_k=top_k)

    def _retrieve_rrf_fallback_chain(
        self,
        question: str,
        *,
        modes: list[str],
        top_k: int,
        per_source_limit: int,
        top_n_sources: int,
    ) -> RagRetrievalResult:
        failures = []
        requested_mode = modes[0]
        for mode in modes:
            try:
                if mode == "hierarchical-bge-m3-rrf":
                    result = self._retrieve_hierarchical_mode_once(
                        question,
                        top_k=top_k,
                        per_source_limit=per_source_limit,
                        top_n_sources=top_n_sources,
                    )
                elif mode in {"auto-source", "global"}:
                    result = self.retrieve_with_details(
                        question,
                        top_k=top_k,
                        per_source_limit=per_source_limit,
                        mode=mode,
                        top_n_sources=top_n_sources,
                    )
                else:
                    if mode == "openai-embedding-rrf":
                        print(
                            "Attempting OpenAI embedding fallback. "
                            "This may build missing cache records and sends the query "
                            "to the OpenAI embeddings API."
                        )
                        self._ensure_openai_embedding_cache()
                    result = self._retrieve_rrf_mode_once(
                        question,
                        mode=mode,
                        top_k=top_k,
                        per_source_limit=per_source_limit,
                        top_n_sources=top_n_sources,
                    )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                failures.append(f"{mode}:{type(exc).__name__}:{exc}")
                continue
            except Exception as exc:
                if mode != "openai-embedding-rrf":
                    raise
                failures.append(f"{mode}:{type(exc).__name__}:{exc}")
                continue

            if (
                result.temporal_evidence_status in {"insufficient", "conflict"}
                or result.current_evidence_status in {"insufficient", "conflict"}
            ):
                result.requested_mode = requested_mode
                return result
            result.requested_mode = requested_mode
            if mode != requested_mode or failures:
                result.fallback_used = True
                reason_parts = failures + [f"selected:{mode}"]
                result.fallback_reason = "; ".join(reason_parts)
            return result

        result = self.retrieve_with_details(
            question,
            top_k=top_k,
            per_source_limit=per_source_limit,
            mode="global",
            top_n_sources=top_n_sources,
        )
        result.requested_mode = requested_mode
        result.fallback_used = True
        result.fallback_reason = "; ".join(failures + ["selected:global"])
        return result

    def _retrieve_hierarchical_mode_once(
        self,
        question: str,
        *,
        top_k: int,
        per_source_limit: int,
        top_n_sources: int,
    ) -> RagRetrievalResult:
        routes = self.routing_units
        routing_index = self.routing_embedding_index
        chunk_index = self._embedding_index_for_rrf_mode("bge-m3-rrf")
        if routing_index.provider != "bge-m3" or chunk_index.provider != "bge-m3":
            raise RuntimeError("Hierarchical retrieval requires BGE-M3 routing and chunk embedding caches.")
        if routing_index.model != chunk_index.model:
            raise RuntimeError("Routing and chunk embedding caches must use the same BGE-M3 model.")

        query_features = routing_index.encode_query(question)
        route_dense_scores = self._dense_scores_for_items(routes, routing_index, query_features.dense)
        route_sparse_scores = self._sparse_scores_for_items(routes, routing_index, query_features.sparse)
        route_items = self._rank_routing_units(
            question,
            routes=routes,
            dense_scores=route_dense_scores,
            sparse_scores=route_sparse_scores,
        )
        source_selection, selected_route_ids = self._select_sources_from_routes(
            question,
            route_items=route_items,
            top_n_sources=top_n_sources,
        )
        selected_source_ids = set(source_selection.selected_source_ids)
        candidate_chunks = [
            chunk for chunk in self.chunks
            if chunk.get("source_id") in selected_source_ids
        ]
        compared_units = len(routes) + len(candidate_chunks)
        reduction_percent = (1.0 - (compared_units / max(1, len(self.chunks)))) * 100.0

        chunk_dense_scores = self._dense_scores_for_items(candidate_chunks, chunk_index, query_features.dense)
        chunk_sparse_scores = self._sparse_scores_for_items(candidate_chunks, chunk_index, query_features.sparse)
        retrieved = self._retrieve_from_chunks_rrf(
            question,
            chunks=candidate_chunks,
            mode="bge-m3-rrf",
            dense_scores=chunk_dense_scores,
            sparse_scores=chunk_sparse_scores,
            embedding_index=chunk_index,
            top_k=top_k * 2,
            per_source_limit=self._effective_per_source_limit(question, per_source_limit),
            required_source_ids=source_selection.coverage_source_ids,
        )
        retrieved = suppress_near_duplicate_chunks(
            retrieved,
            top_k=top_k,
            required_source_ids=source_selection.coverage_source_ids,
        )
        result = RagRetrievalResult(
            retrieved_chunks=retrieved,
            retrieval_mode="hierarchical-bge-m3-rrf",
            requested_mode="hierarchical-bge-m3-rrf",
            source_selection=source_selection,
            embedding_provider=routing_index.provider,
            embedding_model=routing_index.model,
            embeddings_path=str(self.routing_embeddings_path),
            routing_units_total=len(routes),
            routing_units_scored=len(routes),
            selected_route_ids=selected_route_ids,
            candidate_chunks_scored=len(candidate_chunks),
            full_chunk_count=len(self.chunks),
            comparison_reduction_percent=round(reduction_percent, 2),
            colbert_candidates=min(len(candidate_chunks), COLBERT_RERANK_CANDIDATES),
        )
        return self._apply_moon_count_evidence_gate(question, result, top_k=top_k)

    def _rank_routing_units(
        self,
        question: str,
        *,
        routes: list[dict[str, Any]],
        dense_scores: dict[str, float],
        sparse_scores: dict[str, float],
    ) -> list[RetrievedChunk]:
        lexical_items = self._score_chunks_lexical(question, chunks=routes)
        dense_items = [
            RetrievedChunk(chunk=route, score=dense_scores[route["chunk_id"]] * 100.0, reasons=[])
            for route in routes
            if route["chunk_id"] in dense_scores
        ]
        sparse_items = [
            RetrievedChunk(chunk=route, score=sparse_scores[route["chunk_id"]], reasons=[])
            for route in routes
            if route["chunk_id"] in sparse_scores
        ]
        rank_maps = {
            "lexical": chunk_rank_map(lexical_items),
            "dense": chunk_rank_map(sorted(dense_items, key=lambda item: item.score, reverse=True)),
            "sparse": chunk_rank_map(sorted(sparse_items, key=lambda item: item.score, reverse=True)),
        }
        item_maps = {
            "lexical": {item.chunk["chunk_id"]: item for item in lexical_items},
            "dense": {item.chunk["chunk_id"]: item for item in dense_items},
            "sparse": {item.chunk["chunk_id"]: item for item in sparse_items},
        }
        ranked = combine_chunk_rrf(
            chunk_ids={route["chunk_id"] for route in routes},
            chunk_by_id={route["chunk_id"]: route for route in routes},
            rank_maps=rank_maps,
            item_maps=item_maps,
            weights=ROUTE_RRF_WEIGHTS,
        )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _select_sources_from_routes(
        self,
        question: str,
        *,
        route_items: list[RetrievedChunk],
        top_n_sources: int,
    ) -> tuple[SourceSelection, list[str]]:
        content_by_source: dict[str, list[RetrievedChunk]] = defaultdict(list)
        identity_by_source: dict[str, RetrievedChunk] = {}
        catalog_by_source: dict[str, RetrievedChunk] = {}
        for item in route_items:
            source_id = item.chunk["source_id"]
            route_type = item.chunk.get("route_type")
            if route_type == "source_identity":
                identity_by_source.setdefault(source_id, item)
                continue
            if route_type in {"source_catalog", "metadata"}:
                # Treat pre-split metadata indexes as low-weight catalog routes.
                catalog_by_source.setdefault(source_id, item)
                continue
            existing = content_by_source[source_id]
            if any(routes_overlap(item.chunk, previous.chunk) for previous in existing):
                continue
            existing.append(item)

        scores: list[SourceScore] = []
        for source_id in sorted(set(content_by_source) | set(identity_by_source) | set(catalog_by_source)):
            items = content_by_source.get(source_id, [])
            best_items = items[: len(ROUTE_SOURCE_HIT_WEIGHTS)]
            content_score = sum(
                weight * item.score
                for weight, item in zip(ROUTE_SOURCE_HIT_WEIGHTS, best_items)
            )
            identity_item = identity_by_source.get(source_id)
            identity_prior = (
                min(ROUTE_IDENTITY_PRIOR_CAP, identity_item.score * ROUTE_IDENTITY_PRIOR_WEIGHT)
                if identity_item
                else 0.0
            )
            catalog_item = catalog_by_source.get(source_id)
            catalog_prior = (
                min(ROUTE_CATALOG_PRIOR_CAP, catalog_item.score * ROUTE_CATALOG_PRIOR_WEIGHT)
                if catalog_item
                else 0.0
            )
            score = content_score + identity_prior + catalog_prior
            best_item = best_items[0] if best_items else identity_item or catalog_item
            best = best_item.chunk
            source_chunks = [chunk for chunk in self.chunks if chunk["source_id"] == source_id]
            reasons = [
                f"content_route_weighted_score:{content_score:.4f}",
                *[
                    f"route_hit_{index}:{item.chunk['route_id']}:{item.score:.4f}"
                    for index, item in enumerate(best_items, start=1)
                ],
            ]
            if identity_item:
                reasons.append(
                    f"identity_prior:{identity_item.chunk['route_id']}:{identity_prior:.4f}"
                )
            if catalog_item:
                reasons.append(
                    f"catalog_prior:{catalog_item.chunk['route_id']}:{catalog_prior:.4f}"
                )
            scores.append(SourceScore(
                source_id=source_id,
                score=score,
                reasons=reasons,
                title=best.get("title", ""),
                url=best.get("url", ""),
                chunk_count=len(source_chunks),
                char_count=sum(len(chunk.get("text", "")) for chunk in source_chunks),
            ))
        scores.sort(key=lambda item: item.score, reverse=True)

        budget = self._hierarchical_source_budget(question, top_n_sources)
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        profiles = self.source_selector.profiles if hasattr(self, "_source_selector") else {}
        pdf_only = profiles_are_pdf_only(profiles)
        if pdf_only:
            selected = [score.source_id for score in scores[:budget]]
            coverage_source_ids: list[str] = []
            coverage_reasons: list[str] = []
            selected, pdf_reasons = apply_pdf_raw_frequency_safety_include(
                selected,
                profiles,
                scores,
                intent,
                top_n_sources=budget,
            )
            coverage_reasons.extend(pdf_reasons)
            selected_route_items = [
                item
                for source_id in selected
                for item in content_by_source.get(source_id, [])[: len(ROUTE_SOURCE_HIT_WEIGHTS)]
            ]
            selected_route_items.extend(
                identity_by_source[source_id]
                for source_id in selected
                if source_id in identity_by_source
            )
            selected_route_items.extend(
                catalog_by_source[source_id]
                for source_id in selected
                if source_id in catalog_by_source
            )
            selected_route_items.sort(key=lambda item: item.score, reverse=True)
            selected_routes = [
                item.chunk["route_id"]
                for item in selected_route_items
            ][: max(10, budget * 3)]
            return SourceSelection(
                mode="hierarchical-bge-m3-rrf",
                selected_source_ids=selected,
                scores=scores,
                coverage_source_ids=coverage_source_ids,
                coverage_reasons=coverage_reasons,
            ), selected_routes

        guard_sources = []
        content_route_items = [
            item for item in route_items
            if item.chunk.get("route_type") not in {"source_identity", "source_catalog", "metadata"}
        ]
        for item in content_route_items[: max(3, budget)]:
            source_id = item.chunk["source_id"]
            if source_id not in guard_sources:
                guard_sources.append(source_id)
            if len(guard_sources) >= min(2, budget):
                break
        identity_scan_limit = max(12, budget * 4)
        top_identity_candidates = [
            item for item in route_items[:identity_scan_limit]
            if item.chunk.get("route_type") == "source_identity"
        ][:ROUTE_IDENTITY_GUARD_SOURCES]
        for item in top_identity_candidates:
            source_id = item.chunk["source_id"]
            if source_id not in guard_sources:
                guard_sources.append(source_id)
            if len(guard_sources) >= budget:
                break
        selected = list(guard_sources)
        for source_score in scores:
            if source_score.source_id not in selected:
                selected.append(source_score.source_id)
            if len(selected) >= budget:
                break
        relational_coverage_needed = bool(
            intent.target_class
            and any(
                condition.get("operator") in {"<", ">"}
                and condition.get("attribute") not in {None, "", "unknown"}
                for condition in intent.filter_conditions
            )
        )
        selected, coverage_source_ids, coverage_reasons = apply_constraint_coverage(
            selected,
            profiles if relational_coverage_needed else {},
            intent,
        )
        if profiles:
            selected, pdf_reasons = apply_pdf_raw_frequency_safety_include(
                selected,
                profiles,
                scores,
                intent,
                top_n_sources=budget,
            )
            coverage_reasons.extend(pdf_reasons)
        selected_route_items = [
            item
            for source_id in selected
            for item in content_by_source.get(source_id, [])[: len(ROUTE_SOURCE_HIT_WEIGHTS)]
        ]
        selected_route_items.extend(
            identity_by_source[source_id]
            for source_id in selected
            if source_id in identity_by_source
        )
        selected_route_items.extend(
            catalog_by_source[source_id]
            for source_id in selected
            if source_id in catalog_by_source
        )
        selected_route_items.sort(key=lambda item: item.score, reverse=True)
        selected_routes = [
            item.chunk["route_id"]
            for item in selected_route_items
        ][: max(10, budget * 3)]
        return SourceSelection(
            mode="hierarchical-bge-m3-rrf",
            selected_source_ids=selected,
            scores=scores,
            coverage_source_ids=coverage_source_ids,
            coverage_reasons=coverage_reasons,
        ), selected_routes

    def _hierarchical_source_budget(self, question: str, maximum: int) -> int:
        classified = self.question_classifier.classify(question)
        modifiers = set(classified.logical_modifiers)
        if str(classified.primary_type.name) == "LIST":
            return min(maximum, 12)
        if LogicalModifier.COMPARISON in modifiers:
            return min(maximum, 8)
        if LogicalModifier.TIME_LOOKUP in modifiers:
            return min(maximum, 6)
        return min(maximum, 5)

    def _apply_moon_count_evidence_gate(
        self,
        question: str,
        result: RagRetrievalResult,
        *,
        top_k: int,
    ) -> RagRetrievalResult:
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        if is_current_count_intent(intent):
            return self._apply_current_count_gate(intent, result, top_k=top_k)
        if self._needs_class_moon_count_coverage(intent):
            return self._apply_class_moon_count_coverage(question, intent, result)
        if not is_temporal_count_intent(intent):
            return result

        compatible = compatible_temporal_chunks(self.chunks, intent)
        if not compatible:
            result.retrieved_chunks = []
            result.temporal_evidence_status = "insufficient"
            result.temporal_evidence_reason = "no_validated_temporal_fact_covers_requested_time"
            return result

        values = {
            chunk["temporal_fact"].get("value")
            for chunk, _kind in compatible
        }
        if len(values) > 1:
            result.temporal_evidence_status = "conflict"
            result.temporal_evidence_reason = "conflicting_validated_temporal_facts"
        else:
            result.temporal_evidence_status = "supported"
            result.temporal_evidence_reason = compatible[0][1]

        time_constraints = extract_time_constraints(question)
        evidence_items = []
        for chunk, match_kind in compatible:
            score, reasons = self._score_chunk(
                chunk,
                intent=intent,
                time_constraints=time_constraints,
            )
            evidence_items.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                reasons=[f"required_temporal_evidence:{match_kind}", *reasons],
            ))
        evidence_ids = {item.chunk["chunk_id"] for item in evidence_items}
        retained = [
            item for item in result.retrieved_chunks
            if (
                (
                    not item.chunk.get("temporal_fact")
                    and not contains_target_count_assertion(item.chunk, intent)
                )
                or item.chunk["chunk_id"] in evidence_ids
            )
        ]
        merged = sorted(
            {item.chunk["chunk_id"]: item for item in [*retained, *evidence_items]}.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        result.retrieved_chunks = merged[:top_k]
        if result.source_selection:
            for chunk, _kind in compatible:
                source_id = chunk["source_id"]
                if source_id not in result.source_selection.selected_source_ids:
                    result.source_selection.selected_source_ids.append(source_id)
        return result

    def _needs_class_moon_count_coverage(self, intent: RetrievalIntent) -> bool:
        return bool(
            intent.target_class
            and "moon_count" in intent.predicate_terms
            and any(condition.get("attribute") == "moon_count" for condition in intent.filter_conditions)
        )

    def _apply_class_moon_count_coverage(
        self,
        question: str,
        intent: RetrievalIntent,
        result: RagRetrievalResult,
    ) -> RagRetrievalResult:
        entities = self._moon_count_coverage_entities(intent)
        if not entities:
            return result

        resolved_lines: list[str] = []
        missing_entities: list[str] = []
        resolved_source_ids: list[str] = []
        is_temporal_lookup = intent.has_time_lookup and bool(intent.time_value)
        for entity in entities:
            match = self._resolve_moon_count_for_entity(entity, intent)
            if match is None:
                resolved_lines.append(f"- {entity}: missing")
                missing_entities.append(entity)
                continue
            chunk, _match_kind = match
            fact = chunk["temporal_fact"]
            value = fact.get("value")
            if is_temporal_lookup:
                observed = fact.get("observed_at") or intent.time_value
                valid_until = fact.get("valid_until_exclusive")
                interval = f", valid until {valid_until}" if valid_until else ""
                resolved_lines.append(f"- {entity}: {value} moons as of {observed}{interval}")
            else:
                resolved_lines.append(
                    f"- {entity}: {value} moons "
                    f"(highest structured count found; source={chunk.get('source_id', '')})"
                )
            resolved_source_ids.append(chunk.get("source_id", ""))

        table_title = (
            "Resolved temporal moon-count facts"
            if is_temporal_lookup
            else "Highest structured moon-count facts"
        )
        table_intro = (
            "Resolved moon-count facts for the target class:"
            if is_temporal_lookup
            else (
                "Highest structured moon-count claims found for the target class "
                "(non-temporal retrieval summary):"
            )
        )
        table_chunk = {
            "chunk_id": f"resolved_moon_count_facts_{slug(intent.target_class or 'entities')}",
            "document_id": "resolved_moon_count_facts",
            "source_id": "resolved_moon_count_facts",
            "url": "",
            "title": table_title,
            "section": table_title,
            "text": "\n".join([table_intro, *resolved_lines]),
            "target_questions": [],
            "needed_evidence": [],
            "entities": entities,
            "predicate_hints": ["moon_count"],
            "tokens_estimate": max(1, len(resolved_lines) * 8),
            "trust_level": "resolved",
            "content_type": "resolved_fact_table",
        }
        table_item = RetrievedChunk(
            chunk=table_chunk,
            score=100.0,
            reasons=["required_moon_count_coverage_table"],
        )

        existing_ids = {item.chunk["chunk_id"] for item in result.retrieved_chunks}
        support_entities = missing_entities if is_temporal_lookup else entities
        support_items = self._missing_moon_count_support_chunks(
            support_entities,
            intent=intent,
            existing_chunk_ids=existing_ids | {table_chunk["chunk_id"]},
            allowed_source_ids=None,
        )
        context_items = result.retrieved_chunks[:MOON_COUNT_CONTEXT_CHUNK_LIMIT]
        merged = {item.chunk["chunk_id"]: item for item in [table_item, *context_items, *support_items]}
        result.retrieved_chunks = list(merged.values())

        if result.source_selection:
            for source_id in resolved_source_ids:
                if source_id and source_id not in result.source_selection.selected_source_ids:
                    result.source_selection.selected_source_ids.append(source_id)
            for item in support_items:
                source_id = item.chunk["source_id"]
                if source_id not in result.source_selection.selected_source_ids:
                    result.source_selection.selected_source_ids.append(source_id)
        return result

    def _moon_count_coverage_entities(self, intent: RetrievalIntent) -> list[str]:
        configured = CLASS_ENTITY_MEMBERS.get(intent.target_class or "", [])
        if configured:
            return configured
        references = {
            str(condition.get("reference_entity", "")).lower()
            for condition in intent.filter_conditions
            if condition.get("reference_entity")
        }
        return [
            entity.title()
            for entity in intent.entity_terms
            if entity and entity not in references
        ]

    def _resolve_moon_count_for_entity(
        self,
        entity: str,
        intent: RetrievalIntent,
    ) -> tuple[dict[str, Any], str] | None:
        entity_intent = self._single_entity_moon_count_intent(entity, intent)
        if intent.has_time_lookup and intent.time_value:
            matches = compatible_temporal_chunks(self.chunks, entity_intent)
        else:
            matches = compatible_current_chunks(self.chunks, entity_intent)
            if not matches:
                raw_match = self._resolve_raw_current_moon_count_for_entity(entity)
                if raw_match:
                    return raw_match
        return matches[0] if matches else None

    def _resolve_raw_current_moon_count_for_entity(self, entity: str) -> tuple[dict[str, Any], str] | None:
        entity_lower = entity.lower()
        for chunk in self.chunks:
            text = " ".join([
                chunk.get("title", ""),
                chunk.get("section", ""),
                chunk.get("text", ""),
            ]).lower()
            if entity_lower not in text:
                continue
            value = None
            if re.search(rf"\b{re.escape(entity_lower)}\b[^.]*\bhas\s+no\s+natural\s+satellites\b", text):
                value = 0
            elif re.search(rf"\b{re.escape(entity_lower)}\b[^.]*\bhas\s+no\s+moons\b", text):
                value = 0
            elif re.search(rf"\b{re.escape(entity_lower)}\b[^.]*\bis\s+orbited\s+by\s+one\s+(?:permanent\s+)?natural\s+satellite\b", text):
                value = 1
            elif re.search(rf"\b(?:moon|the\s+moon)\s+is\s+{re.escape(entity_lower)}'s\s+only\s+natural\s+satellite\b", text):
                value = 1
            if value is None:
                continue

            resolved = dict(chunk)
            resolved["chunk_id"] = f"{chunk['chunk_id']}__resolved_current_moon_count"
            resolved["text"] = f"Resolved current source assertion: {entity} has {value} moons."
            resolved["content_type"] = "resolved_fact"
            resolved["temporal_fact"] = {
                "subject": entity,
                "predicate": "moon_count",
                "value": value,
                "evidence_type": "explicit_current_sentence",
                "validation_status": "source_asserted",
                "claim_type": "moon_count",
                "observed_at": "",
                "valid_until_exclusive": "",
                "interval_semantics": "",
            }
            return resolved, "raw_current_assertion"
        return None

    @staticmethod
    def _single_entity_moon_count_intent(entity: str, intent: RetrievalIntent) -> Any:
        return SimpleNamespace(
            has_time_lookup=intent.has_time_lookup,
            time_value=intent.time_value,
            time_semantic=intent.time_semantic,
            predicate_terms=["moon_count"],
            entity_terms=[entity.lower()],
            has_comparison=False,
            filter_conditions=[],
            ordering_attribute=None,
        )

    def _missing_moon_count_support_chunks(
        self,
        missing_entities: list[str],
        *,
        intent: RetrievalIntent,
        existing_chunk_ids: set[str],
        allowed_source_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        support_items: list[RetrievedChunk] = []
        time_constraints = {"phrases": [], "years": [], "months": []}
        for entity in missing_entities:
            entity_lower = entity.lower()
            candidates = []
            for chunk in self.chunks:
                if chunk["chunk_id"] in existing_chunk_ids:
                    continue
                if allowed_source_ids is not None and chunk.get("source_id") not in allowed_source_ids:
                    continue
                text = " ".join([
                    chunk.get("title", ""),
                    chunk.get("section", ""),
                    chunk.get("text", ""),
                ]).lower()
                if entity_lower not in text:
                    continue
                if "moon_count" not in set(chunk.get("predicate_hints", [])) and not any(
                    value in text for value in ("moon", "moons", "satellite", "satellites")
                ):
                    continue
                score, reasons = self._score_chunk(chunk, intent=intent, time_constraints=time_constraints)
                candidates.append(RetrievedChunk(
                    chunk=chunk,
                    score=score + 12.0,
                    reasons=[f"missing_moon_count_support:{slug(entity)}", *reasons],
                ))
            candidates.sort(key=lambda item: item.score, reverse=True)
            for item in candidates[:MOON_COUNT_SUPPORT_PER_ENTITY]:
                support_items.append(item)
                existing_chunk_ids.add(item.chunk["chunk_id"])
                if len(support_items) >= MOON_COUNT_SUPPORT_LIMIT:
                    return support_items
        return support_items

    def _apply_current_count_gate(
        self,
        intent: RetrievalIntent,
        result: RagRetrievalResult,
        *,
        top_k: int,
    ) -> RagRetrievalResult:
        compatible = compatible_current_chunks(self.chunks, intent)
        if not compatible:
            result.retrieved_chunks = []
            result.current_evidence_status = "insufficient"
            result.current_evidence_reason = "no_admissible_current_count_assertion"
            return result

        result.current_evidence_status = "supported"
        result.current_evidence_reason = "max_admissible_current_assertion"
        evidence_items = []
        for chunk, match_kind in compatible:
            score, reasons = self._score_chunk(
                chunk,
                intent=intent,
                time_constraints={"phrases": [], "years": [], "months": []},
            )
            evidence_items.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                reasons=[f"required_current_evidence:{match_kind}", *reasons],
            ))
        evidence_ids = {item.chunk["chunk_id"] for item in evidence_items}
        retained = [
            item for item in result.retrieved_chunks
            if (
                item.chunk["chunk_id"] in evidence_ids
                or (
                    not item.chunk.get("temporal_fact")
                    and not contains_numeric_moon_count_assertion(item.chunk)
                )
            )
        ]
        merged = sorted(
            {item.chunk["chunk_id"]: item for item in [*retained, *evidence_items]}.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        result.retrieved_chunks = merged[:top_k]
        if result.source_selection:
            for chunk, _kind in compatible:
                source_id = chunk["source_id"]
                if source_id not in result.source_selection.selected_source_ids:
                    result.source_selection.selected_source_ids.append(source_id)
        return result

    def _retrieve_rrf_mode_once(
        self,
        question: str,
        *,
        mode: str,
        top_k: int,
        per_source_limit: int,
        top_n_sources: int,
    ) -> RagRetrievalResult:
        embedding_index = self._embedding_index_for_rrf_mode(mode)
        query_features = embedding_index.encode_query(question)
        dense_scores = self._dense_scores_by_chunk(embedding_index, query_features.dense)
        sparse_scores = (
            self._sparse_scores_by_chunk(embedding_index, query_features.sparse)
            if mode == "bge-m3-rrf"
            else {}
        )
        source_selection = self._select_sources_rrf(
            question,
            mode=mode,
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            top_n_sources=top_n_sources,
        )
        selected_source_ids = set(source_selection.selected_source_ids)
        candidate_chunks = [
            chunk for chunk in self.chunks
            if not selected_source_ids or chunk.get("source_id") in selected_source_ids
        ]
        retrieved = self._retrieve_from_chunks_rrf(
            question,
            chunks=candidate_chunks,
            mode=mode,
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            embedding_index=embedding_index,
            top_k=top_k,
            per_source_limit=self._effective_per_source_limit(question, per_source_limit),
            required_source_ids=source_selection.coverage_source_ids,
        )
        result = RagRetrievalResult(
            retrieved_chunks=retrieved,
            retrieval_mode=mode,
            source_selection=source_selection,
            embedding_provider=embedding_index.provider,
            embedding_model=embedding_index.model,
            embeddings_path=str(embedding_index.path),
        )
        return self._apply_moon_count_evidence_gate(question, result, top_k=top_k)

    def _ensure_openai_embedding_cache(self) -> None:
        result = ensure_embedding_cache(
            self.chunks,
            path=self.openai_embeddings_path,
            provider="openai",
        )
        if result.built:
            print(
                f"Built/updated OpenAI embedding cache: {result.path} "
                f"({result.total_chunks} chunks)."
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
        required_source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        scored = self._score_chunks_lexical(question, chunks=chunks)
        return cap_per_source(
            scored,
            top_k=top_k,
            per_source_limit=per_source_limit,
            required_source_ids=required_source_ids,
        )

    def _score_chunks_lexical(
        self,
        question: str,
        *,
        chunks: list[dict[str, Any]],
    ) -> list[RetrievedChunk]:
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        time_constraints = extract_time_constraints(question)

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            score, reasons = self._score_chunk(
                chunk,
                intent=intent,
                time_constraints=time_constraints,
            )
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score, reasons=reasons))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    @property
    def embedding_index(self) -> EmbeddingIndex:
        if self._embedding_index is None:
            self._embedding_index = EmbeddingIndex(self.embeddings_path)
        return self._embedding_index

    @property
    def source_selector(self) -> SourceSelector:
        if self._source_selector is None:
            self._source_selector = SourceSelector(self.chunks, documents_path=self.documents_path)
        return self._source_selector

    @property
    def routing_units(self) -> list[dict[str, Any]]:
        if self._routing_units is None:
            if not self.routing_units_path.exists():
                raise FileNotFoundError(
                    f"Routing units not found: {self.routing_units_path}. "
                    "Run: python scripts\\build_rag_routing_index.py"
                )
            self._routing_units = self._load_chunks(self.routing_units_path)
        return self._routing_units

    @property
    def routing_embedding_index(self) -> EmbeddingIndex:
        if self._routing_embedding_index is None:
            self._routing_embedding_index = EmbeddingIndex(self.routing_embeddings_path)
        return self._routing_embedding_index

    def _embedding_index_for_rrf_mode(self, mode: str) -> EmbeddingIndex:
        if mode == "bge-m3-rrf":
            path = self.embeddings_path
        elif mode == "bge-base-rrf":
            path = self.bge_base_embeddings_path
        elif mode == "openai-embedding-rrf":
            path = self.openai_embeddings_path
        else:
            raise ValueError(f"RRF embedding mode not supported: {mode}")
        path = Path(path)
        if path not in self._embedding_indexes:
            self._embedding_indexes[path] = EmbeddingIndex(path)
        return self._embedding_indexes[path]

    def _vector_scores_by_chunk(self, query_embedding: list[float]) -> dict[str, float]:
        return self._dense_scores_by_chunk(self.embedding_index, query_embedding)

    def _dense_scores_by_chunk(
        self,
        embedding_index: EmbeddingIndex,
        query_embedding: list[float],
    ) -> dict[str, float]:
        return self._dense_scores_for_items(self.chunks, embedding_index, query_embedding)

    def _dense_scores_for_items(
        self,
        items: list[dict[str, Any]],
        embedding_index: EmbeddingIndex,
        query_embedding: list[float],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item in items:
            embedding = embedding_index.get(item["chunk_id"])
            if embedding is None:
                continue
            scores[item["chunk_id"]] = cosine_similarity(query_embedding, embedding)
        return scores

    def _sparse_scores_by_chunk(
        self,
        embedding_index: EmbeddingIndex,
        query_sparse: dict[str, float] | None,
    ) -> dict[str, float]:
        return self._sparse_scores_for_items(self.chunks, embedding_index, query_sparse)

    def _sparse_scores_for_items(
        self,
        items: list[dict[str, Any]],
        embedding_index: EmbeddingIndex,
        query_sparse: dict[str, float] | None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not query_sparse:
            return scores
        for item in items:
            sparse_weights = embedding_index.get_sparse(item["chunk_id"])
            score = sparse_dot(query_sparse, sparse_weights)
            if score > 0:
                scores[item["chunk_id"]] = score
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
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        if profiles_are_pdf_only(self.source_selector.profiles):
            coverage_source_ids = []
            coverage_reasons = []
        else:
            selected, coverage_source_ids, coverage_reasons = apply_constraint_coverage(
                selected,
                self.source_selector.profiles,
                intent,
            )
        selected, pdf_reasons = apply_pdf_raw_frequency_safety_include(
            selected,
            self.source_selector.profiles,
            source_scores,
            intent,
            top_n_sources=top_n_sources,
        )
        coverage_reasons.extend(pdf_reasons)
        return SourceSelection(
            mode="vector",
            selected_source_ids=selected,
            scores=source_scores,
            coverage_source_ids=coverage_source_ids,
            coverage_reasons=coverage_reasons,
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
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        if profiles_are_pdf_only(self.source_selector.profiles):
            coverage_source_ids = []
            coverage_reasons = []
        else:
            selected, coverage_source_ids, coverage_reasons = apply_constraint_coverage(
                selected,
                self.source_selector.profiles,
                intent,
            )
        selected, pdf_reasons = apply_pdf_raw_frequency_safety_include(
            selected,
            self.source_selector.profiles,
            hybrid_scores,
            intent,
            top_n_sources=top_n_sources,
        )
        coverage_reasons.extend(pdf_reasons)
        return SourceSelection(
            mode="hybrid",
            selected_source_ids=selected,
            scores=hybrid_scores,
            coverage_source_ids=coverage_source_ids,
            coverage_reasons=coverage_reasons,
        )

    def _select_sources_rrf(
        self,
        question: str,
        *,
        mode: str,
        dense_scores: dict[str, float],
        sparse_scores: dict[str, float],
        top_n_sources: int,
    ) -> SourceSelection:
        lexical_selection = self.source_selector.select(
            question,
            top_n_sources=max(top_n_sources, len(self.source_selector.profiles)),
            min_score=0.0,
        )
        dense_source_scores = self._aggregate_vector_source_scores(dense_scores)
        sparse_source_scores = self._aggregate_sparse_source_scores(sparse_scores)
        lexical_ranks = source_rank_map(lexical_selection.scores)
        dense_ranks = source_rank_map(dense_source_scores)
        sparse_ranks = source_rank_map(sparse_source_scores)
        source_ids = set(lexical_ranks) | set(dense_ranks) | set(sparse_ranks)
        weights = (
            BGE_M3_SOURCE_RRF_WEIGHTS
            if mode == "bge-m3-rrf"
            else DENSE_LEXICAL_SOURCE_RRF_WEIGHTS
        )
        lexical_by_id = {score.source_id: score for score in lexical_selection.scores}
        dense_by_id = {score.source_id: score for score in dense_source_scores}
        sparse_by_id = {score.source_id: score for score in sparse_source_scores}

        rrf_scores = []
        for source_id in source_ids:
            rank_values = {
                "lexical": lexical_ranks.get(source_id),
                "dense": dense_ranks.get(source_id),
                "sparse": sparse_ranks.get(source_id),
            }
            score = weighted_rrf(rank_values, weights) * 1000.0
            profile = self.source_selector.profiles.get(source_id)
            reasons = [
                f"rrf:{score:.4f}",
                f"rank_lexical:{rank_values['lexical'] or 'none'}",
                f"rank_dense:{rank_values['dense'] or 'none'}",
            ]
            if "sparse" in weights:
                reasons.append(f"rank_sparse:{rank_values['sparse'] or 'none'}")
            if lexical_by_id.get(source_id):
                reasons.extend(lexical_by_id[source_id].reasons[:4])
            if dense_by_id.get(source_id):
                reasons.extend(dense_by_id[source_id].reasons[:2])
            if sparse_by_id.get(source_id):
                reasons.extend(sparse_by_id[source_id].reasons[:2])
            rrf_scores.append(SourceScore(
                source_id=source_id,
                score=score,
                reasons=reasons,
                title=(profile.title if profile else ""),
                url=(profile.url if profile else ""),
                chunk_count=(profile.chunk_count if profile else 0),
                char_count=(profile.char_count if profile else 0),
            ))

        rrf_scores.sort(key=lambda item: item.score, reverse=True)
        selected = [score.source_id for score in rrf_scores[:top_n_sources]]
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        if profiles_are_pdf_only(self.source_selector.profiles):
            coverage_source_ids = []
            coverage_reasons = []
        else:
            selected, coverage_source_ids, coverage_reasons = apply_constraint_coverage(
                selected,
                self.source_selector.profiles,
                intent,
            )
        selected, pdf_reasons = apply_pdf_raw_frequency_safety_include(
            selected,
            self.source_selector.profiles,
            rrf_scores,
            intent,
            top_n_sources=top_n_sources,
        )
        coverage_reasons.extend(pdf_reasons)
        return SourceSelection(
            mode=mode,
            selected_source_ids=selected,
            scores=rrf_scores,
            coverage_source_ids=coverage_source_ids,
            coverage_reasons=coverage_reasons,
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

    def _aggregate_sparse_source_scores(self, sparse_scores: dict[str, float]) -> list[SourceScore]:
        scores_by_source: dict[str, list[float]] = defaultdict(list)
        for chunk in self.chunks:
            score = sparse_scores.get(chunk["chunk_id"])
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
            source_scores.append(SourceScore(
                source_id=source_id,
                score=top_average,
                reasons=[
                    f"sparse_top_score:{top_score:.4f}",
                    f"sparse_top3_avg:{top_average:.4f}",
                    f"sparse_chunks:{len(scores)}",
                ],
                title=(profile.title if profile else ""),
                url=(profile.url if profile else ""),
                chunk_count=(profile.chunk_count if profile else len(scores)),
                char_count=(profile.char_count if profile else 0),
            ))
        source_scores.sort(key=lambda item: item.score, reverse=True)
        return source_scores

    def _retrieve_from_chunks_vector(
        self,
        chunks: list[dict[str, Any]],
        *,
        vector_scores: dict[str, float],
        top_k: int,
        per_source_limit: int,
        required_source_ids: list[str] | None = None,
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
        return cap_per_source(
            scored,
            top_k=top_k,
            per_source_limit=per_source_limit,
            required_source_ids=required_source_ids,
        )

    def _retrieve_from_chunks_hybrid(
        self,
        question: str,
        *,
        chunks: list[dict[str, Any]],
        vector_scores: dict[str, float],
        top_k: int,
        per_source_limit: int,
        required_source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        intent = build_retrieval_intent(question, classifier=self.question_classifier)
        time_constraints = extract_time_constraints(question)

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            lexical_score, lexical_reasons = self._score_chunk(
                chunk,
                intent=intent,
                time_constraints=time_constraints,
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
        return cap_per_source(
            scored,
            top_k=top_k,
            per_source_limit=per_source_limit,
            required_source_ids=required_source_ids,
        )

    def _retrieve_from_chunks_rrf(
        self,
        question: str,
        *,
        chunks: list[dict[str, Any]],
        mode: str,
        dense_scores: dict[str, float],
        sparse_scores: dict[str, float],
        embedding_index: EmbeddingIndex,
        top_k: int,
        per_source_limit: int,
        required_source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        lexical_items = self._score_chunks_lexical(question, chunks=chunks)
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        dense_items = [
            RetrievedChunk(chunk=chunk, score=dense_scores[chunk["chunk_id"]] * 100.0, reasons=[])
            for chunk in chunks
            if chunk["chunk_id"] in dense_scores
        ]
        sparse_items = [
            RetrievedChunk(chunk=chunk, score=sparse_scores[chunk["chunk_id"]], reasons=[])
            for chunk in chunks
            if chunk["chunk_id"] in sparse_scores
        ]
        lexical_ranks = chunk_rank_map(lexical_items)
        dense_ranks = chunk_rank_map(sorted(dense_items, key=lambda item: item.score, reverse=True))
        sparse_ranks = chunk_rank_map(sorted(sparse_items, key=lambda item: item.score, reverse=True))
        lexical_by_id = {item.chunk["chunk_id"]: item for item in lexical_items}
        dense_by_id = {item.chunk["chunk_id"]: item for item in dense_items}
        sparse_by_id = {item.chunk["chunk_id"]: item for item in sparse_items}
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        rank_maps = {
            "lexical": lexical_ranks,
            "dense": dense_ranks,
            "sparse": sparse_ranks,
        }
        item_maps = {
            "lexical": lexical_by_id,
            "dense": dense_by_id,
            "sparse": sparse_by_id,
        }
        weights = (
            BGE_M3_CHUNK_RRF_WEIGHTS
            if mode == "bge-m3-rrf"
            else DENSE_LEXICAL_CHUNK_RRF_WEIGHTS
        )
        initial_items = combine_chunk_rrf(
            chunk_ids=chunk_ids,
            chunk_by_id=chunk_by_id,
            rank_maps=rank_maps,
            item_maps=item_maps,
            weights=weights,
        )
        initial_items.sort(key=lambda item: item.score, reverse=True)
        if mode != "bge-m3-rrf":
            return cap_per_source(
                initial_items,
                top_k=top_k,
                per_source_limit=per_source_limit,
                required_source_ids=required_source_ids,
            )

        rerank_candidates = cap_per_source(
            initial_items,
            top_k=max(COLBERT_RERANK_CANDIDATES, top_k),
            per_source_limit=max(COLBERT_RERANK_CANDIDATES, top_k),
            required_source_ids=required_source_ids,
        )
        colbert_scores = bge_m3_colbert_scores(
            question,
            [embedding_text_for_chunk(item.chunk) for item in rerank_candidates],
            model=embedding_index.model,
        )
        colbert_items = [
            RetrievedChunk(chunk=item.chunk, score=score, reasons=[f"colbert_score:{score:.4f}"])
            for item, score in zip(rerank_candidates, colbert_scores)
        ]
        colbert_items.sort(key=lambda item: item.score, reverse=True)
        final_rank_maps = dict(rank_maps)
        final_rank_maps["colbert"] = chunk_rank_map(colbert_items)
        final_item_maps = dict(item_maps)
        final_item_maps["colbert"] = {item.chunk["chunk_id"]: item for item in colbert_items}
        candidate_ids = {item.chunk["chunk_id"] for item in rerank_candidates}
        final_items = combine_chunk_rrf(
            chunk_ids=candidate_ids,
            chunk_by_id=chunk_by_id,
            rank_maps=final_rank_maps,
            item_maps=final_item_maps,
            weights=BGE_M3_FINAL_RRF_WEIGHTS,
        )
        final_items.sort(key=lambda item: item.score, reverse=True)
        return cap_per_source(
            final_items,
            top_k=top_k,
            per_source_limit=per_source_limit,
            required_source_ids=required_source_ids,
        )

    def format_context(self, retrieved_chunks: list[RetrievedChunk], *, max_chars: int = 12000) -> str:
        """Format retrieved chunks with metadata for debugging and audit output."""
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

    def format_context_for_llm(self, retrieved_chunks: list[RetrievedChunk], *, max_chars: int = 12000) -> str:
        """Format only chunk text for the LLM prompt; metadata stays in debug fields."""
        parts = []
        current_chars = 0
        for item in retrieved_chunks:
            block = f"{item.chunk['text']}\n"
            if current_chars + len(block) > max_chars:
                break
            parts.append(block)
            current_chars += len(block)
        return "\n\n".join(part.strip() for part in parts if part.strip()).strip()

    def _score_chunk(
        self,
        chunk: dict[str, Any],
        *,
        intent: RetrievalIntent,
        time_constraints: dict[str, list[str]],
    ) -> tuple[float, list[str]]:
        text = " ".join([
            chunk.get("title", ""),
            chunk.get("section", ""),
            chunk.get("text", ""),
        ]).lower()
        tokens = Counter(tokenize(text))
        score = 0.0
        reasons: list[str] = []

        for term in intent.query_terms:
            if " " in term:
                if has_phrase(text, term):
                    score += 3.0
                    reasons.append(f"phrase:{term}")
            elif tokens.get(term):
                score += 1.0 + math.log(tokens[term])

        for entity in set(intent.entity_terms):
            if entity and entity in text:
                score += 5.0
                reasons.append(f"entity:{entity}")

        predicate_hints = set(chunk.get("predicate_hints", []))
        for predicate in intent.predicate_terms:
            if predicate in predicate_hints:
                score += 3.0
                reasons.append(f"predicate:{predicate}")

        temporal_fact = chunk.get("temporal_fact")
        if isinstance(temporal_fact, dict):
            temporal_match = temporal_fact_match_kind(temporal_fact, intent)
            if temporal_match:
                score += temporal_match_score(temporal_match)
                reasons.append(f"temporal_fact:{temporal_match}")
            current_match = current_fact_match_kind(temporal_fact, intent)
            if current_match:
                score += current_match_score(current_match)
                reasons.append(f"current_fact:{current_match}")

        filter_score, filter_reasons = score_filter_evidence(
            text,
            predicate_hints,
            intent.filter_conditions,
            weight=2.5,
        )
        score += filter_score
        reasons.extend(filter_reasons)
        entity_match_present = any(entity and entity in text for entity in set(intent.entity_terms))

        if "moon_count" in intent.predicate_terms:
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
        if "distance_from_sun" in intent.predicate_terms:
            distance_score, distance_reasons = score_orbit_order_context(chunk, text)
            score += distance_score
            reasons.extend(distance_reasons)
        ordering_score, ordering_reasons = score_ordering_evidence(
            text,
            intent.ordering_attribute,
            weight=2.5,
            detailed=True,
        )
        score += ordering_score
        reasons.extend(ordering_reasons)
        class_score, class_reasons = score_target_class_evidence(
            text,
            intent.target_class,
            weight=2.0,
        )
        score += class_score
        reasons.extend(class_reasons)

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


def source_rank_map(items: list[SourceScore]) -> dict[str, int]:
    return {item.source_id: index for index, item in enumerate(items, start=1)}


def chunk_rank_map(items: list[RetrievedChunk]) -> dict[str, int]:
    return {item.chunk["chunk_id"]: index for index, item in enumerate(items, start=1)}


def weighted_rrf(
    ranks: dict[str, int | None],
    weights: dict[str, float],
    *,
    k: int = RRF_K,
) -> float:
    score = 0.0
    for name, weight in weights.items():
        rank = ranks.get(name)
        if rank is None:
            continue
        score += weight / (k + rank)
    return score


def combine_chunk_rrf(
    *,
    chunk_ids: set[str],
    chunk_by_id: dict[str, dict[str, Any]],
    rank_maps: dict[str, dict[str, int]],
    item_maps: dict[str, dict[str, RetrievedChunk]],
    weights: dict[str, float],
) -> list[RetrievedChunk]:
    combined = []
    for chunk_id in chunk_ids:
        rank_values = {
            name: rank_map.get(chunk_id)
            for name, rank_map in rank_maps.items()
        }
        score = weighted_rrf(rank_values, weights) * 1000.0
        if score <= 0:
            continue
        reasons = [f"rrf:{score:.4f}"]
        for name in weights:
            rank = rank_values.get(name)
            reasons.append(f"rank_{name}:{rank or 'none'}")
            item = item_maps.get(name, {}).get(chunk_id)
            if item:
                reasons.append(f"{name}_score:{item.score:.4f}")
                reasons.extend(item.reasons[:4])
        combined.append(RetrievedChunk(
            chunk=chunk_by_id[chunk_id],
            score=score,
            reasons=reasons,
        ))
    return combined


def cap_per_source(
    items: list[RetrievedChunk],
    *,
    top_k: int,
    per_source_limit: int,
    required_source_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    required = set(required_source_ids or [])
    counts: dict[str, int] = defaultdict(int)
    selected: list[RetrievedChunk] = []
    selected_chunk_ids: set[str] = set()

    # Relational list questions need at least one available evidence chunk for
    # each candidate/baseline source before the remaining ranked slots are filled.
    for item in items:
        source_id = item.chunk.get("source_id", "")
        if source_id not in required or counts[source_id]:
            continue
        selected.append(item)
        selected_chunk_ids.add(item.chunk.get("chunk_id", ""))
        counts[source_id] += 1
        if len(selected) >= top_k:
            break

    for item in items:
        if len(selected) >= top_k:
            break
        source_id = item.chunk.get("source_id", "")
        if item.chunk.get("chunk_id", "") in selected_chunk_ids:
            continue
        if counts[source_id] >= per_source_limit:
            continue
        selected.append(item)
        counts[source_id] += 1
        if len(selected) >= top_k:
            break
    rank = {
        item.chunk.get("chunk_id", ""): index
        for index, item in enumerate(items)
    }
    selected.sort(key=lambda item: rank.get(item.chunk.get("chunk_id", ""), len(items)))
    return selected


def suppress_near_duplicate_chunks(
    items: list[RetrievedChunk],
    *,
    top_k: int,
    required_source_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    required = set(required_source_ids or [])
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    represented_sources: set[str] = set()
    for item in items:
        source_id = item.chunk.get("source_id", "")
        if source_id not in required or source_id in represented_sources:
            continue
        selected.append(item)
        selected_ids.add(item.chunk.get("chunk_id", ""))
        represented_sources.add(source_id)
        if len(selected) >= top_k:
            break

    for item in items:
        if len(selected) >= top_k:
            break
        if item.chunk.get("chunk_id", "") in selected_ids:
            continue
        if any(text_overlap_ratio(item.chunk.get("text", ""), previous.chunk.get("text", "")) >= 0.80 for previous in selected):
            continue
        selected.append(item)
    rank = {
        item.chunk.get("chunk_id", ""): index
        for index, item in enumerate(items)
    }
    selected.sort(key=lambda item: rank.get(item.chunk.get("chunk_id", ""), len(items)))
    return selected


def routes_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("source_id") != right.get("source_id"):
        return False
    if left.get("route_type") != "section_window" or right.get("route_type") != "section_window":
        return False
    left_start = left.get("word_start")
    left_end = left.get("word_end")
    right_start = right.get("word_start")
    right_end = right.get("word_end")
    if None in {left_start, left_end, right_start, right_end}:
        return False
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    shorter = min(left_end - left_start, right_end - right_start)
    return shorter > 0 and overlap / shorter >= 0.10


def text_overlap_ratio(left: str, right: str) -> float:
    left_tokens = Counter(tokenize(left))
    right_tokens = Counter(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((left_tokens & right_tokens).values())
    return overlap / min(sum(left_tokens.values()), sum(right_tokens.values()))


def has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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

    if "moon" in section or "satellite" in section:
        score += 5.0
        reasons.append("moon_section")
    if any(
        word in text
        for word in (
            "known moons",
            "confirmed moons",
            "natural satellites",
            "known natural satellites",
            "confirmed satellites",
        )
    ):
        score += 5.0
        reasons.append("moon_count_terms")
    if re_moon_count_claim(text):
        score += 8.0
        reasons.append("moon_count_claim")
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
    if (
        re.search(
            r"\b(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d+(?:st|nd|rd|th))\s+planet|orbit(?:s|al|ing)?|semi-major\s+axis|distance\s+from\s+(?:the\s+)?sun)\b",
            text,
        )
        or re.search(r"\b\d+(?:\.\d+)?\s*au\b", text)
    ):
        score += 3.0
        reasons.append("planet_orbit_context")
    return score, reasons


def re_moon_count_claim(text: str) -> bool:
    tokens = count_claim_tokens(text)
    action_terms = {"has", "have", "had", "include", "includes", "included", "possess", "possesses"}
    moon_terms = {"moon", "moons", "satellite", "satellites"}

    for index, token in enumerate(tokens):
        if token not in action_terms:
            continue
        window = tokens[index + 1:index + 15]
        if _window_contains_count_before_moon(window, moon_terms):
            return True

    return bool(
        re.search(
            r"\b[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3}'s\s+"
            r"(?:\d[\d,]*|[a-z]+(?:[-\s]+[a-z]+){0,5})\s+"
            r"(?:\w+\s+){0,5}(?:moon|moons|satellite|satellites)\b",
            text,
        )
    )


def count_claim_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9, -]+", " ", text.lower())
    normalized = normalized.replace("-", " ")
    return re.findall(r"\d[\d,]*|[a-z]+", normalized)


def _window_contains_count_before_moon(
    tokens: list[str],
    moon_terms: set[str],
    *,
    max_gap: int = 10,
) -> bool:
    for moon_index, token in enumerate(tokens):
        if token not in moon_terms:
            continue
        start = max(0, moon_index - max_gap)
        before_moon = tokens[start:moon_index]
        for count_start in range(len(before_moon)):
            for count_end in range(count_start + 1, min(len(before_moon), count_start + 6) + 1):
                if parse_count_phrase(before_moon[count_start:count_end]) is not None:
                    return True
    return False


def parse_count_phrase(tokens: list[str]) -> int | None:
    if not tokens:
        return None
    phrase = " ".join(tokens)
    if len(tokens) == 1 and re.fullmatch(r"\d[\d,]*", tokens[0]):
        return int(tokens[0].replace(",", ""))
    if any(re.fullmatch(r"\d[\d,]*", token) for token in tokens):
        return None
    if parse_number_words is not None:
        try:
            value = parse_number_words(phrase)
            if isinstance(value, int) and value >= 0:
                return value
        except Exception:
            pass
    return fallback_parse_count_words(tokens)


def fallback_parse_count_words(tokens: list[str]) -> int | None:
    ones = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    allowed = set(ones) | set(tens) | {"hundred", "and"}
    if any(token not in allowed for token in tokens):
        return None
    total = 0
    current = 0
    used_number = False
    for token in tokens:
        if token == "and":
            continue
        if token in ones:
            current += ones[token]
            used_number = True
        elif token in tens:
            current += tens[token]
            used_number = True
        elif token == "hundred":
            if current == 0:
                current = 1
            current *= 100
            used_number = True
    total += current
    if not used_number or total > 999:
        return None
    return total

"""Embedding cache and scoring helpers for RAG retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EMBEDDING_PROVIDER = "bge-m3"
DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODEL = DEFAULT_BGE_M3_MODEL
DEFAULT_EMBEDDINGS_PATH = Path("data/rag_sources/rag_index/chunk_embeddings_bge_m3.jsonl")
DEFAULT_BGE_BASE_EMBEDDINGS_PATH = Path("data/rag_sources/rag_index/chunk_embeddings_bge_base.jsonl")
DEFAULT_OPENAI_EMBEDDINGS_PATH = Path("data/rag_sources/rag_index/chunk_embeddings_openai.jsonl")
EMBEDDING_PROVIDERS = {"bge-m3", "local", "openai"}

_LOCAL_MODEL_CACHE: dict[str, Any] = {}
_BGE_M3_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class EmbeddingFeatures:
    dense: list[float]
    sparse: dict[str, float] | None = None


@dataclass
class EmbeddingRecord:
    chunk_id: str
    provider: str
    model: str
    text_hash: str
    embedding: list[float]
    sparse_weights: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "chunk_id": self.chunk_id,
            "provider": self.provider,
            "model": self.model,
            "text_hash": self.text_hash,
            "embedding": self.embedding,
        }
        if self.sparse_weights:
            data["sparse_weights"] = self.sparse_weights
        return data


class EmbeddingIndex:
    def __init__(self, path: str | Path = DEFAULT_EMBEDDINGS_PATH) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Embedding cache not found: {self.path}. "
                "Run: python scripts\\build_rag_embeddings.py"
            )
        self.records = load_embedding_records(self.path)
        if not self.records:
            raise ValueError(f"Embedding cache is empty: {self.path}")
        first_record = next(iter(self.records.values()))
        self.provider = first_record.provider
        self.model = first_record.model
        self._query_cache: dict[tuple[str, bool], EmbeddingFeatures] = {}

    def get(self, chunk_id: str) -> list[float] | None:
        record = self.records.get(chunk_id)
        return record.embedding if record else None

    def get_sparse(self, chunk_id: str) -> dict[str, float] | None:
        record = self.records.get(chunk_id)
        return record.sparse_weights if record else None

    def embed_query(self, text: str) -> list[float]:
        return self.encode_query(text).dense

    def encode_query(self, text: str) -> EmbeddingFeatures:
        cache_key = (text, False)
        if cache_key not in self._query_cache:
            self._query_cache[cache_key] = embed_text_features(
                [text],
                provider=self.provider,
                model=self.model,
                return_sparse=self.provider == "bge-m3",
            )[0]
        return self._query_cache[cache_key]


def build_chunk_embedding_records(
    chunks: list[dict[str, Any]],
    *,
    existing_records: dict[str, EmbeddingRecord],
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model: str | None = None,
    batch_size: int = 64,
    refresh: bool = False,
) -> list[EmbeddingRecord]:
    provider = normalize_provider(provider)
    model = model or default_model_for_provider(provider)
    records_by_chunk_id = {
        chunk_id: record
        for chunk_id, record in existing_records.items()
        if record.provider == provider and record.model == model
    }
    pending: list[tuple[dict[str, Any], str, str]] = []
    output_records: list[EmbeddingRecord] = []

    for chunk in chunks:
        text = embedding_text_for_chunk(chunk)
        text_hash = hash_text(text)
        existing = records_by_chunk_id.get(chunk["chunk_id"])
        if existing and existing.text_hash == text_hash and not refresh:
            output_records.append(existing)
            continue
        pending.append((chunk, text, text_hash))

    for batch in batched(pending, batch_size):
        texts = [item[1] for item in batch]
        features = embed_text_features(
            texts,
            provider=provider,
            model=model,
            return_sparse=provider == "bge-m3",
        )
        for (chunk, _text, text_hash), feature in zip(batch, features):
            output_records.append(EmbeddingRecord(
                chunk_id=chunk["chunk_id"],
                provider=provider,
                model=model,
                text_hash=text_hash,
                embedding=feature.dense,
                sparse_weights=feature.sparse,
            ))

    order = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}
    output_records.sort(key=lambda record: order.get(record.chunk_id, 10**9))
    return output_records


def embedding_text_for_chunk(chunk: dict[str, Any], *, max_chars: int = 12000) -> str:
    text = "\n".join([
        str(chunk.get("title", "")),
        str(chunk.get("section", "")),
        str(chunk.get("text", "")),
    ]).strip()
    return text[:max_chars]


def embed_texts(
    texts: list[str],
    *,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model: str | None = None,
) -> list[list[float]]:
    return [
        feature.dense
        for feature in embed_text_features(texts, provider=provider, model=model)
    ]


def embed_text_features(
    texts: list[str],
    *,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model: str | None = None,
    return_sparse: bool = False,
) -> list[EmbeddingFeatures]:
    if not texts:
        return []
    provider = normalize_provider(provider)
    model = model or default_model_for_provider(provider)
    if provider == "bge-m3":
        return embed_text_features_bge_m3(texts, model=model, return_sparse=return_sparse)
    if provider == "local":
        return [EmbeddingFeatures(dense=embedding) for embedding in embed_texts_local(texts, model=model)]
    if provider == "openai":
        return [EmbeddingFeatures(dense=embedding) for embedding in embed_texts_openai(texts, model=model)]
    raise ValueError(f"Unsupported embedding provider: {provider}")


def embed_text_features_bge_m3(
    texts: list[str],
    *,
    model: str,
    return_sparse: bool,
) -> list[EmbeddingFeatures]:
    encoder = get_bge_m3_model(model)
    output = encoder.encode(
        texts,
        return_dense=True,
        return_sparse=return_sparse,
        return_colbert_vecs=False,
    )
    dense_vectors = output.get("dense_vecs")
    if dense_vectors is None:
        dense_vectors = output.get("dense")
    sparse_vectors = output.get("lexical_weights")
    if sparse_vectors is None:
        sparse_vectors = output.get("sparse")
    if dense_vectors is None:
        raise RuntimeError("BGE-M3 encode output did not include dense vectors.")

    features = []
    for index, dense in enumerate(dense_vectors):
        sparse = None
        if return_sparse and sparse_vectors is not None:
            sparse = normalize_sparse_weights(sparse_vectors[index])
        features.append(EmbeddingFeatures(
            dense=[float(value) for value in dense],
            sparse=sparse,
        ))
    return features


def bge_m3_colbert_scores(
    query: str,
    passages: list[str],
    *,
    model: str = DEFAULT_BGE_M3_MODEL,
    max_passage_length: int = 512,
) -> list[float]:
    if not passages:
        return []
    encoder = get_bge_m3_model(model)
    pairs = [[query, passage] for passage in passages]
    output = encoder.compute_score(pairs, max_passage_length=max_passage_length)
    if isinstance(output, dict):
        scores = output.get("colbert")
        if scores is None:
            scores = output.get("colbert+sparse+dense")
    else:
        scores = output
    if scores is None:
        raise RuntimeError("BGE-M3 compute_score output did not include ColBERT scores.")
    try:
        return [float(score) for score in scores]
    except TypeError:
        return [float(scores)]


def get_bge_m3_model(model: str) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            "BGE-M3 embeddings require FlagEmbedding. "
            "Install it with: pip install FlagEmbedding"
        ) from exc

    if model not in _BGE_M3_MODEL_CACHE:
        _BGE_M3_MODEL_CACHE[model] = BGEM3FlagModel(model, use_fp16=False)
    return _BGE_M3_MODEL_CACHE[model]


def embed_texts_local(texts: list[str], *, model: str) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    if model not in _LOCAL_MODEL_CACHE:
        _LOCAL_MODEL_CACHE[model] = SentenceTransformer(model)
    encoder = _LOCAL_MODEL_CACHE[model]
    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [[float(value) for value in vector] for vector in embeddings]


def embed_texts_openai(texts: list[str], *, model: str) -> list[list[float]]:
    from src.models import _get_openai_client

    response = _get_openai_client().embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def sparse_dot(left: dict[str, float] | None, right: dict[str, float] | None) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(token_id, 0.0) for token_id, weight in left.items())


def load_embedding_records(path: str | Path) -> dict[str, EmbeddingRecord]:
    target = Path(path)
    if not target.exists():
        return {}
    records: dict[str, EmbeddingRecord] = {}
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records[data["chunk_id"]] = EmbeddingRecord(
                chunk_id=data["chunk_id"],
                provider=data.get("provider", "openai"),
                model=data["model"],
                text_hash=data["text_hash"],
                embedding=[float(value) for value in data["embedding"]],
                sparse_weights=normalize_sparse_weights(data.get("sparse_weights")),
            )
    return records


def write_embedding_records(path: str | Path, records: Iterable[EmbeddingRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_provider(provider: str) -> str:
    value = provider.lower().strip()
    if value == "bge-m3-rrf":
        value = "bge-m3"
    if value not in EMBEDDING_PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(sorted(EMBEDDING_PROVIDERS))}")
    return value


def default_model_for_provider(provider: str) -> str:
    provider = normalize_provider(provider)
    if provider == "bge-m3":
        return DEFAULT_BGE_M3_MODEL
    if provider == "local":
        return DEFAULT_LOCAL_EMBEDDING_MODEL
    return DEFAULT_OPENAI_EMBEDDING_MODEL


def default_embeddings_path_for_provider(provider: str) -> Path:
    provider = normalize_provider(provider)
    if provider == "bge-m3":
        return DEFAULT_EMBEDDINGS_PATH
    if provider == "local":
        return DEFAULT_BGE_BASE_EMBEDDINGS_PATH
    return DEFAULT_OPENAI_EMBEDDINGS_PATH


def normalize_sparse_weights(value: Any) -> dict[str, float] | None:
    if not value:
        return None
    if isinstance(value, dict):
        return {
            str(token_id): float(weight)
            for token_id, weight in value.items()
            if float(weight) != 0.0
        }
    return {
        str(index): float(weight)
        for index, weight in enumerate(value)
        if float(weight) != 0.0
    }


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

"""Embedding cache and cosine similarity helpers for RAG retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.models import _get_openai_client


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDINGS_PATH = Path("data/rag_sources/rag_index/chunk_embeddings.jsonl")


@dataclass
class EmbeddingRecord:
    chunk_id: str
    model: str
    text_hash: str
    embedding: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "model": self.model,
            "text_hash": self.text_hash,
            "embedding": self.embedding,
        }


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
        self.model = next(iter(self.records.values())).model
        self._query_cache: dict[str, list[float]] = {}

    def get(self, chunk_id: str) -> list[float] | None:
        record = self.records.get(chunk_id)
        return record.embedding if record else None

    def embed_query(self, text: str) -> list[float]:
        if text not in self._query_cache:
            self._query_cache[text] = embed_texts([text], model=self.model)[0]
        return self._query_cache[text]


def build_chunk_embedding_records(
    chunks: list[dict[str, Any]],
    *,
    existing_records: dict[str, EmbeddingRecord],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    refresh: bool = False,
) -> list[EmbeddingRecord]:
    records_by_chunk_id = {
        chunk_id: record
        for chunk_id, record in existing_records.items()
        if record.model == model
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
        embeddings = embed_texts(texts, model=model)
        for (chunk, _text, text_hash), embedding in zip(batch, embeddings):
            output_records.append(EmbeddingRecord(
                chunk_id=chunk["chunk_id"],
                model=model,
                text_hash=text_hash,
                embedding=embedding,
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


def embed_texts(texts: list[str], *, model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    if not texts:
        return []
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
                model=data["model"],
                text_hash=data["text_hash"],
                embedding=[float(value) for value in data["embedding"]],
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


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

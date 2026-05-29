"""Default filesystem paths for the PDF-only RAG corpus."""

from __future__ import annotations

from pathlib import Path


PDF_SOURCE_ROOT = Path("data/rag_pdf_sources")
PDF_RAW_DIR = PDF_SOURCE_ROOT / "pdf_raw"
PDF_TEXT_DIR = PDF_SOURCE_ROOT / "pdf_text"
PDF_INDEX_DIR = PDF_SOURCE_ROOT / "rag_index"
PDF_DOCUMENTS_PATH = PDF_INDEX_DIR / "documents.jsonl"
PDF_CHUNKS_PATH = PDF_INDEX_DIR / "chunks.jsonl"
PDF_BGE_M3_EMBEDDINGS_PATH = PDF_INDEX_DIR / "chunk_embeddings_bge_m3.jsonl"
PDF_BGE_BASE_EMBEDDINGS_PATH = PDF_INDEX_DIR / "chunk_embeddings_bge_base.jsonl"
PDF_OPENAI_EMBEDDINGS_PATH = PDF_INDEX_DIR / "chunk_embeddings_openai.jsonl"
PDF_ROUTING_UNITS_PATH = PDF_INDEX_DIR / "routing_units.jsonl"
PDF_ROUTING_EMBEDDINGS_PATH = PDF_INDEX_DIR / "routing_embeddings_bge_m3.jsonl"

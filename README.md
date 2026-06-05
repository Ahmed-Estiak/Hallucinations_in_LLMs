# Hallucinations in LLMs: Astronomy KG and RAG Benchmark

An astronomy question-answering benchmark for studying how grounding affects
LLM hallucination. The project compares:

- vanilla LLM answers;
- knowledge-graph-assisted LLM answers;
- web-source RAG + LLM answers;
- PDF-only RAG + LLM answers;
- multiple lexical, dense, sparse, hybrid, hierarchical, and temporal retrieval strategies.

The current RAG implementation is designed around a small evaluated question
set, while keeping retrieval and scoring rules generic for astronomy sources.

## Current Status

The implemented pipeline supports:

- web ingestion from Wikipedia and generic HTML pages;
- copyable-text PDF ingestion with page/layout diagnostics;
- single-, two-, and three-column PDF text ordering;
- table-region preservation and cross-page table continuation markers;
- overlapping section-aware chunks and structured astronomy facts;
- BGE-M3 dense, sparse, RRF, and ColBERT retrieval;
- BGE-base and OpenAI embedding retrieval;
- lexical auto-source and global retrieval;
- hierarchical source routing before chunk retrieval;
- temporal-first retrieval for dated moon-count questions;
- class-wide moon-count evidence coverage for comparison/list questions;
- OpenAI and Gemini RAG answer generation;
- detailed CSV timing, retrieval, prompt, fallback, and answer audit logs.

The main evaluated development questions are:

| ID | Question |
|---|---|
| Q9 | Which dwarf planet located in the Kuiper Belt was discovered first? |
| Q11 | Which planets orbit beyond Earth yet have fewer moons than Jupiter? |
| Q15 | As of November 2021, how many confirmed moons did Saturn have? |

## Architecture

```text
source URLs / PDFs
        |
        v
cleaning and layout-aware extraction
        |
        v
documents.jsonl
        |
        v
overlapping chunks + structured temporal/current facts
        |
        +----------------------+
        |                      |
        v                      v
embedding caches          routing units/index
        |                      |
        +----------+-----------+
                   |
                   v
retrieval method + evidence gates
                   |
                   v
compact resolved facts + supporting raw chunks
                   |
                   v
LLM prompt -> answer -> evaluator -> CSV audit
```

Web and PDF corpora are intentionally separate. A PDF-only run does not mix web
evidence into its retrieval context.

## Retrieval Methods

The recommended comparison methods are:

| Method | Description |
|---|---|
| `hierarchical-bge-m3-rrf` | BGE-M3 route ranking, source shortlist, selected-source chunk RRF, and ColBERT reranking. |
| `bge-m3-rrf` | Full-chunk BGE-M3 dense + sparse + lexical RRF with ColBERT reranking. |
| `bge-base-rrf` | BGE-base dense retrieval combined with lexical ranking through RRF. |
| `openai-embedding-rrf` | OpenAI dense embedding retrieval combined with lexical ranking through RRF. |
| `auto-source` | Lexical source selection followed by lexical chunk ranking. |
| `global` | Lexical ranking directly across the available chunks. |

Additional legacy/debug modes include `vector` and `hybrid`.

### Temporal-First Behavior

Temporal-first retrieval is enabled by default for questions that contain:

- a date/as-of constraint;
- a moon-count predicate;
- a target astronomy entity.

It first resolves structured dated count anchors, then scans and ranks only
relevant dated raw support chunks. Other temporal questions continue through
their requested retrieval method.

For method-isolation audits, disable it with:

```powershell
--disable-temporal-first
```

## Repository Layout

```text
.
|-- data/
|   |-- qa_92.json                     # benchmark questions and expected answers
|   |-- rag_sources/                   # web corpus manifests, text, and indexes
|   `-- rag_pdf_sources/               # PDF-only corpus, extracted text, and indexes
|-- docs/                              # KG and integration design notes
|-- external/                          # optional external Wikipedia cleaner
|-- reports/
|   |-- final/                         # final benchmark/comparison reports
|   `-- debug/                         # temporary audit, timing, and retrieval reports
|-- scripts/                           # ingestion, index building, audit, and debug CLIs
|   `-- smoke/                         # targeted provider/API smoke tests
|-- src/
|   |-- rag/                           # RAG extraction, indexing, retrieval, and evidence logic
|   |-- evaluator.py                   # answer-format-aware evaluation
|   |-- kg_*.py                        # knowledge graph retrieval/reasoning
|   `-- *_models.py                    # OpenAI/Gemini integrations
|-- tests/                             # regression and retrieval tests
|-- main.py                            # vanilla LLM benchmark
|-- main_kg.py                         # KG + LLM benchmark
`-- main_rag.py                        # web RAG + LLM benchmark
```

Generated PDF files, extracted text, embedding caches, and report outputs are
excluded from Git through `.gitignore`.

## Requirements

- Windows PowerShell commands are shown below.
- Python 3.12 is the currently tested environment.
- A CUDA-capable GPU is strongly recommended for BGE-M3.
- OpenAI/Gemini keys are needed only when calling those providers.

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```text
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

The `.env` file is Git ignored.

## Quick Start

Run the vanilla benchmark:

```powershell
.\.venv\Scripts\python.exe main.py
```

Run the KG-assisted benchmark:

```powershell
.\.venv\Scripts\python.exe main_kg.py
```

Run selected web RAG questions with the default retrieval mode:

```powershell
.\.venv\Scripts\python.exe main_rag.py --ids 9 11 15
```

Preview retrieval context without making an LLM call:

```powershell
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode hierarchical-bge-m3-rrf
```

Smoke-test one question through vanilla and KG-grounded OpenAI/Gemini paths:

```powershell
.\.venv\Scripts\python.exe scripts\smoke\test_single_question.py --id 5
```

This smoke test calls both provider APIs and incurs API usage.

## Web RAG Pipeline

Web sources are configured in:

```text
data/rag_sources/sources_master.json
```

Each source includes a stable `source_id`, URL, cleaner, domain, and trust
metadata. The cleaner can be `wikipedia` or `generic_html`.

Ingest or refresh web sources:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_rag_sources.py
.\.venv\Scripts\python.exe scripts\ingest_rag_sources.py --refresh
```

Build chunks and routing units:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_index.py
.\.venv\Scripts\python.exe scripts\build_rag_routing_index.py
```

Build BGE-M3 chunk and routing embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --provider bge-m3
.\.venv\Scripts\python.exe scripts\build_rag_routing_embeddings.py
```

Build BGE-base embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --provider local
```

## PDF-Only RAG Pipeline

Place copyable-text PDFs in:

```text
data/rag_pdf_sources/pdf_raw/
```

The raw PDF files are Git ignored. The folder contains a tracked `.gitkeep`.

Extract PDF text and layout diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_rag_pdfs.py --refresh
```

Build PDF chunks, page signals, and routing units:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_pdf_index.py
```

Build PDF BGE-M3 embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --source-set pdf --provider bge-m3 --batch-size 32
.\.venv\Scripts\python.exe scripts\build_rag_routing_embeddings.py --source-set pdf --batch-size 16
```

Build PDF BGE-base embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --source-set pdf --provider local --batch-size 32
```

Build PDF OpenAI embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --source-set pdf --provider openai
```

The OpenAI embedding command sends chunk text to the OpenAI embeddings API and
incurs API usage. The runtime retriever does not silently build a missing
temporal OpenAI candidate cache; it logs the missing/incomplete cache and uses
its configured fallback.

## Run RAG Method Audits

Run all six recommended PDF retrieval methods over Q9, Q11, and Q15 through
OpenAI:

```powershell
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe scripts\run_rag_llm_method_matrix.py --source-set pdf --questions 9 11 15 --methods hierarchical-bge-m3-rrf bge-m3-rrf bge-base-rrf openai-embedding-rrf auto-source global --providers openai --output reports\debug\pdf\openai_6_methods_3_questions.csv
```

Run a fair method-isolation audit without temporal-first interception:

```powershell
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe scripts\run_rag_llm_method_matrix.py --source-set pdf --questions 15 --methods hierarchical-bge-m3-rrf bge-m3-rrf bge-base-rrf openai-embedding-rrf auto-source global --providers openai --disable-temporal-first --output reports\debug\pdf\q15_without_temporal_first.csv
```

Warm the persistent retriever before measured requests:

```powershell
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe scripts\run_rag_llm_method_matrix.py --source-set pdf --questions 11 --methods hierarchical-bge-m3-rrf --providers openai --warmup-retriever --output reports\debug\pdf\q11_hierarchical_warm.csv
```

Warm-up preloads BGE-M3/CUDA, vector indexes, lexical features, and ColBERT. It
does not remove retrieval stages. Warm-up time is recorded separately from the
measured query time.

## Performance Optimizations

The retriever keeps the following caches within a running Python process:

- BGE-M3 model and query representations;
- normalized route/chunk lexical features;
- NumPy dense embedding matrices;
- sparse inverted posting indexes;
- ColBERT query and passage multi-vector representations;
- OpenAI/Gemini clients.

Starting a new Python command creates a new process and requires cold startup
again. For production-style latency, keep the retriever process alive or use
`--warmup-retriever` before a multi-query audit.

## Audit Output

`run_rag_llm_method_matrix.py` writes one CSV row per question/method pair. It
includes:

- requested and executed retrieval methods;
- fallback status and reason;
- retrieved chunk IDs and prompt token estimate;
- OpenAI/Gemini answers and evaluator results;
- total retrieval and provider-call timing;
- route, chunk, RRF, ColBERT, moon-gate, and temporal-first timing;
- OpenAI temporal embedding cache status and missing count.

Final benchmark and comparison outputs belong in `reports/final/`. Temporary
retrieval audits, prompt inspections, generated fact tables, and timing reports
belong in `reports/debug/`.

## Tests

Run the full current regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rag_embedding_optimizations tests.test_rag_temporal_evidence tests.test_rag_generic_scoring tests.test_pdf_page_filter tests.test_pdf_extractor_layout
```

The current suite covers:

- generic retrieval scoring and anti-overfitting behavior;
- temporal/current moon-count evidence;
- PDF page filtering and layout extraction;
- dense/sparse vectorized scoring;
- retrieval cache behavior.

## Known Limitations

- PDF ingestion expects a usable text layer; OCR-only PDFs are not yet a primary target.
- Table extraction is heuristic and preserves table-like rows rather than fully reconstructing every table schema.
- Temporal-first retrieval currently focuses on dated moon-count questions.
- Number-word count parsing focuses on standard English number expressions; vague quantities such as `several` or `a dozen` are not treated as exact counts.
- OpenAI and Gemini answers depend on provider availability, latency, and API credentials.
- In-memory retrieval caches are rebuilt when a new Python process starts.

## Rebuilding After Source Changes

After web text, PDF text, chunking, or structured-fact extraction changes,
rebuild downstream artifacts in this order:

```text
ingest sources/PDFs
-> build chunks and routing units
-> build chunk embeddings
-> build routing embeddings
-> preview/audit retrieval
-> run LLM matrix
```

Embedding builders are cache-first and reuse unchanged records unless
`--refresh` is supplied.

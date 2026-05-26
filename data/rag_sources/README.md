# RAG Sources

This folder stores raw and cleaned source materials for the RAG prototype.

Recommended initial questions:
- Q9: Which dwarf planet located in the Kuiper Belt was discovered first?
- Q11: Which planets orbit beyond Earth yet have fewer moons than Jupiter?
- Q15: As of November 2021, how many confirmed moons did Saturn have?

Subfolders:
- `docs_raw/`: copied plain-text versions of your source documents
- `docs_clean/`: cleaned text generated from web/PDF sources
- `web_raw/`: cached raw HTML/API responses used to reproduce cleaned outputs
- `pdf_raw/`: exported PDF versions of the same documents
- `notes/`: optional planning or source notes

Suggested file naming:
- `rag_doc_01_pluto_kuiper`
- `rag_doc_02_planets_moons`
- `rag_doc_03_saturn_moons_2021`
- `rag_doc_04_mixed_astronomy_facts`
- `rag_doc_05_temporal_astronomy_notes`

Recommended local Python:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

If `.venv` is missing, install Python 3.12 and run:

```powershell
.\scripts\setup_rag_gpu_env.ps1
```

Wikipedia cleaning:

```powershell
.\.venv\Scripts\python.exe scripts\clean_wikipedia_source.py --url "https://en.wikipedia.org/wiki/Saturn" --source-id wiki_saturn
```

The script uses `external/Wikipedia_text_extractor` by default. Set
`WIKIPEDIA_TEXT_EXTRACTOR_DIR` or pass `--cleaner-dir` if the cleaner lives in a
different folder.

Q9 RAG+LLM vertical slice:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_rag_sources.py --sources data\rag_sources\sources_master.json
.\.venv\Scripts\python.exe scripts\build_rag_index.py
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py
.\.venv\Scripts\python.exe scripts\build_rag_routing_index.py
.\.venv\Scripts\python.exe scripts\build_rag_routing_embeddings.py --batch-size 16
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode hierarchical-bge-m3-rrf
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode bge-m3-rrf
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode bge-base-rrf
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode auto-source
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode vector
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode hybrid
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 15 --retrieval-mode auto-source
.\.venv\Scripts\python.exe main_rag.py --ids 9 11 15 --retrieval-mode bge-m3-rrf
.\.venv\Scripts\python.exe main_rag.py --ids 9 11 15 --retrieval-mode auto-source
.\.venv\Scripts\python.exe main_rag.py --ids 9 11 15 --retrieval-mode vector
.\.venv\Scripts\python.exe main_rag.py --ids 9 11 15 --retrieval-mode hybrid
```

Embeddings default to BGE-M3 with FlagEmbedding:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --provider bge-m3 --model BAAI/bge-m3
```

This writes `data\rag_sources\rag_index\chunk_embeddings_bge_m3.jsonl` with
dense vectors and sparse lexical weights. The first local run downloads the
model; later runs reuse the local model cache and the chunk embedding cache.

The default retrieval mode is `hierarchical-bge-m3-rrf`. It searches a
smaller routing-unit BGE-M3 dense+sparse+lexical index first, selects sources,
then ranks only selected-source original chunks and reranks top candidates
with BGE-M3 ColBERT scores. Routing units cover the original cleaned text with
overlapping coarse windows, plus metadata and structured-fact routes. Metadata
routes contribute only a capped source-routing prior; content and structured
fact routes supply the primary source score. Lexical filter, ordering, and
target-class bonuses are derived from parsed query intent, so an unrelated
location or ordering attribute does not receive a question-specific boost.

The automatic fallback order is:

```text
hierarchical-bge-m3-rrf
 -> bge-m3-rrf
 -> bge-base-rrf
 -> openai-embedding-rrf
 -> auto-source
 -> global
```

`bge-m3-rrf` is the full-chunk BGE-M3 baseline. Hierarchical retrieval falls
back to it when routing is unavailable or evidence is weak. Comparison
reduction is reported as a diagnostic metric only; low savings do not replace
an otherwise useful hierarchical result. When automatic fallback reaches
`openai-embedding-rrf`, it prints a warning, builds any missing OpenAI cache
records, and sends query embeddings to the OpenAI API.

Build the first local fallback cache:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --provider local --model BAAI/bge-base-en-v1.5
```

This writes `data\rag_sources\rag_index\chunk_embeddings_bge_base.jsonl`.
To use OpenAI embeddings explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_embeddings.py --provider openai --model text-embedding-3-small
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode openai-embedding-rrf
```

Or let the selected retrieval command build the missing cache explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\preview_rag_context.py --id 9 --retrieval-mode openai-embedding-rrf --build-missing-embeddings
.\.venv\Scripts\python.exe main_rag.py --ids 9 11 15 --retrieval-mode openai-embedding-rrf --build-missing-embeddings
```

The explicit command is useful for preparing the paid OpenAI cache before a
benchmark. Automatic fallback also builds it if the preceding local retrieval
methods cannot supply adequate evidence, and prints a cost warning first.

To build or refresh the hierarchical routing files:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_routing_index.py
.\.venv\Scripts\python.exe scripts\build_rag_routing_embeddings.py --batch-size 16
```

Preview output reports routing units scored, selected-source candidate chunks,
full-chunk baseline count, comparison reduction, selected routes, and any
fallback reason.

For satellite discovery tables, the index builder adds structured count fact
chunks such as `As of November 2021, Saturn had 83 confirmed moons...` from the
flattened source table. This keeps temporal moon-count questions grounded
without asking the LLM to count table rows itself.

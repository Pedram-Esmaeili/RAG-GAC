## Technical Design: Medicall Chatbot API

This document explains the system architecture, Retrieval-Augmented Generation (RAG) pipeline, Generate-Augmented Cache (GAC), hybrid retrieval, rationale for using LlamaIndex, and concrete optimization strategies as implemented in this repository.

### Key components (by file)
- `main.py`: FastAPI app init, CORS, router wiring, conditional indexing bootstrap.
- `app/api/router_factory.py`: Central router assembly.
- `app/api/routes/chat_routes.py`: HTTP `POST /query`, WebSocket `/stream/{connection_id}` with `ping`/`stop`/`chat_request` lifecycle, task management via `socket_manager`.
- `app/services/rag_service.py`: Core RAG pipeline (query expansion → hybrid search → generation → optional voice summarization), GAC integration.
- `app/services/rag_chatbot_service.py`: Chat orchestration over sockets, cancellation semantics.
- `app/services/embedding_service.py`: Embedding model configuration for retrieval and indexing.
- `app/services/gac_service.py`: Semantic cache implementation backed by SQLite (see `app/db/gac_cache`).
- `app/utils/config.py`: Centralized configuration via environment variables.
- `scripts/index_documents.py`: Index builder populating Chroma with document chunks.

---

## Architecture overview

Request flow (HTTP):
1. Client calls `POST /query` with `query` and `user_id`.
2. `RAGService.ask_async` executes:
   - Optional GAC lookup
   - Optional query expansion
   - Hybrid retrieval (vector via Chroma + BM25)
   - OpenAI generation over retrieved contexts
   - Optional voice-oriented summarization
   - GAC write-through for cacheable outputs
3. Response returns `answer`, `usage`, `latency` and `contexts` with scores.

Streaming flow (WebSocket):
1. Client connects `GET /stream/{connection_id}`.
2. Sends messages with `type`: `ping` | `stop` | `chat_request`.
3. `rag_chatbot_service.py` handles lifecycle, cancellation and emits structured status/events.

Data flow:
- Source docs: `data/documents/*.txt`
- Vector store: Chroma persistent collection at `app/db/chroma_db`
- GAC store: `app/db/gac_cache/cache.db`

---

## Generate-Augmented Cache (GAC)

GAC accelerates repeated or semantically similar queries by caching final answers (and optionally usage/contexts) keyed by a semantic fingerprint rather than exact strings.

Implementation highlights:
- Enabled via `ENABLE_GAC=true` (see `app/utils/config.py`).
- Backed by SQLite (`app/db/gac_cache/cache.db`).
- Similarity threshold: `GAC_SIMILARITY_THRESHOLD` (default `0.92`).
- Capacity and TTL: `GAC_MAX_CACHE_SIZE_GB`, `GAC_TTL_HOURS`.
- Integrated in `RAGService.ask_async`:
  - On cache hit: Return cached answer quickly; for voice mode, perform a short summarization pass to optimize TTS delivery.
  - On cache miss: Run full pipeline, then write-through to cache when appropriate (text responses only by default).

Tuning guidance:
- Lower `GAC_SIMILARITY_THRESHOLD` if near-duplicate queries are missing cache hits; raise it to avoid over-matching.
- Increase `GAC_TTL_HOURS` for long-lived corpora; reduce if documents change frequently.
- Keep `GAC_MAX_CACHE_SIZE_GB` aligned with disk constraints; monitor hit ratio and eviction.
- Consider embedding model alignment: GAC should use the same embeddings as retrieval for consistent similarity behavior.

Operational notes:
- Voice requests are typically not cached verbatim; instead, the text answer may be summarized per request for better speech cadence.
- Provide admin endpoints if you need runtime cache stats or invalidation (see `RAGService.get_cache_stats`, `clear_cache`).

---

## Hybrid retrieval (BM25 + Vector)

Objective: Improve recall and precision by combining sparse (BM25) and dense (vector) retrieval.

Pipeline in `RAGService`:
1. Optional query expansion creates a small set of semantically related queries.
2. For each variant:
   - Vector search using LlamaIndex `VectorIndexRetriever` on Chroma-backed `VectorStoreIndex`.
   - BM25 search over tokenized document text cached in-memory.
3. Results are merged per-document/node with a blended score:
   - `combined_score = alpha * vector_score + (1 - alpha) * normalized_bm25`.
4. Top-K results are passed to the LLM for grounded generation.

Key parameters (in `config.py`):
- `TOP_K`: final number of contexts sent to the LLM.
- `BM25_TOP_K`: candidates from BM25 before merge.
- `HYBRID_SEARCH_ALPHA`: blending factor [0..1]; 0 = BM25 only, 1 = vector only.

Tuning strategies:
- Start with `alpha=0.5`. If queries are keyword-heavy, shift toward BM25 (e.g., 0.3). If semantic similarity matters more, raise toward vector (e.g., 0.7).
- Ensure BM25 indexing is up-to-date; it is rebuilt from Chroma documents at `RAGService` init.
- Increase `BM25_TOP_K` modestly to capture long-tail matches, then rely on the blended score to rank.

---

## Why LlamaIndex

LlamaIndex provides a high-level, composable interface for retrieval and indexing without locking the system into a single vector store or embedding backend. In this project it:
- Smoothly integrates with Chroma via `ChromaVectorStore` and `VectorStoreIndex`.
- Provides `VectorIndexRetriever` with a well-defined scoring API (`NodeWithScore`).
- Keeps embedding configuration unified through `Settings.embed_model`, ensuring retrieval is aligned with indexing.
- Offers a consistent path to evolve the system (e.g., multi-vector retrievers, routers, re-ranking) with minimal refactors.

Tradeoffs:
- Slight indirection vs. direct SDK calls, but faster iteration and portability generally outweigh this.

---

## Optimizing LlamaIndex and the pipeline

Retrieval performance
- Use a fit-for-purpose embedding model (`EMBED_MODEL_NAME`), matching domain language; smaller models improve latency.
- Ensure the same embed model is used for both indexing and retrieval (already enforced via `Settings.embed_model`).
- Tune Chroma HNSW parameters for your corpus size (see `HNSW_*` in `config.py`):
  - `HNSW_M`: graph degree; higher increases recall and memory.
  - `HNSW_CONSTRUCTION_EF`: affects indexing accuracy/time; raise for better quality.
  - `HNSW_SEARCH_EF`: raises recall at query time; increases latency.
  - `HNSW_SPACE`: `cosine` recommended for sentence embeddings.
- Adjust `TOP_K` to balance context quality and token costs. Common range: 3–8.

Indexing and chunking
- Match `CHUNK_SIZE` and `CHUNK_OVERLAP` to document characteristics; too small harms coherence, too large reduces retrievability.
- Rebuild BM25 index after significant corpus updates (handled on startup via `_build_bm25_index`).
- Disable telemetry (`CHROMA_TELEMETRY=off`) in production for deterministic performance.

Generation
- Choose `OPENAI_MODEL` for cost/latency vs. quality. Lower temperature is used for factuality and stability.
- Use concise prompts and include retrieval scores for transparency/debugging.
- For voice, post-summarize answers with a short, low-temperature pass (already implemented) to minimize TTS artifacts.

Caching (GAC)
- Monitor hit rate; adjust `GAC_SIMILARITY_THRESHOLD` and TTL accordingly.
- Cache only text answers to avoid TTS artifacts; re-summarize per voice request.

Concurrency & async
- Retrieval and expansion are batched and awaited concurrently where possible (`asyncio.gather`).
- WebSocket route spawns tasks per `connection_id` and supports cancellation to prevent work overrun.

Observability
- Structured logging is present across services. Consider adding request IDs and per-stage timers if deeper profiling is needed.

---

## Configuration reference (env via `.env`)

Core
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `RAG_TOP_K` (maps to `TOP_K`), `HYBRID_SEARCH_ALPHA`, `BM25_TOP_K`
- `ENABLE_QUERY_EXPANSION`, `QUERY_EXPANSION_MODEL`

Data & stores
- `DATA_DIR`, `DOCUMENTS_DIR`
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `CHROMA_TELEMETRY`
- `HNSW_SPACE`, `HNSW_CONSTRUCTION_EF`, `HNSW_SEARCH_EF`, `HNSW_M`, `HNSW_BATCH_SIZE`, `HNSW_SYNC_THRESHOLD`

Chunking & embeddings
- `EMBED_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

GAC
- `ENABLE_GAC`, `GAC_DIR`, `GAC_SIMILARITY_THRESHOLD`, `GAC_MAX_CACHE_SIZE_GB`, `GAC_TTL_HOURS`

Voice
- `TTS_KEY`, `TTS_URL`

---

## Operational tips

- Warm-up: Run a representative set of queries after deployment to build caches and JIT state.
- Memory: Monitor Chroma memory footprint as HNSW parameters are tuned upward.
- Upgrades: Keep `llama-index` and `chromadb` versions in sync with tested combinations from `requirements.txt`.
- Safety: If documents change frequently, consider embedding reindex triggers or time-based rebuilds.



## Medicall Chatbot API

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot for EMR-related documents. It supports:
- **Hybrid retrieval** (BM25 + vector) over ChromaDB via LlamaIndex
- **Generate-Augmented Cache (GAC)** for ultra-low latency repeated queries
- **Query expansion** (optional)
- **HTTP** endpoint for one-shot questions
- **WebSocket** endpoint for streaming responses and control (ping/stop)
- Optional **voice-friendly summaries** (TTS integration hooks)

### Project structure
```
project/
  app/
    api/
      router_factory.py
      routes/
        chat_routes.py
        responses.py
        schemas.py
    services/
      rag_service.py
      rag_chatbot_service.py
      embedding_service.py
      gac_service.py
      voice_service.py
    utils/
      config.py
      socket_manager.py
  data/
    documents/              # Source documents to index (plain text)
  app/db/
    chroma_db/              # Chroma persistent store (auto-generated)
    gac_cache/              # GAC SQLite cache (auto-generated)
  scripts/
    index_documents.py      # Document indexing script (auto-run at startup if needed)
  main.py                   # FastAPI app entry
  requirements.txt
```

---

## Prerequisites
- Python 3.10+
- macOS/Linux/Windows
- An OpenAI API key (or Azure OpenAI compatible) for generation

## Quickstart
1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Prepare environment variables
Create a `.env` file in the project root with at least:
```bash
OPENAI_API_KEY=YOUR_OPENAI_KEY

# Optional overrides (defaults shown)
# OPENAI_MODEL=gpt-4o
# RAG_TOP_K=3
# HYBRID_SEARCH_ALPHA=0.5
# BM25_TOP_K=10
# ENABLE_QUERY_EXPANSION=false
# QUERY_EXPANSION_MODEL=gpt-3.5-turbo
# ENABLE_GAC=true
# CHROMA_TELEMETRY=off
# DATA_DIR=app/db
# DOCUMENTS_DIR=data/documents
# CHROMA_PERSIST_DIR=app/db/chroma_db
# CHROMA_COLLECTION=emr_documents
# HNSW_SPACE=cosine
# HNSW_CONSTRUCTION_EF=100
# HNSW_SEARCH_EF=10
# HNSW_M=16
# HNSW_BATCH_SIZE=100
# HNSW_SYNC_THRESHOLD=1000

# Voice/TTS integration (optional)
# TTS_KEY=
# TTS_URL=
```

4) Add or update documents
Place plain-text files in `data/documents`. Indexing runs automatically on startup if new or missing.

5) Run the API
```bash
python main.py
```
Or
```bash
uvicorn main:app --port 5003  
```
The API starts on `http://0.0.0.0:5003`.

6) TTS
You should create API key form :
```bash
https://console.sws.speechify.com/login
```
---

## API

### WebSocket: Streaming & control
`GET /stream/{connection_id}`

Connect via WebSocket and send messages in JSON.

Supported message types:
- `{"type": "ping"}` — liveness probe; server replies `{ "type": "pong" }`
- `{"type": "stop"}` — cancels active tasks and ongoing generation for this connection
```bash
{
 "type": "chat_request",
 "user_id": "user-123",
 "query": "...", 
 "voice": false, 
 "chat_id": "optional-uuid"
 }
```
starts a chat

Server emits structured events via `socket_manager`, including:
- Status: `{ "type": "text"|"voice", "status": "start"|"complete" }`
- Final message: `{ "type": "message", "messages": "..." }`
- Errors/cancellation

---

## Indexing & Data
- Documents live in `data/documents` (plain text). Add/edit/remove files as needed.
- On startup, the app checks for new/changed documents and runs `scripts/index_documents.py` when needed.
- Vector store is persisted under `app/db/chroma_db`.

## Configuration
All configuration is centralized in `app/utils/config.py` and loaded from environment variables (see `.env` example). Notable settings:
- **OPENAI_API_KEY, OPENAI_MODEL**: model and key for generation
- **RAG_TOP_K**: number of retrieved chunks
- **HYBRID_SEARCH_ALPHA**: blend of BM25 (0) and vector (1)
- **BM25_TOP_K**: number of BM25 documents considered
- **ENABLE_QUERY_EXPANSION**: turn query expansion on/off
- **ENABLE_GAC**: enable semantic caching (GAC)
- **CHROMA_* & HNSW_***: vector store configuration
- **TTS_KEY, TTS_URL**: optional TTS integration

## Caching (GAC)
When enabled, semantically similar queries can be served from a local cache (`app/db/gac_cache`). This reduces latency for repeated questions. The cache can be inspected/cleared via `GACService` methods in `rag_service.py`.

## Development
Run with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5003
```

Lint/format per your team standards. The code is structured for clarity with minimal side effects and explicit configuration.

## Troubleshooting
- "Missing OPENAI_API_KEY": Ensure `.env` is present and loaded, or export the variable in your shell.
- No results returned: Confirm documents exist in `data/documents` and that indexing has completed; check `app/db/chroma_db`.
- Slow first query: The first request builds indices/warms caches; subsequent queries are faster, especially with GAC.
- WebSocket disconnects: Confirm the client keeps the connection alive (`ping`), and that `connection_id` is stable per session.
- GPU/torch errors: The embedding model defaults to `sentence-transformers/all-MiniLM-L6-v2`; CPU is sufficient, but large models may require proper Torch setup.

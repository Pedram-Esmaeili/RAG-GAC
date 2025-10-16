# Paths and settings for embedding/indexing
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Base project directory (default to two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data dirs
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(PROJECT_ROOT, "app/db")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR") or os.path.join(PROJECT_ROOT, "data/documents")

# Vector database (Chroma) settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR") or os.path.join(DATA_DIR, "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "emr_documents")
CHROMA_TELEMETRY = os.getenv("CHROMA_TELEMETRY", "off").lower()  # 'on' or 'off'

# Chroma HNSW Indexing parameters
HNSW_SPACE = os.getenv("HNSW_SPACE", "cosine")
HNSW_CONSTRUCTION_EF = int(os.getenv("HNSW_CONSTRUCTION_EF", "100")) # num_edges for index construction
HNSW_SEARCH_EF = int(os.getenv("HNSW_SEARCH_EF", "10")) # num_edges for search
HNSW_M = int(os.getenv("HNSW_M", "16")) # max_connections
HNSW_BATCH_SIZE = int(os.getenv("HNSW_BATCH_SIZE", "100"))
HNSW_SYNC_THRESHOLD = int(os.getenv("HNSW_SYNC_THRESHOLD", "1000"))

# Embedding model
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Hybrid Search
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5")) # 0 for BM25, 1 for Vector
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10")) # How many BM25 results to consider

# Query Expansion
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "false").lower() == "true"
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "gpt-3.5-turbo")

# GAC (Generate-Augmented Caching) settings
ENABLE_GAC = os.getenv("ENABLE_GAC", "true").lower() == "true"
GAC_DIR = os.getenv("GAC_DIR") or os.path.join(DATA_DIR, "gac_cache")
GAC_SIMILARITY_THRESHOLD = float(os.getenv("GAC_SIMILARITY_THRESHOLD", "0.92"))  # High threshold for semantic matching
GAC_MAX_CACHE_SIZE_GB = float(os.getenv("GAC_MAX_CACHE_SIZE_GB", "1.0"))
GAC_TTL_HOURS = int(os.getenv("GAC_TTL_HOURS", "168"))  # 7 days default

#TTS settings
TTS_KEY=os.getenv("TTS_KEY")
TTS_URL=os.getenv("TTS_URL")
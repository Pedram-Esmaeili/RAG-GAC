from typing import List, Optional
import os

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from app.utils import config  # Import the config module


def get_embed_model():
	return HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)


def get_chroma_client() -> chromadb.PersistentClient:
	os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
	# Disable telemetry if configured
	if config.CHROMA_TELEMETRY == "off":
		os.environ["ANONYMIZED_TELEMETRY"] = "false"
	return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_or_create_chroma_collection(client: chromadb.PersistentClient):
	"""
	Create optimized Chroma collection with HNSW index parameters.
	These settings significantly improve search speed and accuracy.
	"""
	return client.get_or_create_collection(
		name=config.CHROMA_COLLECTION,
		metadata={
			"hnsw:space": config.HNSW_SPACE,
			"hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
			"hnsw:search_ef": config.HNSW_SEARCH_EF,
			"hnsw:M": config.HNSW_M,
			"hnsw:batch_size": config.HNSW_BATCH_SIZE,
			"hnsw:sync_threshold": config.HNSW_SYNC_THRESHOLD
		}
	)


def load_documents(directory: Optional[str] = None) -> List[Document]:
	dir_to_read = directory or config.DOCUMENTS_DIR
	return SimpleDirectoryReader(dir_to_read).load_data()


def build_index(documents: Optional[List[Document]] = None) -> VectorStoreIndex:
	if documents is None:
		documents = load_documents()

	Settings.embed_model = get_embed_model()
	Settings.chunk_size = config.CHUNK_SIZE
	Settings.chunk_overlap = config.CHUNK_OVERLAP

	client = get_chroma_client()
	collection = get_or_create_chroma_collection(client)
	vector_store = ChromaVectorStore(chroma_collection=collection)
	storage_context = StorageContext.from_defaults(vector_store=vector_store)

	index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
	return index


def persist_index() -> None:
	# Chroma persistence handled by PersistentClient path
	# Nothing else required; ensure directory exists
	os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)


def reindex_documents(directory: Optional[str] = None) -> None:
	docs = load_documents(directory)
	build_index(docs)
	persist_index()


def ensure_index_exists() -> VectorStoreIndex:
	Settings.embed_model = get_embed_model()
	Settings.chunk_size = config.CHUNK_SIZE
	Settings.chunk_overlap = config.CHUNK_OVERLAP

	client = get_chroma_client()
	collection = get_or_create_chroma_collection(client)
	vector_store = ChromaVectorStore(chroma_collection=collection)
	storage_context = StorageContext.from_defaults(vector_store=vector_store)

	# Building without documents will not re-add chunks; but if empty DB, you need to index
	if collection.count() == 0:
		index = build_index(load_documents())
		persist_index()
		return index
	return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def has_new_documents() -> bool:
    """Checks if there are new documents in the DOCUMENTS_DIR that are not yet indexed."""
    indexed_documents_ids = set()
    try:
        client = get_chroma_client()
        collection = get_or_create_chroma_collection(client)
        if collection.count() > 0:
            # Fetch all metadata, assuming file paths are stored in 'file_path' or similar
            # Adjust this if your metadata structure is different
            all_docs = collection.get(include=['metadatas'])
            for metadata in all_docs.get('metadatas', []):
                if 'file_path' in metadata:
                    indexed_documents_ids.add(os.path.abspath(metadata['file_path']))
    except Exception as e:
        print(f"Error fetching indexed documents from Chroma: {e}")
        # If there's an error, assume there might be new documents to be safe
        return True

    filesystem_documents_paths = set()
    try:
        for root, _, files in os.walk(config.DOCUMENTS_DIR):
            for file in files:
                if file.endswith('.txt'): # Assuming only text files are indexed
                    file_path = os.path.join(root, file)
                    filesystem_documents_paths.add(os.path.abspath(file_path))
    except Exception as e:
        print(f"Error reading documents from filesystem: {e}")
        return True

    # Check if there's any document in the filesystem that is not in the indexed set
    if not filesystem_documents_paths.issubset(indexed_documents_ids):
        print("New documents detected or some documents are not indexed.")
        return True
    
    print("No new documents detected. Index is up to date.")
    return False

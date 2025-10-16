import time
import asyncio
from typing import Dict, List, Optional, Tuple
import os
from collections import defaultdict

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
import logging

from app.utils import config  # Import the config module

from openai import OpenAI, AsyncOpenAI
from app.services.embedding_service import get_embed_model
from app.services.gac_service import GACService
from app.services.voice_service import VoiceService

logger = logging.getLogger(__name__)

class RAGService:
	"""
	Advanced RAG Service with:
	- GAC (Generate-Agumented Cache) for ultra-low latency
	- Hybrid Search (BM25 + Vector)
	- Query Transformation & Expansion
	- Async/Parallel Retrieval
	- Optimized HNSW Indexing
	"""
	
	def __init__(self, top_k: int = config.TOP_K, alpha: float = config.HYBRID_SEARCH_ALPHA, enable_gac: bool = True):
		self.top_k = top_k
		self.alpha = alpha  # Balance between BM25 (0) and vector (1)
		
		# Ensure retrieval uses the same embedding model
		embed_model = get_embed_model()
		Settings.embed_model = embed_model
		
		# OpenAI clients (sync and async)
		self._client = OpenAI(api_key=config.OPENAI_API_KEY)
		self._async_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
		
		# Vector store setup
		self._vector_store = self._get_vector_store()
		self._index = VectorStoreIndex.from_vector_store(self._vector_store)
		
		# BM25 index setup
		self._bm25_index = None
		self._documents_cache = []
		self._build_bm25_index()
		
		# GAC setup - use same embedding model for consistency
		self.gac = GACService(embed_model=embed_model, enabled=enable_gac) if enable_gac else None
		self.voice_service = VoiceService() # Instantiate VoiceService

	def _get_vector_store(self) -> ChromaVectorStore:
		"""Initialize optimized Chroma vector store with HNSW parameters"""
		if config.CHROMA_TELEMETRY == "off":
			os.environ["ANONYMIZED_TELEMETRY"] = "false"
			
		client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
		
		# Get or create collection with HNSW optimizations
		collection = client.get_or_create_collection(
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
		return ChromaVectorStore(chroma_collection=collection)

	def _build_bm25_index(self):
		"""Build BM25 index from documents in vector store"""
		try:
			# Get all documents from Chroma
			collection = self._vector_store._collection
			results = collection.get(include=["documents", "metadatas"])
			
			if results and results["documents"]:
				self._documents_cache = [
					{
						"id": results["ids"][i],
						"text": results["documents"][i],
						"metadata": results["metadatas"][i] if results["metadatas"] else {}
					}
					for i in range(len(results["documents"]))
				]
				
				# Tokenize documents for BM25
				tokenized_docs = [doc["text"].lower().split() for doc in self._documents_cache]
				self._bm25_index = BM25Okapi(tokenized_docs)
				logger.info(f"BM25 index built with {len(self._documents_cache)} documents")
			else:
				logger.warning("No documents found in vector store. BM25 index is empty.")
		except Exception as e:
			logger.warning(f"Error building BM25 index: {e}")
			self._bm25_index = None

	async def _expand_query_async(self, query: str) -> List[str]:
		"""
		Transform and expand query using LLM to improve retrieval.
		Returns multiple query variations for better coverage.
		"""
		if not config.ENABLE_QUERY_EXPANSION:
			return [query]
		
		try:
			prompt = f"""Given this user question, generate 2-3 alternative phrasings or related queries that would help retrieve relevant information. Focus on:
1. Synonyms and alternative terms
2. More specific technical variations
3. Related concepts

Original question: {query}

Return only the alternative queries, one per line, without numbering or explanation."""

			response = await self._async_client.chat.completions.create(
				model=config.QUERY_EXPANSION_MODEL,
				messages=[
					{"role": "system", "content": "You are a query expansion expert for medical EMR systems."},
					{"role": "user", "content": prompt}
				],
				temperature=0.3,
				max_tokens=150
			)
			
			expanded = response.choices[0].message.content.strip().split("\n")
			expanded = [q.strip() for q in expanded if q.strip()]
			
			# Include original query
			return [query] + expanded[:2]  # Original + 2 expansions
		except Exception as e:
			logger.warning(f"Query expansion failed: {e}")
			return [query]

	def _bm25_search(self, query: str, top_k: int = config.BM25_TOP_K) -> List[Dict]:
		"""Perform BM25 sparse retrieval"""
		if not self._bm25_index or not self._documents_cache:
			return []
		
		tokenized_query = query.lower().split()
		scores = self._bm25_index.get_scores(tokenized_query)
		
		# Get top-k indices
		top_indices = np.argsort(scores)[::-1][:top_k]
		
		results = []
		for idx in top_indices:
			if scores[idx] > 0:  # Only include positive scores
				results.append({
					"doc": self._documents_cache[idx],
					"score": float(scores[idx])
				})
		
		return results

	async def _vector_search_async(self, query: str, top_k: int) -> List[NodeWithScore]:
		"""Perform async vector search"""
		# Run in thread pool since LlamaIndex retriever is synchronous
		loop = asyncio.get_event_loop()
		retriever = VectorIndexRetriever(index=self._index, similarity_top_k=top_k)
		return await loop.run_in_executor(None, retriever.retrieve, query)

	async def _hybrid_search_async(self, queries: List[str]) -> List[NodeWithScore]:
		"""
		Perform hybrid search combining BM25 and vector search.
		Uses multiple query variations for comprehensive retrieval.
		"""
		all_results = defaultdict(lambda: {"score": 0, "node": None, "sources": []})
		
		# Parallel search for all query variations
		tasks = []
		for query in queries:
			# Vector search task
			tasks.append(self._vector_search_async(query, self.top_k))
		
		# Wait for all vector searches
		vector_results_list = await asyncio.gather(*tasks)
		
		# Perform BM25 searches (fast, can do synchronously)
		bm25_results_list = [self._bm25_search(query, config.BM25_TOP_K) for query in queries]
		
		# Merge results with hybrid scoring
		for query_idx, (vector_results, bm25_results) in enumerate(zip(vector_results_list, bm25_results_list)):
			# Process vector results
			for node in vector_results:
				node_id = node.node.node_id
				vector_score = float(node.score) if node.score else 0.0
				
				if node_id not in all_results or all_results[node_id]["node"] is None:
					all_results[node_id]["node"] = node
				
				all_results[node_id]["score"] += self.alpha * vector_score
				all_results[node_id]["sources"].append(f"vector_q{query_idx}")
			
			# Process BM25 results
			for bm25_result in bm25_results:
				doc_text = bm25_result["doc"]["text"]
				bm25_score = bm25_result["score"]
				
				# Normalize BM25 score (typical range 0-10, normalize to 0-1)
				normalized_bm25 = min(bm25_score / 10.0, 1.0)
				
				# Find or create node
				doc_id = bm25_result["doc"]["id"]
				if doc_id not in all_results or all_results[doc_id]["node"] is None:
					# Create a TextNode for BM25 result
					text_node = TextNode(
						text=doc_text,
						metadata=bm25_result["doc"]["metadata"]
					)
					text_node.node_id = doc_id
					all_results[doc_id]["node"] = NodeWithScore(node=text_node, score=0.0)
				
				all_results[doc_id]["score"] += (1 - self.alpha) * normalized_bm25
				all_results[doc_id]["sources"].append(f"bm25_q{query_idx}")
		
		# Sort by combined score and return top-k
		sorted_results = sorted(
			all_results.values(),
			key=lambda x: x["score"],
			reverse=True
		)
		
		# Update scores in NodeWithScore objects
		final_results = []
		for result in sorted_results[:self.top_k]:
			if result["node"]:
				result["node"].score = result["score"]
				final_results.append(result["node"])
		
		return final_results

	async def retrieve_async(self, query: str) -> List[NodeWithScore]:
		"""
		Async retrieval with query expansion and hybrid search
		"""
		# Step 1: Expand query
		expanded_queries = await self._expand_query_async(query)
		logger.info(f"Expanded queries: {expanded_queries}")
		
		# Step 2: Hybrid search
		results = await self._hybrid_search_async(expanded_queries)
		
		return results

	async def generate_async(self, query: str, contexts: List[NodeWithScore], language: str = "en-US") -> Tuple[str, Dict[str, int], float]:
		"""Async generation with OpenAI"""
		prompt_parts: List[str] = []
		
		system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say you don't know."
		if language != "en-US":
			system_prompt += f" Provide the answer in {language}."
		
		prompt_parts.append(system_prompt)
		prompt_parts.append("Context:")
		
		for i, node in enumerate(contexts, start=1):
			score = f" (relevance: {node.score:.3f})" if node.score else ""
			prompt_parts.append(f"[Chunk {i}{score}]\n{node.node.get_content()}\n")
		
		prompt_parts.append(f"Question: {query}")
		prompt = "\n\n".join(prompt_parts)

		start_time = time.perf_counter()
		response = await self._async_client.chat.completions.create(
			model=config.OPENAI_MODEL,
			messages=[
				{"role": "system", "content": "You are a knowledgeable assistant for EMR documents."},
				{"role": "user", "content": prompt},
			],
			temperature=0.2,
		)
		latency_s = time.perf_counter() - start_time

		usage = {
			"prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
			"completion_tokens": getattr(response.usage, "completion_tokens", 0),
			"total_tokens": getattr(response.usage, "total_tokens", 0),
		}
		answer = response.choices[0].message.content if response.choices else ""
		return answer, usage, latency_s

	async def ask_async(self, query: str, use_cache: bool = True, language: str = "en-US", voice: bool = False,connection_id: str = None) -> Dict[str, object]:
		"""
		Async end-to-end RAG query with all optimizations:
		- GAC (Generate-Augmented Cache) - check first
		- Query expansion
		- Hybrid search (BM25 + Vector)
		- Parallel retrieval
		
		Args:
			query: User question
			use_cache: Whether to use GAC (default: True)
			language: Output language (default: en-US)
			voice: If true, format the response for voice playback (default: False)
		"""
		start_total = time.perf_counter()
		
		# Step 1: Check GAC first for cached response
		if use_cache and self.gac and self.gac.enabled:
			cached = self.gac.get_cached_response(query)
			if cached :  # Only use cache for text, not voice
				# Add cache-specific latency info
				cached["latency_s"] = cached.get("cache_lookup_time_s", 0.0)
				cached["generation_latency_s"] = 0.0  # No generation needed
				logger.info("Served from cache")
				if voice:
					gen_latency_s = 0
					summary_start_time = time.perf_counter()
					answer_for_voice = await self._summarize_for_voice_async(cached["answer"], language)
					summary_latency_s = time.perf_counter() - summary_start_time
					gen_latency_s += summary_latency_s
					await self.voice_service.process_response_parts_for_voice(answer_for_voice, language,connection_id=connection_id)  # Add summarization time to generation latency
					result = {
						"answer": answer_for_voice,
						"generation_latency_s": gen_latency_s,
						"from_cache": True,
					}
				else:
					result = {
						"answer": cached["answer"],
						"generation_latency_s": 0,
						"from_cache": True,
					}
				# Ensure total latency is set if coming from cache
				result["latency_s"] = cached.get("cache_lookup_time_s", 0.0)
				return result
		
		# Step 2: Cache miss - perform full RAG pipeline
		nodes = await self.retrieve_async(query)
		answer, usage, gen_latency_s = await self.generate_async(query, nodes, language)

		# Conditionally summarize for voice
		if voice:
			summary_start_time = time.perf_counter()
			answer = await self._summarize_for_voice_async(answer, language)
			summary_latency_s = time.perf_counter() - summary_start_time
			gen_latency_s += summary_latency_s  # Add summarization time to generation latency
			self.voice_service.process_response_parts_for_voice(answer, language)

		total_latency_s = time.perf_counter() - start_total
		
		contexts = [
			{
				"score": float(n.score) if n.score is not None else None,
				"text": n.node.get_content(),
			}
			for n in nodes
		]
		
		result = {
			"answer": answer,
			"usage": usage,
			"latency_s": total_latency_s,
			"generation_latency_s": gen_latency_s,
			"contexts": contexts,
			"from_cache": False,
		}
		
		# Step 3: Cache the response for future queries
		if use_cache and self.gac and self.gac.enabled and not voice: # Only cache text responses
			self.gac.cache_response(
				query=query,
				response=result,
				contexts=contexts,
				latency_s=total_latency_s
			)
		
		return result

	def ask(self, query: str, use_cache: bool = True, language: str = "en-US", voice: bool = False) -> Dict[str, object]:
		"""Synchronous wrapper for ask_async"""
		return asyncio.run(self.ask_async(query, use_cache=use_cache, language=language, voice=voice))
	
	async def _summarize_for_voice_async(self, text: str, language: str) -> str:
		"""
		Summarize text for voice playback using OpenAI, with restricted punctuation.
		"""
		try:
			system_prompt = "You are a helpful assistant. Extremely concisely summarize the following text for voice playback. Use ONLY '.' or '?' as punctuation. Do NOT use newlines. Preserve key information in the fewest possible words."
			if language != "en-US":
				system_prompt += f" Provide the summary in {language}."
			
			response = await self._async_client.chat.completions.create(
				model=config.OPENAI_MODEL,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": f"Summarize: {text}"},
				],
				temperature=0.1,
				max_tokens=70,
			)
			summary = response.choices[0].message.content.strip()
			# Post-process to ensure only '.' and '?' are used and no newlines
			summary = ''.join(char if char in ('.', '?') or char.isalnum() or char.isspace() else '' for char in summary)
			summary = ' '.join(summary.split())
			return summary
		except Exception as e:
			logger.error(f"Error summarizing for voice: {e}")
			return text  # Return original text if summarization fails
	
	def get_cache_stats(self) -> Dict:
		"""Get GAC statistics"""
		if not self.gac or not self.gac.enabled:
			return {"enabled": False}
		return self.gac.get_stats()
	
	def clear_cache(self) -> bool:
		"""Clear GAC cache"""
		if not self.gac or not self.gac.enabled:
			return False
		return self.gac.clear_cache()
	
	def list_cached_queries(self, limit: int = 20) -> List[Dict]:
		"""List cached queries"""
		if not self.gac or not self.gac.enabled:
			return []
		return self.gac.list_cached_queries(limit)

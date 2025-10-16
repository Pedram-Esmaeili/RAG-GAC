"""
GAC (Generate-Augmented Cache) Service

Provides 10-100x lower latency for similar queries by avoiding retrieval and generation.

Key Features:
- Semantic query matching using embeddings
- Persistent disk-based cache with TTL
- Automatic cache management (size limits, expiration)
- Cache analytics (hit rate, latency savings)
"""

import os
import time
import hashlib
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from diskcache import Cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils import config  # Import the config module


class GACService:
	"""
	Generate-Augmented Cache Service for proactive intelligence.
	
	Instead of retrieving on every query, pre-generates and caches responses
	with semantic matching for similar queries.
	"""
	
	def __init__(
		self,
		embed_model=None,
		similarity_threshold: float = config.GAC_SIMILARITY_THRESHOLD,
		ttl_hours: int = config.GAC_TTL_HOURS,
		enabled: bool = config.ENABLE_GAC
	):
		self.enabled = enabled
		self.similarity_threshold = similarity_threshold
		self.ttl_seconds = ttl_hours * 3600
		self.embed_model = embed_model
		
		# Cache statistics
		self.stats = {
			"hits": 0,
			"misses": 0,
			"total_latency_saved_s": 0.0,
		}
		
		if not self.enabled:
			print("⚠️  GAC disabled. Set ENABLE_GAC=true to enable caching.")
			self.cache = None
			return
		
		# Initialize disk-based cache
		os.makedirs(config.GAC_DIR, exist_ok=True)
		max_size_bytes = int(config.GAC_MAX_CACHE_SIZE_GB * 1024 * 1024 * 1024)
		
		self.cache = Cache(
			directory=config.GAC_DIR,
			size_limit=max_size_bytes,
			eviction_policy='least-recently-used'
		)
		
		print(f"✅ GAC initialized: {config.GAC_DIR}")
		print(f"   Similarity threshold: {similarity_threshold}")
		print(f"   TTL: {ttl_hours} hours")
		print(f"   Max size: {config.GAC_MAX_CACHE_SIZE_GB} GB")
		print(f"   Current cache size: {len(self.cache)} entries")
	
	def _get_query_embedding(self, query: str) -> np.ndarray:
		"""Generate embedding for query using the same model as retrieval"""
		if self.embed_model is None:
			from app.services.embedding_service import get_embed_model
			self.embed_model = get_embed_model()
		
		# Get embedding as numpy array
		embedding = self.embed_model.get_text_embedding(query)
		return np.array(embedding)
	
	def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
		"""Compute cosine similarity between two embeddings"""
		return float(cosine_similarity([emb1], [emb2])[0][0])
	
	def _generate_cache_key(self, query: str) -> str:
		"""Generate stable cache key from query"""
		# Normalize query (lowercase, strip whitespace)
		normalized = query.lower().strip()
		return hashlib.md5(normalized.encode()).hexdigest()
	
	def get_cached_response(self, query: str) -> Optional[Dict]:
		"""
		Attempt to retrieve cached response using semantic matching.
		
		Returns:
			Cached response dict if found with sufficient similarity, else None
		"""
		if not self.enabled or self.cache is None:
			return None
		
		start_time = time.perf_counter()
		
		try:
			# Get query embedding
			query_embedding = self._get_query_embedding(query)
			
			# Search for similar cached queries
			best_match = None
			best_similarity = 0.0
			
			# Iterate through cache entries with embeddings
			for key in self.cache.iterkeys():
				if key.startswith("emb_"):
					continue  # Skip embedding entries
				
				cached_data = self.cache.get(key)
				if not cached_data:
					continue
				
				# Get cached query embedding
				emb_key = f"emb_{key}"
				cached_embedding = self.cache.get(emb_key)
				
				if cached_embedding is None:
					continue
				
				# Compute similarity
				similarity = self._compute_similarity(query_embedding, cached_embedding)
				
				if similarity > best_similarity:
					best_similarity = similarity
					best_match = cached_data
			
			# Check if best match exceeds threshold
			if best_match and best_similarity >= self.similarity_threshold:
				# Cache hit!
				lookup_time = time.perf_counter() - start_time
				
				self.stats["hits"] += 1
				original_latency = best_match.get("latency_s", 0)
				self.stats["total_latency_saved_s"] += max(0, original_latency - lookup_time)
				
				# Return cached response with metadata
				result = best_match.copy()
				result["from_cache"] = True
				result["cache_similarity"] = best_similarity
				result["cache_lookup_time_s"] = lookup_time
				
				print(f"🎯 GAC HIT! Similarity: {best_similarity:.3f}, Lookup: {lookup_time*1000:.1f}ms")
				return result
			
			# Cache miss
			self.stats["misses"] += 1
			return None
			
		except Exception as e:
			print(f"⚠️  GAC lookup error: {e}")
			return None
	
	def cache_response(
		self,
		query: str,
		response: Dict,
		contexts: List[Dict],
		latency_s: float,
		metadata: Optional[Dict] = None
	) -> bool:
		"""
		Cache a query-response pair with semantic embedding.
		
		Args:
			query: Original user query
			response: Generated response dict
			contexts: Retrieved contexts
			latency_s: Time taken to generate response
			metadata: Optional additional metadata
			
		Returns:
			True if successfully cached, False otherwise
		"""
		if not self.enabled or self.cache is None:
			return False
		
		try:
			# Generate cache key and embedding
			cache_key = self._generate_cache_key(query)
			query_embedding = self._get_query_embedding(query)
			
			# Prepare cache entry
			cache_entry = {
				"query": query,
				"answer": response.get("answer", ""),
				"usage": response.get("usage", {}),
				"contexts": contexts,
				"latency_s": latency_s,
				"cached_at": datetime.now().isoformat(),
				"metadata": metadata or {}
			}
			
			# Store response and embedding separately
			self.cache.set(cache_key, cache_entry, expire=self.ttl_seconds)
			self.cache.set(f"emb_{cache_key}", query_embedding, expire=self.ttl_seconds)
			
			return True
			
		except Exception as e:
			print(f"⚠️  GAC cache error: {e}")
			return False
	
	async def pre_generate_cache_async(self, common_queries: List[str], rag_service) -> Dict:
		"""
		Pre-generate cache for common queries (proactive caching).
		
		Args:
			common_queries: List of anticipated user queries
			rag_service: RAGService instance to generate responses
			
		Returns:
			Statistics about pre-generation
		"""
		if not self.enabled:
			return {"error": "GAC not enabled"}
		
		results = {
			"total": len(common_queries),
			"cached": 0,
			"failed": 0,
			"total_time_s": 0.0
		}
		
		print(f"\n🚀 Pre-generating cache for {len(common_queries)} queries...")
		
		start_time = time.perf_counter()
		
		for i, query in enumerate(common_queries, 1):
			try:
				# Check if already cached
				if self.get_cached_response(query):
					print(f"  [{i}/{len(common_queries)}] Already cached: {query[:60]}...")
					results["cached"] += 1
					continue
				
				# Generate response using async method
				print(f"  [{i}/{len(common_queries)}] Generating: {query[:60]}...")
				response = await rag_service.ask_async(query)
				
				# Cache the response
				self.cache_response(
					query=query,
					response=response,
					contexts=response.get("contexts", []),
					latency_s=response.get("latency_s", 0.0),
					metadata={"pre_generated": True}
				)
				
				results["cached"] += 1
				
			except Exception as e:
				print(f"  ❌ Failed to generate for: {query[:60]}... - {e}")
				results["failed"] += 1
		
		results["total_time_s"] = time.perf_counter() - start_time
		
		print(f"\n✅ Pre-generation complete!")
		print(f"   Cached: {results['cached']}/{results['total']}")
		print(f"   Failed: {results['failed']}")
		print(f"   Time: {results['total_time_s']:.1f}s")
		
		return results
	
	def pre_generate_cache(self, common_queries: List[str], rag_service) -> Dict:
		"""Synchronous wrapper for pre_generate_cache_async"""
		import asyncio
		try:
			loop = asyncio.get_running_loop()
			# Already in an event loop - can't use asyncio.run()
			raise RuntimeError("pre_generate_cache cannot be called from an async context. Use pre_generate_cache_async instead.")
		except RuntimeError as e:
			if "no running event loop" in str(e).lower():
				# Not in an event loop, safe to use asyncio.run()
				return asyncio.run(self.pre_generate_cache_async(common_queries, rag_service))
			else:
				# Already in event loop
				raise RuntimeError("pre_generate_cache cannot be called from an async context. Use pre_generate_cache_async instead.") from e
	
	def get_stats(self) -> Dict:
		"""Get cache statistics"""
		if not self.enabled:
			return {"enabled": False}
		
		total_queries = self.stats["hits"] + self.stats["misses"]
		hit_rate = (self.stats["hits"] / total_queries * 100) if total_queries > 0 else 0.0
		avg_latency_saved = (
			self.stats["total_latency_saved_s"] / self.stats["hits"]
			if self.stats["hits"] > 0 else 0.0
		)
		
		return {
			"enabled": True,
			"cache_entries": len(self.cache) // 2,  # Divide by 2 (response + embedding)
			"cache_hits": self.stats["hits"],
			"cache_misses": self.stats["misses"],
			"hit_rate_percent": hit_rate,
			"total_latency_saved_s": self.stats["total_latency_saved_s"],
			"avg_latency_saved_per_hit_s": avg_latency_saved,
			"similarity_threshold": self.similarity_threshold,
		}
	
	def clear_cache(self) -> bool:
		"""Clear all cache entries"""
		if not self.enabled or self.cache is None:
			return False
		
		try:
			self.cache.clear()
			self.stats = {
				"hits": 0,
				"misses": 0,
				"total_latency_saved_s": 0.0,
			}
			print("✅ Cache cleared")
			return True
		except Exception as e:
			print(f"❌ Error clearing cache: {e}")
			return False
	
	def list_cached_queries(self, limit: int = 20) -> List[Dict]:
		"""List cached queries for inspection"""
		if not self.enabled or self.cache is None:
			return []
		
		queries = []
		count = 0
		
		for key in self.cache.iterkeys():
			if key.startswith("emb_"):
				continue
			
			if count >= limit:
				break
			
			cached_data = self.cache.get(key)
			if cached_data:
				queries.append({
					"query": cached_data.get("query", ""),
					"cached_at": cached_data.get("cached_at", ""),
					"answer_preview": cached_data.get("answer", "")[:100] + "...",
				})
				count += 1
		
		return queries


# Example usage
if __name__ == "__main__":
	print("Testing GAC Service...\n")
	
	# Initialize
	gac = GACService()
	
	# Test caching
	test_query = "What are the benefits of EMR systems?"
	test_response = {
		"answer": "EMR systems improve healthcare quality, efficiency, and patient safety...",
		"usage": {"total_tokens": 150}
	}
	test_contexts = [{"text": "EMR benefits include...", "score": 0.95}]
	
	# Cache response
	gac.cache_response(test_query, test_response, test_contexts, latency_s=2.5)
	print("✅ Cached test response\n")
	
	# Try to retrieve with similar query
	similar_query = "What are EMR system advantages?"
	cached = gac.get_cached_response(similar_query)
	
	if cached:
		print(f"✅ Retrieved from cache!")
		print(f"   Similarity: {cached['cache_similarity']:.3f}")
		print(f"   Answer: {cached['answer'][:60]}...")
	else:
		print("❌ Not found in cache")
	
	# Show stats
	print(f"\n📊 Cache Stats:")
	stats = gac.get_stats()
	for key, value in stats.items():
		print(f"   {key}: {value}")


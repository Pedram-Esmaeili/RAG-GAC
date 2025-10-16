"""
Semantic Chunking Utility

Provides intelligent text chunking that:
- Respects sentence boundaries
- Maintains semantic coherence
- Creates overlapping windows for context preservation
"""

import re
from typing import List


class SemanticChunker:
	"""
	Advanced text chunker that respects sentence boundaries and semantic structure.
	Better than naive character-based chunking for preserving context.
	"""
	
	def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
	
	def _split_sentences(self, text: str) -> List[str]:
		"""
		Split text into sentences using regex patterns.
		Handles common abbreviations and edge cases.
		"""
		# Pattern for sentence boundaries
		# Matches . ! ? followed by space and capital letter or end of string
		sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z]|$)'
		sentences = re.split(sentence_pattern, text)
		
		# Clean and filter empty sentences
		sentences = [s.strip() for s in sentences if s.strip()]
		return sentences
	
	def chunk_text(self, text: str) -> List[str]:
		"""
		Create semantic chunks that respect sentence boundaries.
		
		Args:
			text: Input text to chunk
			
		Returns:
			List of text chunks with semantic coherence and overlap
		"""
		sentences = self._split_sentences(text)
		
		if not sentences:
			return []
		
		chunks = []
		current_chunk = []
		current_length = 0
		
		for sentence in sentences:
			sentence_length = len(sentence)
			
			# If adding this sentence exceeds chunk size
			if current_length + sentence_length > self.chunk_size and current_chunk:
				# Save current chunk
				chunks.append(" ".join(current_chunk))
				
				# Start new chunk with overlap
				# Calculate how many sentences to keep for overlap
				overlap_chunk = []
				overlap_length = 0
				
				for s in reversed(current_chunk):
					if overlap_length + len(s) <= self.chunk_overlap:
						overlap_chunk.insert(0, s)
						overlap_length += len(s)
					else:
						break
				
				current_chunk = overlap_chunk
				current_length = overlap_length
			
			# Add sentence to current chunk
			current_chunk.append(sentence)
			current_length += sentence_length
		
		# Add final chunk
		if current_chunk:
			chunks.append(" ".join(current_chunk))
		
		return chunks
	
	def chunk_documents(self, documents: List[str]) -> List[dict]:
		"""
		Chunk multiple documents and preserve metadata.
		
		Args:
			documents: List of document texts
			
		Returns:
			List of dicts with 'text', 'doc_index', and 'chunk_index'
		"""
		all_chunks = []
		
		for doc_idx, doc_text in enumerate(documents):
			chunks = self.chunk_text(doc_text)
			
			for chunk_idx, chunk in enumerate(chunks):
				all_chunks.append({
					"text": chunk,
					"doc_index": doc_idx,
					"chunk_index": chunk_idx,
					"total_chunks": len(chunks)
				})
		
		return all_chunks


def chunk_by_tokens(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
	"""
	Simple token-based chunking fallback.
	Uses whitespace tokenization for speed.
	"""
	tokens = text.split()
	chunks = []
	
	for i in range(0, len(tokens), chunk_size - overlap):
		chunk_tokens = tokens[i:i + chunk_size]
		chunks.append(" ".join(chunk_tokens))
	
	return chunks


# Example usage
if __name__ == "__main__":
	sample_text = """
	Electronic Medical Records (EMR) are digital versions of patient charts. They contain 
	comprehensive patient medical history. EMRs improve healthcare quality and efficiency.
	
	Benefits include better care coordination. Providers can access complete patient history.
	This leads to more informed clinical decisions. EMRs also reduce medical errors significantly.
	"""
	
	chunker = SemanticChunker(chunk_size=100, chunk_overlap=30)
	chunks = chunker.chunk_text(sample_text.strip())
	
	print(f"Created {len(chunks)} chunks:\n")
	for i, chunk in enumerate(chunks, 1):
		print(f"Chunk {i} ({len(chunk)} chars):")
		print(f"  {chunk}\n")


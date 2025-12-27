"""
Cross-encoder re-ranking module for DocuMind.

This module provides re-ranking functionality using cross-encoder models
to improve retrieval quality after initial vector search.
"""
from dataclasses import dataclass
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import CrossEncoder

from app.config import get_settings


@dataclass
class RankedDocument:
    """A document with its relevance score after re-ranking."""
    id: str
    text: str
    score: float
    original_rank: int
    metadata: Optional[dict] = None


class CrossEncoderReranker:
    """
    Cross-encoder re-ranking service.
    
    Cross-encoders process query-document pairs together, allowing them
    to capture more nuanced relevance signals than bi-encoder (embedding)
    approaches. However, they are slower, so we use them to re-rank
    a smaller set of candidates retrieved by vector search.
    
    Trade-offs:
    - More accurate relevance scoring than vector similarity
    - Slower (processes pairs, not individual texts)
    - Memory efficient (smaller model)
    
    Typical workflow:
    1. Vector search returns top-20 candidates
    2. Cross-encoder re-ranks to find top-5
    3. Final 5 are used for context assembly
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the re-ranker.
        
        Args:
            model_name: Name of the cross-encoder model
            max_length: Maximum sequence length for the model
        """
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.max_length = max_length
        
        print(f"Loading cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=max_length
        )
        print("Cross-encoder loaded.")
        
        # Thread pool for async execution
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    async def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """
        Re-rank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents with 'id', 'text', and optional metadata
            top_k: Number of top documents to return
            
        Returns:
            List of RankedDocument sorted by relevance score (descending)
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, doc.get('text', '')) for doc in documents]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            self._executor,
            self._score_pairs,
            pairs
        )
        
        # Create ranked documents
        ranked = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            ranked.append(RankedDocument(
                id=doc.get('id', str(i)),
                text=doc.get('text', ''),
                score=float(score),
                original_rank=i,
                metadata=doc.get('metadata')
            ))
        
        # Sort by score descending and return top-k
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
    
    def _score_pairs(self, pairs: List[tuple]) -> List[float]:
        """
        Score query-document pairs synchronously.
        
        Args:
            pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    async def rerank_with_scores(
        self,
        query: str,
        documents: List[dict]
    ) -> List[RankedDocument]:
        """
        Re-rank all documents and return with scores.
        
        Similar to rerank() but returns all documents, not just top-k.
        Useful for analysis or when you want to apply your own threshold.
        
        Args:
            query: The search query
            documents: List of documents
            
        Returns:
            All documents with scores, sorted by relevance
        """
        if not documents:
            return []
        
        pairs = [(query, doc.get('text', '')) for doc in documents]
        
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            self._executor,
            self._score_pairs,
            pairs
        )
        
        ranked = [
            RankedDocument(
                id=doc.get('id', str(i)),
                text=doc.get('text', ''),
                score=float(score),
                original_rank=i,
                metadata=doc.get('metadata')
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked


class RerankerFactory:
    """Factory for creating reranker instances."""
    
    _instance: Optional[CrossEncoderReranker] = None
    
    @classmethod
    def get_instance(cls) -> CrossEncoderReranker:
        """Get or create the singleton reranker."""
        if cls._instance is None:
            cls._instance = CrossEncoderReranker()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton."""
        cls._instance = None

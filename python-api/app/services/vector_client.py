"""
Vector service client for DocuMind.

This module provides an async HTTP client for communicating
with the Go vector search service.
"""
from typing import List, Optional
from dataclasses import dataclass
import httpx

from app.config import get_settings


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    id: str
    score: float
    document_id: str
    chunk_index: int
    text: str
    page_number: Optional[int] = None


class VectorServiceClient:
    """
    Async client for the Go vector search service.
    
    This client handles all communication with the Go service,
    including vector insertion and similarity search.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the vector service client.
        
        Args:
            base_url: URL of the Go vector service
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.base_url = base_url or settings.vector_service_url
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def insert(
        self,
        id: str,
        embedding: List[float],
        document_id: str,
        chunk_index: int,
        text: str,
        page_number: Optional[int] = None
    ) -> bool:
        """
        Insert a single vector.
        
        Args:
            id: Unique vector ID
            embedding: The embedding vector
            document_id: Source document ID
            chunk_index: Index of chunk in document
            text: Original text content
            page_number: Optional page number
            
        Returns:
            True if successful
        """
        payload = {
            "id": id,
            "embedding": embedding,
            "metadata": {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text": text,
                "page_number": page_number or 0
            }
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/insert",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("success", False)
        except httpx.HTTPError as e:
            raise VectorServiceError(f"Insert failed: {str(e)}")
    
    async def insert_batch(
        self,
        vectors: List[dict]
    ) -> int:
        """
        Insert multiple vectors in a batch.
        
        Args:
            vectors: List of vector dicts with id, embedding, and metadata
            
        Returns:
            Number of vectors inserted
        """
        payload = {"vectors": vectors}
        
        try:
            response = await self._client.post(
                f"{self.base_url}/insert/batch",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("inserted", 0)
        except httpx.HTTPError as e:
            raise VectorServiceError(f"Batch insert failed: {str(e)}")
    
    async def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        algorithm: str = "hnsw"
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
            algorithm: Search algorithm ("hnsw" or "bruteforce")
            
        Returns:
            List of search results
        """
        payload = {
            "embedding": embedding,
            "top_k": top_k,
            "algorithm": algorithm
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/search",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for r in data.get("results", []):
                metadata = r.get("metadata", {})
                results.append(VectorSearchResult(
                    id=r.get("id", ""),
                    score=r.get("score", 0.0),
                    document_id=metadata.get("document_id", ""),
                    chunk_index=metadata.get("chunk_index", 0),
                    text=metadata.get("text", ""),
                    page_number=metadata.get("page_number")
                ))
            
            return results
        except httpx.HTTPError as e:
            raise VectorServiceError(f"Search failed: {str(e)}")
    
    async def health(self) -> dict:
        """Check the health of the vector service."""
        try:
            response = await self._client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return {"status": "unavailable"}
    
    async def stats(self) -> dict:
        """Get statistics from the vector service."""
        try:
            response = await self._client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise VectorServiceError(f"Stats request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class VectorServiceError(Exception):
    """Exception raised for vector service errors."""
    pass


class VectorServiceFactory:
    """Factory for vector service client instances."""
    
    _instance: Optional[VectorServiceClient] = None
    
    @classmethod
    def get_instance(cls) -> VectorServiceClient:
        """Get or create the singleton client."""
        if cls._instance is None:
            cls._instance = VectorServiceClient()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton."""
        cls._instance = None

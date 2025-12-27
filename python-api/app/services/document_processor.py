"""
Document processor service for DocuMind.

This module handles document upload, text extraction, chunking,
embedding generation, and vector indexing.
"""
import os
import uuid
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import asyncio

import fitz  # PyMuPDF

from app.config import get_settings
from app.core.chunking import SentenceAwareChunker, Chunk
from app.core.embeddings import EmbeddingServiceFactory
from app.services.vector_client import VectorServiceFactory
from app.services.redis_client import RedisClientFactory


@dataclass
class ProcessingResult:
    """Result of document processing."""
    document_id: str
    filename: str
    num_chunks: int
    num_pages: int
    status: str
    error: Optional[str] = None


class DocumentProcessor:
    """
    Document processing service.
    
    Handles the complete pipeline from upload to indexed vectors:
    1. Text extraction (PDF/TXT)
    2. Chunking with sentence awareness
    3. Embedding generation (with caching)
    4. Vector insertion into Go service
    
    Processing is done asynchronously with task status updates.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        settings = get_settings()
        self.chunker = SentenceAwareChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_file(
        self,
        file_content: bytes,
        filename: str,
        task_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process an uploaded file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            task_id: Optional task ID for status updates
            
        Returns:
            ProcessingResult with processing details
        """
        document_id = str(uuid.uuid4())
        task_id = task_id or document_id
        
        cache = RedisClientFactory.get_cache_service()
        
        try:
            # Update task status
            await cache.set_task_status(task_id, {
                "status": "extracting",
                "progress": 0.1,
                "document_id": document_id
            })
            
            # Extract text
            ext = Path(filename).suffix.lower()
            if ext == ".pdf":
                text, num_pages = self._extract_pdf(file_content)
            elif ext == ".txt":
                text = file_content.decode("utf-8")
                num_pages = 1
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            if not text.strip():
                raise ValueError("No text content found in document")
            
            # Update status
            await cache.set_task_status(task_id, {
                "status": "chunking",
                "progress": 0.3,
                "document_id": document_id
            })
            
            # Chunk the document
            chunks = self.chunker.chunk(text, document_id=document_id)
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Update status
            await cache.set_task_status(task_id, {
                "status": "embedding",
                "progress": 0.5,
                "document_id": document_id,
                "num_chunks": len(chunks)
            })
            
            # Generate embeddings
            embedding_service = EmbeddingServiceFactory.get_instance()
            texts = [chunk.text for chunk in chunks]
            embeddings = await embedding_service.embed(texts)
            
            # Update status
            await cache.set_task_status(task_id, {
                "status": "indexing",
                "progress": 0.8,
                "document_id": document_id,
                "num_chunks": len(chunks)
            })
            
            # Insert vectors
            vector_client = VectorServiceFactory.get_instance()
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors.append({
                    "id": f"{document_id}_{chunk.chunk_index}",
                    "embedding": embedding,
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "page_number": chunk.page_number or 0
                    }
                })
            
            inserted = await vector_client.insert_batch(vectors)
            
            # Complete
            await cache.set_task_status(task_id, {
                "status": "completed",
                "progress": 1.0,
                "document_id": document_id,
                "num_chunks": len(chunks),
                "num_pages": num_pages
            })
            
            # Store document metadata
            await cache.redis.set_json(f"doc:{document_id}:meta", {
                "filename": filename,
                "num_chunks": len(chunks),
                "num_pages": num_pages
            }, ttl=86400 * 7)  # 7 days
            
            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                num_chunks=len(chunks),
                num_pages=num_pages,
                status="completed"
            )
            
        except Exception as e:
            await cache.set_task_status(task_id, {
                "status": "failed",
                "progress": 0,
                "error": str(e)
            })
            
            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                num_chunks=0,
                num_pages=0,
                status="failed",
                error=str(e)
            )
    
    def _extract_pdf(self, content: bytes) -> tuple:
        """
        Extract text from PDF content.
        
        Args:
            content: PDF file bytes
            
        Returns:
            Tuple of (text, num_pages)
        """
        doc = fitz.open(stream=content, filetype="pdf")
        
        text_parts = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                # Add page marker for reference
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        num_pages = len(doc)
        doc.close()
        
        return "\n\n".join(text_parts), num_pages
    
    async def get_document_info(self, document_id: str) -> Optional[dict]:
        """Get metadata for a document."""
        cache = RedisClientFactory.get_cache_service()
        return await cache.redis.get_json(f"doc:{document_id}:meta")


class BackgroundTaskRunner:
    """
    Background task runner for async document processing.
    
    Uses asyncio for concurrent task execution.
    """
    
    def __init__(self):
        """Initialize the task runner."""
        self.processor = DocumentProcessor()
        self._tasks: dict = {}
    
    async def submit_task(
        self,
        file_content: bytes,
        filename: str
    ) -> str:
        """
        Submit a document for background processing.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Task ID for status tracking
        """
        task_id = str(uuid.uuid4())
        
        # Create background task
        task = asyncio.create_task(
            self.processor.process_file(file_content, filename, task_id)
        )
        self._tasks[task_id] = task
        
        # Cleanup when done
        task.add_done_callback(lambda t: self._tasks.pop(task_id, None))
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get the status of a task."""
        cache = RedisClientFactory.get_cache_service()
        return await cache.get_task_status(task_id)


class DocumentProcessorFactory:
    """Factory for document processor instances."""
    
    _processor: Optional[DocumentProcessor] = None
    _task_runner: Optional[BackgroundTaskRunner] = None
    
    @classmethod
    def get_processor(cls) -> DocumentProcessor:
        """Get or create the document processor."""
        if cls._processor is None:
            cls._processor = DocumentProcessor()
        return cls._processor
    
    @classmethod
    def get_task_runner(cls) -> BackgroundTaskRunner:
        """Get or create the task runner."""
        if cls._task_runner is None:
            cls._task_runner = BackgroundTaskRunner()
        return cls._task_runner

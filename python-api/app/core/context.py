"""
Context assembly module for DocuMind.

This module handles assembling retrieved chunks into a coherent context
for LLM generation, respecting token limits and preserving source attribution.
"""
from dataclasses import dataclass
from typing import List, Optional
import tiktoken

from app.config import get_settings
from app.core.reranking import RankedDocument


@dataclass
class ContextChunk:
    """A chunk included in the context with source information."""
    text: str
    document_id: str
    chunk_index: int
    page_number: Optional[int]
    relevance_score: float


@dataclass
class AssembledContext:
    """The assembled context ready for LLM prompting."""
    context_text: str
    chunks: List[ContextChunk]
    total_tokens: int
    truncated: bool
    prompt: str


class ContextAssembler:
    """
    Context assembly service.
    
    This class is responsible for:
    1. Selecting the most relevant chunks that fit within token limits
    2. Formatting chunks with source attribution
    3. Building the final prompt for the LLM
    
    Token counting uses tiktoken for accurate OpenAI-compatible counting.
    For local models (Ollama), this provides a reasonable approximation.
    
    Design decisions:
    - Conservative token estimation (better to have room than overflow)
    - Source attribution embedded in context for citation
    - Template-based prompt construction for flexibility
    """
    
    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        model_encoding: str = "cl100k_base"
    ):
        """
        Initialize the context assembler.
        
        Args:
            max_context_tokens: Maximum tokens for context
            model_encoding: Tiktoken encoding name
        """
        settings = get_settings()
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(model_encoding)
        except Exception:
            # Fallback to cl100k_base if model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def assemble(
        self,
        query: str,
        ranked_docs: List[RankedDocument],
        system_prompt: Optional[str] = None
    ) -> AssembledContext:
        """
        Assemble context from ranked documents.
        
        Args:
            query: The user's query
            ranked_docs: Documents ranked by relevance
            system_prompt: Optional system prompt prefix
            
        Returns:
            AssembledContext with formatted prompt
        """
        if system_prompt is None:
            system_prompt = self._default_system_prompt()
        
        # Calculate token budget
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        template_tokens = 100  # Reserve for formatting
        
        available_tokens = self.max_context_tokens - system_tokens - query_tokens - template_tokens
        
        # Select chunks that fit
        selected_chunks = []
        current_tokens = 0
        truncated = False
        
        for doc in ranked_docs:
            chunk_text = self._format_chunk(doc, len(selected_chunks) + 1)
            chunk_tokens = self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(ContextChunk(
                    text=doc.text,
                    document_id=doc.metadata.get('document_id', 'unknown') if doc.metadata else 'unknown',
                    chunk_index=doc.metadata.get('chunk_index', 0) if doc.metadata else 0,
                    page_number=doc.metadata.get('page_number') if doc.metadata else None,
                    relevance_score=doc.score
                ))
                current_tokens += chunk_tokens
            else:
                truncated = True
                break
        
        # Build context text
        context_parts = []
        for i, (chunk, doc) in enumerate(zip(selected_chunks, ranked_docs[:len(selected_chunks)])):
            context_parts.append(self._format_chunk(doc, i + 1))
        
        context_text = "\n\n".join(context_parts)
        
        # Build full prompt
        prompt = self._build_prompt(system_prompt, context_text, query)
        
        return AssembledContext(
            context_text=context_text,
            chunks=selected_chunks,
            total_tokens=self.count_tokens(prompt),
            truncated=truncated,
            prompt=prompt
        )
    
    def _format_chunk(self, doc: RankedDocument, index: int) -> str:
        """Format a single chunk with source attribution."""
        source_info = []
        if doc.metadata:
            if doc.metadata.get('document_id'):
                source_info.append(f"Document: {doc.metadata['document_id']}")
            if doc.metadata.get('page_number'):
                source_info.append(f"Page: {doc.metadata['page_number']}")
        
        source_str = " | ".join(source_info) if source_info else "Unknown source"
        
        return f"[Source {index}: {source_str}]\n{doc.text}"
    
    def _default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return """You are a helpful assistant that answers questions based on the provided context.
Your answers should be:
1. Accurate - only use information from the provided context
2. Well-cited - reference the source numbers when using information
3. Concise - provide clear, focused answers
4. Honest - if the context doesn't contain enough information, say so

When citing sources, use the format [Source N] where N is the source number."""
    
    def _build_prompt(
        self,
        system_prompt: str,
        context: str,
        query: str
    ) -> str:
        """Build the full prompt for the LLM."""
        return f"""{system_prompt}

## Context
The following are relevant excerpts from the documents:

{context}

## Question
{query}

## Answer
Based on the provided context, """
    
    def estimate_response_tokens(self, prompt_tokens: int, max_response: int = 500) -> int:
        """Estimate maximum response tokens given prompt size."""
        # Most models have a combined input+output limit
        # Leave room for response
        return min(max_response, 4096 - prompt_tokens)


class ContextAssemblerFactory:
    """Factory for context assembler instances."""
    
    _instance: Optional[ContextAssembler] = None
    
    @classmethod
    def get_instance(cls) -> ContextAssembler:
        """Get or create the singleton context assembler."""
        if cls._instance is None:
            cls._instance = ContextAssembler()
        return cls._instance

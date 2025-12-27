"""
Document chunking module for DocuMind.

This module provides manual document chunking without LangChain.
Implements recursive character text splitting with sentence boundary awareness.
"""
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    document_id: Optional[str] = None


class RecursiveCharacterChunker:
    """
    Manual implementation of recursive character text splitting.
    
    This chunker attempts to split text at natural boundaries (paragraphs,
    sentences, words) while respecting the maximum chunk size.
    
    Algorithm:
    1. Try to split at the first separator (e.g., double newline)
    2. If chunks are still too large, use the next separator
    3. Continue until chunks fit or we're splitting at character level
    
    This is similar to LangChain's RecursiveCharacterTextSplitter but
    implemented from scratch to avoid framework dependencies.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try, in order of preference
            length_function: Function to measure chunk length (default: len)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            "! ",      # Exclamations
            "? ",      # Questions
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Words
            ""         # Characters (last resort)
        ]
        self.length_function = length_function or len
    
    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            document_id: Optional document identifier for metadata
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Perform recursive splitting
        splits = self._split_text(text, self.separators)
        
        # Merge small splits and create chunks with overlap
        chunks = self._merge_splits(splits)
        
        # Create Chunk objects with metadata
        result = []
        char_offset = 0
        
        for i, chunk_text in enumerate(chunks):
            # Find actual position in original text
            start_char = text.find(chunk_text[:50], char_offset)
            if start_char == -1:
                start_char = char_offset
            
            chunk = Chunk(
                text=chunk_text,
                chunk_index=i,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                document_id=document_id
            )
            result.append(chunk)
            
            # Move offset forward, accounting for overlap
            char_offset = start_char + len(chunk_text) - self.chunk_overlap
            if char_offset < 0:
                char_offset = 0
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the given separators.
        
        This is the core algorithm. It tries to split at the first separator,
        and if any resulting chunks are still too long, it recursively
        splits them with the remaining separators.
        """
        if not text:
            return []
        
        # Base case: text fits in one chunk
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # No more separators, must split at character level
        if not separators:
            return self._split_by_characters(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Empty separator means character-level split
            return self._split_by_characters(text)
        
        # Split at this separator
        splits = text.split(separator)
        
        # Process each split
        result = []
        for i, split in enumerate(splits):
            if not split:
                continue
            
            # Add separator back (except for first split)
            if i > 0 and separator not in [" ", ""]:
                split = separator + split
            
            if self.length_function(split) <= self.chunk_size:
                result.append(split)
            else:
                # Recursively split with remaining separators
                result.extend(self._split_text(split, remaining_separators))
        
        return result
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text into character-level chunks as a last resort."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into larger chunks while respecting size limits.
        Also handles overlap between chunks.
        """
        if not splits:
            return []
        
        merged = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk size
            if self.length_function(current_chunk) + self.length_function(split) > self.chunk_size:
                if current_chunk:
                    merged.append(current_chunk.strip())
                    
                    # Create overlap for next chunk
                    if self.chunk_overlap > 0:
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = overlap_text + split
                    else:
                        current_chunk = split
                else:
                    current_chunk = split
            else:
                current_chunk += split
        
        # Don't forget the last chunk
        if current_chunk.strip():
            merged.append(current_chunk.strip())
        
        return merged
    
    def _get_overlap(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at a word boundary
        overlap = text[-self.chunk_overlap:]
        space_idx = overlap.find(' ')
        if space_idx > 0:
            overlap = overlap[space_idx + 1:]
        
        return overlap


class SentenceAwareChunker(RecursiveCharacterChunker):
    """
    Enhanced chunker that tries harder to preserve sentence boundaries.
    
    This variant prioritizes keeping sentences intact when possible,
    which often produces more semantically coherent chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        # Sentence-ending patterns
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """Split text into sentence-aware chunks."""
        if not text or not text.strip():
            return []
        
        text = self._clean_text(text)
        
        # First, split into sentences
        sentences = self.sentence_pattern.split(text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's long enough
                if len(current_chunk) >= self.min_chunk_size:
                    end_char = start_char + len(current_chunk)
                    chunks.append(Chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        document_id=document_id
                    ))
                    chunk_index += 1
                    start_char = end_char - self.chunk_overlap
                    
                    # Start new chunk with overlap
                    overlap = self._get_overlap(current_chunk)
                    current_chunk = overlap + " " + sentence
                else:
                    # Current chunk too small, just append
                    current_chunk = potential_chunk
                
                # If single sentence is too long, use parent's splitting
                if len(current_chunk) > self.chunk_size:
                    sub_chunks = super().chunk(current_chunk, document_id)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_index = chunk_index
                        chunks.append(sub_chunk)
                        chunk_index += 1
                    current_chunk = ""
        
        # Don't forget the last chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                document_id=document_id
            ))
        
        return chunks

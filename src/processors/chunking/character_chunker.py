"""
Character-based chunking with semantic boundary awareness.
"""
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker


class CharacterChunker(BaseChunker):
    """
    Chunk content by character count with semantic boundary awareness.
    Falls back to this when no other chunking strategy is available.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize character chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try semantic boundaries first
        )
    
    def can_handle(self, metadata: Dict) -> bool:
        """Can handle any content (universal fallback)."""
        return True
    
    def chunk(self, content: str, source_id: str, metadata: Dict) -> List[Dict]:
        """
        Chunk content by characters with overlap.
        
        Args:
            content: Text content to chunk
            source_id: Unique identifier for the source
            metadata: Metadata to attach to chunks
        
        Returns:
            List of character-based chunks
        """
        chunks = self.text_splitter.split_text(content)
        
        formatted_chunks = []
        duration_sec = metadata.get('duration_sec', 0)
        chars_per_second = len(content) / max(duration_sec, 1) if duration_sec else 0.1
        
        char_position = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}-chunk-{i}"
            
            # Estimate timestamp based on character position
            timestamp = int(char_position / chars_per_second) if chars_per_second > 0 else 0
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_type': 'character',
                'timestamp': timestamp,  # Add timestamp for Q&A interface
                'total_chunks': len(chunks),
                'text_preview': chunk[:200]
            })
            
            char_position += len(chunk)
            
            formatted_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return formatted_chunks

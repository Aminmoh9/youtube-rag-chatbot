"""
Base chunker interface and implementations for content chunking.
"""
from typing import Dict, List
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract base class for content chunkers."""
    
    @abstractmethod
    def chunk(self, content: str, source_id: str, metadata: Dict) -> List[Dict]:
        """
        Chunk content into smaller pieces.
        
        Args:
            content: Text content to chunk
            source_id: Unique identifier for the source
            metadata: Metadata to attach to chunks
        
        Returns:
            List of chunk dictionaries with 'id', 'text', and 'metadata'
        """
        pass
    
    @abstractmethod
    def can_handle(self, metadata: Dict) -> bool:
        """
        Check if this chunker can handle the given content.
        
        Args:
            metadata: Content metadata
        
        Returns:
            True if this chunker should be used
        """
        pass

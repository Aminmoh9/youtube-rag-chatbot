"""
Smart chunker that selects the best strategy automatically.
"""
from typing import Dict, List
from .base_chunker import BaseChunker
from .chapter_chunker import ChapterChunker
from .character_chunker import CharacterChunker


class SmartChunker:
    """
    Intelligent chunker that selects the best strategy based on content metadata.
    Priority: Chapter-based > Character-based
    """
    
    def __init__(self):
        """Initialize with all available chunking strategies."""
        self.strategies: List[BaseChunker] = [
            ChapterChunker(min_chunk_length=50),
            CharacterChunker(chunk_size=1000, chunk_overlap=200)
        ]
    
    def chunk(self, content: str, source_id: str, metadata: Dict) -> List[Dict]:
        """
        Automatically select and apply the best chunking strategy.
        
        Args:
            content: Text content to chunk
            source_id: Unique identifier for the source
            metadata: Content metadata (may contain chapters, timestamps, etc.)
        
        Returns:
            List of chunks using the most appropriate strategy
        """
        # Try each strategy in priority order
        for strategy in self.strategies:
            if strategy.can_handle(metadata):
                chunks = strategy.chunk(content, source_id, metadata)
                if chunks:  # Successfully generated chunks
                    return chunks
        
        # Should never reach here since CharacterChunker always works
        # But as a safety fallback
        return CharacterChunker().chunk(content, source_id, metadata)
    
    def add_strategy(self, strategy: BaseChunker, priority: int = -1):
        """
        Add a custom chunking strategy.
        
        Args:
            strategy: Chunker instance implementing BaseChunker
            priority: Position in strategy list (0 = highest priority, -1 = append)
        """
        if priority == -1:
            self.strategies.append(strategy)
        else:
            self.strategies.insert(priority, strategy)

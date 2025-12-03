"""
Chapter-based chunking for content with chapter metadata.
"""
from typing import Dict, List
from .base_chunker import BaseChunker


class ChapterChunker(BaseChunker):
    """
    Chunk content based on video chapters.
    Each chapter becomes one semantically coherent chunk.
    """
    
    def __init__(self, min_chunk_length: int = 50):
        """
        Initialize chapter chunker.
        
        Args:
            min_chunk_length: Minimum characters for a valid chunk
        """
        self.min_chunk_length = min_chunk_length
    
    def can_handle(self, metadata: Dict) -> bool:
        """Check if content has valid chapters."""
        chapters = metadata.get('chapters', [])
        return isinstance(chapters, list) and len(chapters) >= 2
    
    def chunk(self, content: str, source_id: str, metadata: Dict) -> List[Dict]:
        """
        Chunk transcript based on video chapters.
        
        Args:
            content: Full transcript text
            source_id: Video ID or source identifier
            metadata: Must contain 'chapters' list and optionally 'duration_sec'
        
        Returns:
            List of chapter-based chunks
        """
        chunks = []
        chapters = metadata.get('chapters', [])
        duration_sec = metadata.get('duration_sec', 0)
        
        # Estimate characters per second (rough average for speech)
        chars_per_second = len(content) / max(duration_sec, 1) if duration_sec else 0.1
        
        for i, chapter in enumerate(chapters):
            start_sec = chapter['start']
            
            # Determine end time
            if 'end' in chapter and chapter['end'] is not None:
                end_sec = chapter['end']
            elif i + 1 < len(chapters):
                end_sec = chapters[i + 1]['start']
            else:
                end_sec = duration_sec or (start_sec + 300)  # Default 5 min if unknown
            
            # Calculate character positions
            start_char = int(start_sec * chars_per_second)
            end_char = int(end_sec * chars_per_second)
            
            # Extract chapter text
            chunk_text = content[start_char:end_char].strip()
            
            if chunk_text and len(chunk_text) >= self.min_chunk_length:
                chunk_id = f"{source_id}-chapter-{i}"
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'chunk_type': 'chapter',
                    'chapter_title': chapter.get('title', f'Chapter {i+1}'),
                    'start_time': start_sec,
                    'end_time': end_sec,
                    'timestamp': start_sec,  # Add timestamp for Q&A interface
                    'total_chunks': len(chapters),
                    'text_preview': chunk_text[:200]
                })
                
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
        
        return chunks

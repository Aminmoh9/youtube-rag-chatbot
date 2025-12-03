"""
Content processor - lightweight orchestrator delegating to input method processors.
Reduced from 523 lines to ~150 lines by extracting input method logic.
"""
from typing import Dict, Optional, List

# Core components
from src.embeddings.embedding_generator import EmbeddingGenerator
from .session_manager import SessionManager
from .chunking import SmartChunker, ChapterChunker, CharacterChunker
from .summarization_helper import SummarizationHelper
from src.utils.content_topic_extractor import ContentTopicExtractor
from src.qa.summarization_agent import SummarizationAgent
from src.embeddings.pinecone_topic_isolation import StrictTopicIsolation

# Input method processors
from .input_methods import (
    TopicSearchProcessor,
    YouTubeLinkProcessor,
    AudioVideoProcessor,
    ScriptUploadProcessor
)


class ContentProcessor:
    """
    Lightweight orchestrator for all 4 input methods.
    Delegates actual processing to specialized processors.
    """
    
    def __init__(self):
        """Initialize with shared components and input processors."""
        # Shared components
        self.session_manager = SessionManager()
        self.embedder = EmbeddingGenerator()
        self.topic_extractor = ContentTopicExtractor()
        self.summarization_helper = SummarizationHelper()
        self.summarizer = SummarizationAgent()
        self.isolation_manager = StrictTopicIsolation()
        
        # Initialize chunker with strategies
        self.chunker = SmartChunker()
        self.chunker.add_strategy(ChapterChunker(), priority=1)
        self.chunker.add_strategy(CharacterChunker(), priority=2)
        
        # Initialize specialized input processors
        self.topic_search_processor = TopicSearchProcessor(
            self.chunker, self.embedder, self.session_manager,
            self.summarization_helper, self.summarizer, self.isolation_manager
        )
        self.youtube_link_processor = YouTubeLinkProcessor(
            self.chunker, self.embedder, self.session_manager,
            self.summarizer, self.isolation_manager, self.topic_extractor
        )
        self.audio_video_processor = AudioVideoProcessor(
            self.chunker, self.embedder, self.session_manager,
            self.summarizer, self.isolation_manager, self.topic_extractor
        )
        self.script_upload_processor = ScriptUploadProcessor(
            self.chunker, self.embedder, self.session_manager,
            self.summarizer, self.isolation_manager, self.topic_extractor
        )
    
    # ============================================================
    # Public API - Delegates to specialized processors
    # ============================================================
    
    def process_topic_search(self, topic: str, max_videos: int = 5, 
                            require_consent: bool = True) -> Dict:
        """
        Method 1: Topic Search - Multiple YouTube videos.
        Delegates to TopicSearchProcessor.
        """
        return self.topic_search_processor.process(topic, max_videos, require_consent)
    
    def process_youtube_link(self, youtube_url: str, consent_given: bool = True, summary_only: bool = False) -> Dict:
        """
        Method 2: YouTube Link - Single video URL.
        Delegates to YouTubeLinkProcessor.
        If `summary_only` is True, the processor will only generate a summary and will not upsert embeddings
        or create a new session/namespace.
        """
        return self.youtube_link_processor.process(youtube_url, consent_given, summary_only=summary_only)

    def regenerate_video_summary(self, parent_session_id: str, video_id: str, consent_given: bool = True) -> Dict:
        """
        Regenerate summary for a single video and persist it into an existing parent session
        without creating a new session or upserting new embeddings.

        Args:
            parent_session_id: The session id of the topic session to update
            video_id: YouTube video id to regenerate
            consent_given: Whether to download audio if subtitles missing

        Returns:
            Result dict with success and summary
        """
        # Attempt to construct youtube url
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        # Generate summary only (no upsert)
        result = self.youtube_link_processor.process(youtube_url, consent_given=consent_given, summary_only=True)

        # Persist into parent session if possible
        try:
            parent = self.session_manager.get_session(parent_session_id)
            if not parent:
                return {'success': False, 'error': 'Parent session not found'}

            # Find video entry in either top-level 'video_summaries' or nested 'data.video_summaries'
            target_vs = parent.get('video_summaries')
            updated = False
            if isinstance(target_vs, list):
                for v in target_vs:
                    if v.get('video_id') == video_id:
                        v['summary'] = result.get('summary')
                        updated = True
                        break

            if not updated and isinstance(parent.get('data'), dict):
                target_vs = parent['data'].get('video_summaries')
                if isinstance(target_vs, list):
                    for v in target_vs:
                        if v.get('video_id') == video_id:
                            v['summary'] = result.get('summary')
                            updated = True
                            break

            if not updated:
                # Append an entry if not found
                parent.setdefault('video_summaries', [])
                parent['video_summaries'].append({
                    'video_id': video_id,
                    'title': result.get('title', ''),
                    'summary': result.get('summary')
                })

            self.session_manager.save_session(parent_session_id, parent)
        except Exception as e:
            return {'success': False, 'error': f'Failed to persist summary into parent session: {e}'}

        return result
    
    def process_audio_video_upload(self, file_bytes: bytes, filename: str,
                                   file_type: str, consent_given: bool = True) -> Dict:
        """
        Method 3: Audio/Video Upload - Local media files.
        Delegates to AudioVideoProcessor.
        """
        return self.audio_video_processor.process(file_bytes, filename, file_type, consent_given)
    
    def process_script_upload(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Method 4: Script Upload - Text/transcript files.
        Delegates to ScriptUploadProcessor.
        """
        return self.script_upload_processor.process(file_bytes, filename)
    
    # ============================================================
    # Query & Session Management
    # ============================================================
    
    def query_session(self, session_id: str, question: str, top_k: int = 5) -> Dict:
        """
        Query a processed session with a question.
        
        Args:
            session_id: Session ID to query
            question: User question
            top_k: Number of relevant chunks to retrieve
        
        Returns:
            Answer dictionary with sources
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {'success': False, 'error': 'Session not found'}
        
        # Query Pinecone with topic isolation using question text directly
        topic = session.get('topic', 'general')
        results = self.isolation_manager.query_with_isolation(
            query_text=question,
            topic=topic,
            top_k=top_k
        )
        
        if not results:
            return {
                'success': True,
                'answer': 'No relevant information found for your question.',
                'sources': []
            }
        
        # Format sources (results is already a list of matches)
        sources = []
        for match in results:
            sources.append({
                'text': match.get('text', ''),
                'score': match.get('score', 0.0),
                'metadata': match.get('metadata', {})
            })
        
        # Generate answer using summarizer
        context = '\n\n'.join([s['text'] for s in sources[:3]])
        answer_prompt = f"Question: {question}\n\nContext: {context}"
        answer_result = self.summarizer.summarize(answer_prompt, "executive")
        answer = answer_result.get('summary', 'Unable to generate answer.') if isinstance(answer_result, dict) else str(answer_result)
        
        return {
            'success': True,
            'answer': answer,
            'sources': sources
        }
    
    def get_detailed_summary(self, session_id: str) -> Optional[Dict]:
        """Get detailed summary for a session."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        return session.get('topic_summary') or session.get('summary')
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        return self.session_manager.get_session(session_id)
    
    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """List recent sessions."""
        return self.session_manager.list_sessions(limit)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.session_manager.delete_session(session_id)


# ============================================================
# Global instance for backward compatibility
# ============================================================
content_processor = ContentProcessor()

"""
Manages embeddings for user-provided content (not from YouTube search).
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.embeddings.pinecone_topic_isolation import StrictTopicIsolation
from src.utils.content_topic_extractor import ContentTopicExtractor

class ContentEmbeddingManager:
    """Manages embeddings for user-uploaded/linked content."""
    
    def __init__(self):
        self.isolation_manager = StrictTopicIsolation()
        self.topic_extractor = ContentTopicExtractor()
        
        # Session storage
        self.sessions_dir = Path("data/user_sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Active sessions in memory
        self.active_sessions = {}
    
    def process_youtube_link(self, youtube_url: str, consent_given: bool = True) -> Dict:
        """Process a single YouTube link provided by user."""
        from src.youtube.get_subtitles import SubtitleExtractor
        from src.transcription.whisper_agent import WhisperTranscriptionAgent
        
        # Extract video ID
        video_id = self._extract_video_id(youtube_url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}
        
        # Get metadata (simplified - in production, use YouTube API)
        metadata = self._get_video_metadata(video_id, youtube_url)
        
        # Try to get subtitles
        subtitle_extractor = SubtitleExtractor()
        transcript = subtitle_extractor.get_subtitles(video_id)
        
        # If no subtitles and consent given, download and transcribe
        if not transcript and consent_given:
            # Download audio
            from src.youtube.video_downloader import download_single_video_with_consent
            audio_path = download_single_video_with_consent(
                youtube_url,
                consent_given=consent_given
            )
            
            if audio_path:
                # Transcribe
                whisper_agent = WhisperTranscriptionAgent(model="small", use_local=True)
                result = whisper_agent.transcribe_file(audio_path)
                transcript = result.get('text', '') if result.get('success') else None
        
        if not transcript:
            return {"error": "No transcript available and consent not given"}
        
        # Extract topic from content
        content_data = {
            'source_type': 'youtube_link',
            'identifier': video_id,
            'title': metadata.get('title', 'YouTube Video'),
            'description': metadata.get('description', ''),
            'channel': metadata.get('channel', ''),
            'transcript': transcript,
            'content_type': 'video',
            'created_at': datetime.now().isoformat(),
            'content_length': len(transcript)
        }
        
        return self._process_content(content_data)
    
    def process_uploaded_file(self, file_path: str, file_type: str, 
                            original_filename: str, consent_given: bool = True) -> Dict:
        """Process uploaded audio/video/transcript file."""
        content_data = {
            'source_type': 'file_upload',
            'identifier': hashlib.md5(file_path.encode()).hexdigest()[:12],
            'filename': original_filename,
            'file_type': file_type,
            'created_at': datetime.now().isoformat()
        }
        
        # Process based on file type
        if file_type == 'transcript':
            # Read transcript
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            content_data['transcript'] = transcript
            content_data['content_type'] = 'text'
            content_data['content_length'] = len(transcript)
            
        elif file_type in ['audio', 'video'] and consent_given:
            # Transcribe
            from src.transcription.whisper_agent import WhisperTranscriptionAgent
            whisper_agent = WhisperTranscriptionAgent(model="base", use_local=True)
            result = whisper_agent.transcribe_file(file_path)
            
            if result.get('success'):
                transcript = result.get('text', '')
                content_data['transcript'] = transcript
                content_data['content_type'] = file_type
                content_data['content_length'] = len(transcript)
            else:
                return {"error": f"Transcription failed: {result.get('error', 'Unknown error')}"}
        else:
            return {"error": "Consent required for audio/video processing"}
        
        return self._process_content(content_data)
    
    def _process_content(self, content_data: Dict) -> Dict:
        """Common processing pipeline for all content types."""
        try:
            # Step 1: Create session
            session = self.topic_extractor.create_content_session(content_data)
            session_id = session['content_id']
            
            # Step 2: Chunk the transcript
            chunks = self._chunk_transcript(
                content_data.get('transcript', ''),
                content_data.get('title', 'User Content')
            )
            
            # Step 3: Generate embeddings
            embeddings = self._generate_embeddings(chunks, session)
            
            # Step 4: Store in Pinecone with content-specific isolation
            topic_name = session['topic_info'].get('main_topic', 'User Content')
            
            # Use namespace-based isolation
            pinecone_result = self.isolation_manager.upsert_with_isolation(
                vectors=embeddings,
                topic=topic_name
            )
            
            # Step 5: Store session
            session['pinecone_result'] = pinecone_result
            session['chunk_count'] = len(chunks)
            session['embedding_count'] = len(embeddings)
            
            self.active_sessions[session_id] = session
            self._save_session(session_id, session)
            
            # Step 6: Generate quick summary
            summary = self._generate_content_summary(content_data.get('transcript', ''))
            session['summary'] = summary
            
            return {
                "success": True,
                "session_id": session_id,
                "topic": topic_name,
                "summary": summary,
                "chunks_processed": len(chunks),
                "content_type": content_data.get('content_type', 'unknown'),
                "pinecone_index": pinecone_result.get('index')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content_type": content_data.get('content_type', 'unknown')
            }
    
    def _chunk_transcript(self, transcript: str, title: str) -> List[Dict]:
        """Chunk transcript for embedding."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(transcript)
        
        # Format chunks for embedding
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{hashlib.md5(title.encode()).hexdigest()[:8]}-{i}"
            
            formatted_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source': title
            })
        
        return formatted_chunks
    
    def _generate_embeddings(self, chunks: List[Dict], session: Dict) -> List[Dict]:
        """Generate embeddings for chunks."""
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = []
        
        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    input=chunk['text'],
                    model="text-embedding-ada-002"
                )
                
                embedding_vector = response.data[0].embedding
                
                # Prepare vector for Pinecone
                vector_data = {
                    'id': chunk['id'],
                    'values': embedding_vector,
                    'metadata': {
                        'text': chunk['text'],  # Store full text (Pinecone has 40KB metadata limit)
                        'text_preview': chunk['text'][:200],  # Short preview for display
                        'source': chunk['source'],
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks'],
                        'topic': session['topic_info'].get('main_topic', 'User Content'),
                        'content_type': session.get('content_type', 'unknown'),
                        'session_id': session['content_id'],
                        'is_user_content': True
                    }
                }
                
                embeddings.append(vector_data)
                
            except Exception as e:
                print(f"Error embedding chunk {chunk['id']}: {str(e)}")
                continue
        
        return embeddings
    
    def _generate_content_summary(self, transcript: str) -> str:
        """Generate quick summary of content."""
        from src.qa.summarization_agent import SummarizationAgent
        
        summarizer = SummarizationAgent()
        summary = summarizer.generate_summary(transcript[:3000], "short")
        
        return summary
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        import re
        
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([\w\-]+)',
            r'(?:youtu\.be\/)([\w\-]+)',
            r'(?:youtube\.com\/embed\/)([\w\-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _get_video_metadata(self, video_id: str, url: str) -> Dict:
        """Get basic video metadata (simplified)."""
        # In production, use YouTube API
        return {
            'video_id': video_id,
            'url': url,
            'title': f"YouTube Video: {video_id}",
            'channel': 'Unknown Channel',
            'description': 'User-provided YouTube video'
        }
    
    def _save_session(self, session_id: str, session_data: Dict):
        """Save session to disk."""
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load session from disk."""
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return None
    
    def query_content(self, session_id: str, question: str, top_k: int = 3) -> List[Dict]:
        """Query user-provided content."""
        if session_id not in self.active_sessions:
            # Try to load from disk
            session = self.load_session(session_id)
            if not session:
                return []
            self.active_sessions[session_id] = session
        
        session = self.active_sessions[session_id]
        topic = session['topic_info'].get('main_topic', 'User Content')
        
        # Query with strict isolation
        results = self.isolation_manager.query_with_isolation(
            query_text=question,
            topic=topic,
            use_separate_index=True,  # User content always in separate index
            top_k=top_k
        )
        
        # Filter by session ID
        filtered_results = [
            r for r in results 
            if r.get('metadata', {}).get('session_id') == session_id
        ]
        
        return filtered_results
    
    def cleanup_old_sessions(self, hours_old: int = 24):
        """Clean up old sessions."""
        import time
        current_time = time.time()
        
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            created_at = session.get('created_at')
            if created_at:
                # Parse ISO format time
                from datetime import datetime
                session_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                session_timestamp = session_time.timestamp()
                
                if current_time - session_timestamp > hours_old * 3600:
                    sessions_to_remove.append(session_id)
        
        # Remove from memory
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        # Remove from disk
        for session_file in self.sessions_dir.glob("*.json"):
            file_mtime = session_file.stat().st_mtime
            if current_time - file_mtime > hours_old * 3600:
                session_file.unlink()
        
        return len(sessions_to_remove)


# Global instance
content_manager = ContentEmbeddingManager()
"""
Topic search processor - handles multiple YouTube video research.
"""
import pandas as pd
from pathlib import Path
from typing import Dict

from src.youtube.fetch_metadata import YouTubeMetadataAgent
from src.youtube.get_subtitles import SubtitleExtractor
from src.transcription.whisper_agent import WhisperTranscriptionAgent


class TopicSearchProcessor:
    """Process topic search: fetch and analyze multiple YouTube videos."""
    
    def __init__(self, chunker, embedder, session_manager, summarization_helper, 
                 summarizer, isolation_manager):
        """Initialize with shared components."""
        self.chunker = chunker
        self.embedder = embedder
        self.session_manager = session_manager
        self.summarization_helper = summarization_helper
        self.summarizer = summarizer
        self.isolation_manager = isolation_manager
    
    def process(self, topic: str, max_videos: int = 5, 
                require_consent: bool = True) -> Dict:
        """
        Process topic search → fetch multiple YouTube videos.
        
        Args:
            topic: Search topic
            max_videos: Maximum number of videos to fetch
            require_consent: Whether to require user consent for audio downloads
        
        Returns:
            Processing result dictionary
        """
        # Generate namespace for topic isolation
        namespace = self.isolation_manager.get_topic_namespace(topic)
        
        # Generate session ID
        session_id = self.session_manager.generate_session_id(f"topic:{topic}")
        
        # Step 1: Fetch videos
        metadata_agent = YouTubeMetadataAgent(topic=topic, max_results=max_videos)
        videos_df = metadata_agent.run()
        
        if videos_df.empty:
            return {
                'success': False,
                'error': f'No videos found for topic: {topic}',
                'session_id': session_id
            }
        
        # Step 2: Extract subtitles (with increased rate limiting - 8 seconds)
        subtitle_extractor = SubtitleExtractor(delay_between_requests=8.0)
        videos_df = subtitle_extractor.batch_extract()
        
        # Step 3: Transcribe audio if needed (optional)
        whisper_agent = WhisperTranscriptionAgent(model="small", use_local=True)
        whisper_agent.batch_transcribe(require_consent=require_consent)
        
        # Step 4: Process each video
        all_chunks = []
        video_summaries = []
        videos_without_transcripts = []
        
        for _, video_row in videos_df.iterrows():
            video_id = video_row['video_id']
            transcript_path = video_row.get('local_transcript_path') or video_row.get('transcript_path')
            
            # Skip videos without transcripts
            if pd.isna(transcript_path) or not transcript_path or not Path(transcript_path).exists():
                videos_without_transcripts.append({
                    'video_id': video_id,
                    'title': video_row.get('title', 'Unknown'),
                    'url': video_row.get('url', '')
                })
                continue
            
            # Read transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            if not transcript or len(transcript.strip()) < 50:
                videos_without_transcripts.append({
                    'video_id': video_id,
                    'title': video_row.get('title', 'Unknown'),
                    'url': video_row.get('url', '')
                })
                continue
            
            # Generate video summary
            video_summary = self.summarizer.summarize_video(video_id, transcript_path)
            video_summaries.append({
                'video_id': video_id,
                'title': video_row['title'],
                'summary': video_summary,
                'duration': video_row.get('duration_sec', 0),
                'chapters': video_row.get('chapters') if isinstance(video_row.get('chapters'), list) else [],
                'num_chapters': video_row.get('num_chapters', 0)
            })
            
            # Chunk transcript with chapter support
            chapters_data = video_row.get('chapters')
            metadata = {
                'title': video_row['title'],
                'channel': video_row['channel'],
                'url': video_row['url'],
                'video_id': video_id,
                'input_method': 'topic_search',
                'chapters': chapters_data if isinstance(chapters_data, list) else [],
                'duration_sec': video_row.get('duration_sec', 0)
            }
            
            chunks = self.chunker.chunk(transcript, video_id, metadata)
            all_chunks.extend(chunks)
        
        # Check if we have any valid content
        if not all_chunks:
            return {
                'success': False,
                'error': f'No transcripts available for any of the {len(videos_df)} videos found. ' +
                        f'Videos without transcripts: {len(videos_without_transcripts)}. ' +
                        'This usually means the videos don\'t have captions enabled.',
                'session_id': session_id,
                'videos_without_transcripts': videos_without_transcripts
            }
        
        # Step 5: Generate overall topic summary
        topic_summary = self.summarization_helper.generate_topic_summary(video_summaries, topic)
        
        # Step 6: Embed and store
        embeddings = self.embedder.generate_embeddings(all_chunks, {
            'topic': topic,
            'input_method': 'topic_search',
            'video_count': len(video_summaries)
        })
        
        # Store with topic isolation
        pinecone_result = self.isolation_manager.upsert_with_isolation(
            vectors=embeddings,
            topic=topic
        )
        
        # Create session
        session = {
            'session_id': session_id,
            'input_method': 'topic_search',
            'topic': topic,
            'video_count': len(video_summaries),
            'chunk_count': len(all_chunks),
            'video_summaries': video_summaries,
            'topic_summary': topic_summary,
            'pinecone_result': pinecone_result,
            'content_type': 'youtube_videos',
            'status': 'processed'
        }
        
        self.session_manager.save_session(session_id, session)
        
        result = {
            'success': True,
            'session_id': session_id,
            'input_method': 'topic_search',
            'topic': topic,
            'video_count': len(video_summaries),
            'summary': topic_summary,
            'video_summaries': video_summaries,
            'can_query': True,
            'namespace': namespace,
            'loaded_from_pinecone': False
        }
        
        return result

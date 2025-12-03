"""
YouTube link processor - handles single YouTube video analysis.
"""
from typing import Dict, List
from src.youtube.get_subtitles import SubtitleExtractor
from src.utils.content_topic_extractor import ContentTopicExtractor


class YouTubeLinkProcessor:
    """Process single YouTube video link."""
    
    def __init__(self, chunker, embedder, session_manager, summarizer, 
                 isolation_manager, topic_extractor):
        """Initialize with shared components."""
        self.chunker = chunker
        self.embedder = embedder
        self.session_manager = session_manager
        self.summarizer = summarizer
        self.isolation_manager = isolation_manager
        self.topic_extractor = topic_extractor
    
    def process(self, youtube_url: str, consent_given: bool = True, summary_only: bool = False) -> Dict:
        """
        Process single YouTube link.
        
        Args:
            youtube_url: YouTube video URL
            consent_given: Whether user consents to audio download if needed
        
        Returns:
            Processing result dictionary
        """
        # Extract video ID
        video_id = self._extract_video_id(youtube_url)
        if not video_id:
            return {'success': False, 'error': 'Invalid YouTube URL'}
        
        session_id = self.session_manager.generate_session_id(f"youtube:{video_id}")
        
        # Try to get timed subtitles
        subtitle_extractor = SubtitleExtractor()
        timed_subtitles = subtitle_extractor.get_timed_subtitles(video_id)
        
        if not timed_subtitles:
            # If no subtitles and consent given, download and transcribe
            if consent_given:
                from src.youtube.video_downloader import download_single_video_with_consent
                from src.transcription.whisper_agent import WhisperTranscriptionAgent
                
                audio_path = download_single_video_with_consent(youtube_url, consent_given=consent_given)
                
                if audio_path:
                    # Transcribe with Whisper
                    whisper_agent = WhisperTranscriptionAgent(model="small", use_local=True)
                    result = whisper_agent.transcribe_file(audio_path)
                    
                    if result.get('success'):
                        # Convert Whisper output to timed subtitle format
                        transcript_text = result.get('text', '')
                        segments = result.get('segments', [])
                        
                        if segments:
                            timed_subtitles = [
                                {
                                    'start': seg['start'],
                                    'duration': seg['end'] - seg['start'],
                                    'text': seg['text']
                                }
                                for seg in segments
                            ]
                        else:
                            return {'success': False, 'error': 'Transcription failed'}
                    else:
                        return {'success': False, 'error': 'Failed to transcribe audio'}
                else:
                    return {'success': False, 'error': 'Failed to download audio'}
            else:
                return {'success': False, 'error': 'No transcript available for this video'}
        
        # Combine text for processing
        transcript = ' '.join([entry['text'] for entry in timed_subtitles])
        
        # Fetch video metadata to get actual title
        video_title = self._fetch_video_title(video_id)
        
        # Try to get video chapters
        chapters = self._get_video_chapters(video_id)
        
        # Extract topic from video title
        topic_data = self.topic_extractor.extract_from_youtube_metadata(
            title=video_title,
            description="User-provided YouTube link"
        )
        main_topic = topic_data.get('main_topic', video_title)
        
        # Generate summary using the summarizer's file-based flow (uses fallback when needed)
        try:
            from pathlib import Path
            transcripts_dir = Path("data/transcripts")
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            transcript_file = transcripts_dir / f"{video_id}.txt"
            with open(transcript_file, 'w', encoding='utf-8') as tf:
                tf.write(transcript)

            summary_result = self.summarizer.summarize_video(video_id, str(transcript_file))
            summary = summary_result if isinstance(summary_result, dict) else {'success': True, 'short_summary': str(summary_result)}
        except Exception as e:
            summary = {'success': False, 'error': str(e), 'short_summary': '', 'detailed_summary': ''}
        
        # Calculate video duration
        video_duration = timed_subtitles[-1]['start'] + timed_subtitles[-1].get('duration', 0) if timed_subtitles else 0
        
        # If caller only requested a summary (regeneration), skip embedding/upsert and session creation
        if summary_only:
            return {
                'success': True if isinstance(summary, dict) and summary.get('success') else False,
                'video_id': video_id,
                'summary': summary
            }

        # Chunk content - use chapters if available, character-based for short videos, time-based for long videos
        if chapters:
            print(f"✅ Found {len(chapters)} chapters, using chapter-based chunking")
            chunks = self._chunk_by_chapters(timed_subtitles, chapters, video_id, youtube_url, video_title)
        elif video_duration < 600:  # Less than 10 minutes
            print(f"📝 Short video ({video_duration/60:.1f} min), using character-based chunking for better Q&A")
            chunks = self._chunk_by_characters(transcript, video_id, youtube_url, video_title, timed_subtitles)
        else:
            print(f"⏱️ Long video ({video_duration/60:.1f} min), using time-based chunking (5-min intervals)")
            chunks = self._chunk_by_time(timed_subtitles, video_id, youtube_url, video_title)
        
        # Embed and store
        embeddings = self.embedder.generate_embeddings(chunks, {
            'topic': main_topic,
            'input_method': 'youtube_link',
            'source': youtube_url
        })
        
        pinecone_result = self.isolation_manager.upsert_with_isolation(
            vectors=embeddings,
            topic=main_topic
        )
        
        # Get namespace for immediate Q&A
        namespace = self.isolation_manager.get_topic_namespace(main_topic)
        
        # Create session
        session = {
            'session_id': session_id,
            'input_method': 'youtube_link',
            'url': youtube_url,
            'video_id': video_id,
            'topic': main_topic,
            'summary': summary,
            'chunk_count': len(chunks),
            'pinecone_result': pinecone_result,
            'content_type': 'youtube_video',
            'status': 'processed',
            'transcript_preview': transcript[:500]
        }
        
        self.session_manager.save_session(session_id, session)
        
        return {
            'success': True,
            'session_id': session_id,
            'input_method': 'youtube_link',
            'topic': main_topic,
            'summary': summary,
            'can_query': True,
            'content_type': 'YouTube Video',
            'namespace': namespace,
            'loaded_from_pinecone': False
        }
    
    def _extract_video_id(self, url: str) -> str:
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
    
    def _fetch_video_title(self, video_id: str) -> str:
        """Fetch video title from YouTube API."""
        try:
            import os
            from googleapiclient.discovery import build
            
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                return f"YouTube Video {video_id}"
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            response = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if response.get('items'):
                title = response['items'][0]['snippet']['title']
                # Clean title (remove special chars that might cause issues)
                title = title.strip()
                return title if title else f"YouTube Video {video_id}"
            
            return f"YouTube Video {video_id}"
        except Exception as e:
            print(f"Error fetching video title: {e}")
            return f"YouTube Video {video_id}"
    
    def _get_video_chapters(self, video_id: str) -> List[Dict]:
        """
        Get video chapters from YouTube API.
        
        Returns:
            List of chapters with start_time and title, or empty list if no chapters
        """
        try:
            import os
            from googleapiclient.discovery import build
            
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                return []
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            response = youtube.videos().list(
                part='contentDetails',
                id=video_id
            ).execute()
            
            if not response.get('items'):
                return []
            
            # Check description for chapters (format: 0:00 Chapter Title)
            response = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not response.get('items'):
                return []
            
            description = response['items'][0]['snippet'].get('description', '')
            
            # Parse chapters from description
            import re
            chapters = []
            lines = description.split('\n')
            
            for line in lines:
                # Match timestamps like 0:00, 1:23, 12:34:56
                match = re.match(r'^(\d+:?\d*:?\d+)\s+(.+)$', line.strip())
                if match:
                    time_str, title = match.groups()
                    
                    # Convert time to seconds
                    time_parts = time_str.split(':')
                    if len(time_parts) == 2:  # MM:SS
                        seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                    elif len(time_parts) == 3:  # HH:MM:SS
                        seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                    else:  # Just seconds
                        seconds = int(time_parts[0])
                    
                    chapters.append({
                        'start_time': seconds,
                        'title': title.strip()
                    })
            
            # Only return if we found at least 2 chapters
            return chapters if len(chapters) >= 2 else []
            
        except Exception as e:
            print(f"Error getting video chapters: {e}")
            return []
    
    def _build_timestamp_map(self, timed_subtitles: List[Dict]) -> List[tuple]:
        """Build a map of character positions to timestamps."""
        char_position = 0
        timestamp_map = []
        
        for entry in timed_subtitles:
            text = entry['text']
            start_time = entry['start']
            timestamp_map.append((char_position, start_time))
            char_position += len(text) + 1  # +1 for space
        
        return timestamp_map
    
    def _get_timestamp_for_position(self, pos: int, timestamp_map: List[tuple]) -> int:
        """Get timestamp (in seconds) for a character position."""
        for i, (char_pos, timestamp) in enumerate(timestamp_map):
            if char_pos > pos:
                # Return previous timestamp
                return timestamp_map[i-1][1] if i > 0 else 0
        
        # If position is beyond all entries, return last timestamp
        return timestamp_map[-1][1] if timestamp_map else 0
    
    def _chunk_by_chapters(self, timed_subtitles: List[Dict], chapters: List[Dict], 
                          video_id: str, youtube_url: str, video_title: str) -> List[Dict]:
        """Chunk transcript by video chapters."""
        chunks = []
        
        for i, chapter in enumerate(chapters):
            chapter_start = chapter['start_time']
            chapter_end = chapters[i + 1]['start_time'] if i + 1 < len(chapters) else float('inf')
            chapter_title = chapter['title']
            
            # Get all subtitle entries for this chapter
            chapter_text = []
            for entry in timed_subtitles:
                if chapter_start <= entry['start'] < chapter_end:
                    chapter_text.append(entry['text'])
            
            if chapter_text:
                text = ' '.join(chapter_text)
                
                chunk = {
                    'id': f"{video_id}_chapter_{i}",
                    'text': text,
                    'metadata': {
                        'title': video_title,
                        'url': youtube_url,
                        'video_url': youtube_url,
                        'video_id': video_id,
                        'input_method': 'youtube_link',
                        'has_timestamps': True,
                        'timestamp': chapter_start,
                        'chapter_title': chapter_title,
                        'chunk_type': 'chapter',
                        'text_preview': text[:200]
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_characters(self, transcript: str, video_id: str, 
                            youtube_url: str, video_title: str, 
                            timed_subtitles: List[Dict]) -> List[Dict]:
        """Chunk transcript by character count for short videos."""
        # For short videos, use smaller chunk size for more granular timestamps
        # 1500 chars ≈ 1.5-2 minutes of speech
        from src.processors.chunking.character_chunker import CharacterChunker
        short_video_chunker = CharacterChunker(chunk_size=1500, chunk_overlap=300)
        
        base_chunks = short_video_chunker.chunk(transcript, video_id, {
            'title': video_title,
            'url': youtube_url,
            'video_id': video_id,
            'input_method': 'youtube_link'
        })
        
        print(f"Created {len(base_chunks)} character-based chunks")
        
        # Build character position to timestamp map
        char_to_timestamp = self._build_timestamp_map(timed_subtitles)
        
        # Add timestamps to each chunk
        for i, chunk in enumerate(base_chunks):
            # Find where this chunk starts in the full transcript
            chunk_text_start = chunk['text'][:100]  # First 100 chars
            chunk_start_pos = transcript.find(chunk_text_start)
            
            if chunk_start_pos >= 0:
                timestamp = self._get_timestamp_for_position(chunk_start_pos, char_to_timestamp)
            else:
                timestamp = 0
            
            # Update metadata
            chunk['metadata']['video_url'] = youtube_url
            chunk['metadata']['timestamp'] = timestamp
            chunk['metadata']['has_timestamps'] = True
            chunk['metadata']['chunk_type'] = 'character_based'
        
        return base_chunks
    
    def _chunk_by_time(self, timed_subtitles: List[Dict], video_id: str, 
                      youtube_url: str, video_title: str, chunk_duration: int = 300) -> List[Dict]:
        """Chunk transcript by time intervals (default 5 minutes)."""
        chunks = []
        current_chunk = []
        chunk_start_time = 0
        chunk_index = 0
        
        for entry in timed_subtitles:
            # Check if we should start a new chunk
            if entry['start'] - chunk_start_time >= chunk_duration:
                if current_chunk:
                    # Save current chunk
                    text = ' '.join(current_chunk)
                    chunk = {
                        'id': f"{video_id}_time_{chunk_index}",
                        'text': text,
                        'metadata': {
                            'title': video_title,
                            'url': youtube_url,
                            'video_url': youtube_url,
                            'video_id': video_id,
                            'input_method': 'youtube_link',
                            'has_timestamps': True,
                            'timestamp': chunk_start_time,
                            'chunk_type': 'time_based',
                            'text_preview': text[:200]
                        }
                    }
                    chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk = []
                    chunk_start_time = int(entry['start'])
                    chunk_index += 1
            
            current_chunk.append(entry['text'])
        
        # Add final chunk
        if current_chunk:
            text = ' '.join(current_chunk)
            chunk = {
                'id': f"{video_id}_time_{chunk_index}",
                'text': text,
                'metadata': {
                    'title': video_title,
                    'url': youtube_url,
                    'video_url': youtube_url,
                    'video_id': video_id,
                    'input_method': 'youtube_link',
                    'has_timestamps': True,
                    'timestamp': chunk_start_time,
                    'chunk_type': 'time_based',
                    'text_preview': text[:200]
                }
            }
            chunks.append(chunk)
        
        return chunks

"""
Subtitle Extractor - Legal extraction of YouTube subtitles/captions.
"""
import os
import pandas as pd
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from typing import Optional, List, Dict
from pathlib import Path


class SubtitleExtractor:
    """Extract subtitles/transcripts from YouTube videos legally."""
    
    def __init__(self, delay_between_requests: float = 2.0):
        """
        Initialize SubtitleExtractor.
        
        Args:
            delay_between_requests: Seconds to wait between API calls (default: 2.0)
        """
        self.transcripts_dir = Path("data/transcripts")
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.delay_between_requests = delay_between_requests

    def _human_delay(self) -> float:
        """Return a randomized, human-like delay based on the configured base.

        This avoids calling with a strict, repeating pattern (e.g. 2,4,6,8...) by
        applying jitter and a small random multiplier to the base delay. Uses a
        combination of uniform and fractional jitter to appear non-systematic.
        """
        import random
        base = max(0.1, float(self.delay_between_requests or 0.1))
        # Apply a random multiplier between 0.6x and 1.8x and a small uniform jitter
        multiplier = random.uniform(0.6, 1.8)
        jitter = random.uniform(0.0, 0.6)
        return base * multiplier + jitter
    
    def get_subtitles(self, video_id: str, languages: List[str] = None) -> Optional[str]:
        """
        Get subtitles for a single video.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages (default: ['en'])
            
        Returns:
            Subtitle text or None if not available
        """
        if languages is None:
            languages = ['en']
        
        # Add a human-like randomized delay BEFORE every request to avoid
        # producing a detectable fixed pattern of requests.
        time.sleep(self._human_delay())
        
        # Retry logic with exponential backoff
        for attempt in range(5):
            try:
                # Get transcript using instance method
                api = YouTubeTranscriptApi()
                transcript_data = api.fetch(video_id, languages=languages)
                
                # Extract text from snippets
                if transcript_data and hasattr(transcript_data, 'snippets'):
                    full_text = ' '.join([snippet.text for snippet in transcript_data.snippets])
                else:
                    # Fallback: assume it's a list of dicts
                    full_text = ' '.join([entry.get('text', '') for entry in transcript_data])
                
                # Save transcript
                self._save_transcript(video_id, full_text)
                
                return full_text
                
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                print(f"No subtitles available for video {video_id}: {e}")
                return None
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "blocked" in error_msg:
                    # Use jittered exponential backoff to avoid strict retry timing.
                    import random
                    base_wait = 1.0
                    exp = 2 ** attempt
                    # multiplier between 0.75 and 1.6
                    multiplier = random.uniform(0.75, 1.6)
                    wait = base_wait * exp * multiplier + random.uniform(0.0, 0.8)
                    print(f"⚠️ Rate limit on {video_id}. Retrying in {wait:.1f}s... (attempt {attempt + 1}/5)")
                    time.sleep(wait)
                    continue
                else:
                    print(f"Error extracting subtitles for video {video_id}: {e}")
                    return None
        
        print(f"❌ Failed after 5 retries: {video_id}")
        return None
    
    def batch_extract(self, csv_path: str = "data/video_links.csv") -> pd.DataFrame:
        """
        Extract subtitles for all videos in CSV file.
        
        Args:
            csv_path: Path to CSV file with video_id column
            
        Returns:
            Updated DataFrame with has_subtitles column
        """
        # Load CSV
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        
        if 'video_id' not in df.columns:
            print("CSV file must have 'video_id' column")
            return df
        
        # Add has_subtitles column
        df['has_subtitles'] = False
        df['transcript_path'] = ''
        
        # Extract subtitles for each video
        successful = 0
        failed = 0
        
        for idx, row in df.iterrows():
            video_id = row['video_id']
            print(f"Processing {video_id}... ({idx + 1}/{len(df)})")
            
            # Delay is now handled inside get_subtitles() with jitter
            subtitle_text = self.get_subtitles(video_id)
            
            if subtitle_text:
                df.at[idx, 'has_subtitles'] = True
                df.at[idx, 'transcript_path'] = f"data/transcripts/{video_id}_transcript.txt"
                successful += 1
                print(f"✅ Subtitles found for {video_id} ({successful}/{idx + 1})")
            else:
                df.at[idx, 'has_subtitles'] = False
                failed += 1
                print(f"⚠️ No subtitles for {video_id} ({failed} failed)")
        
        print(f"\n📊 Summary: {successful} successful, {failed} failed out of {len(df)} videos")
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        
        return df
    
    def get_available_languages(self, video_id: str) -> List[str]:
        """
        Get list of available subtitle languages for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of available language codes
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            languages = []
            for transcript in transcript_list:
                languages.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            return languages
            
        except Exception as e:
            print(f"Error getting available languages: {e}")
            return []
    
    def get_timed_subtitles(self, video_id: str, languages: List[str] = None) -> Optional[List[Dict]]:
        """
        Get subtitles with timestamps.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages
            
        Returns:
            List of subtitle entries with timestamps or None
        """
        if languages is None:
            languages = ['en']
        
        try:
            # Use instance method
            api = YouTubeTranscriptApi()
            transcript_result = api.fetch(video_id, languages=languages)
            
            # Convert to list of dicts with 'text', 'start', 'duration'
            if hasattr(transcript_result, 'snippets'):
                transcript_data = [
                    {'text': snippet.text, 'start': snippet.start, 'duration': snippet.duration}
                    for snippet in transcript_result.snippets
                ]
            else:
                # Already in correct format
                transcript_data = transcript_result
            
            print(f"Successfully got {len(transcript_data)} timed subtitle entries")
            return transcript_data
            
        except NoTranscriptFound:
            print(f"No transcript found for video {video_id}")
            return None
        except TranscriptsDisabled:
            print(f"Transcripts disabled for video {video_id}")
            return None
        except Exception as e:
            print(f"Error getting timed subtitles: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_transcript(self, video_id: str, text: str):
        """
        Save transcript to file.
        
        Args:
            video_id: YouTube video ID
            text: Transcript text
        """
        transcript_path = self.transcripts_dir / f"{video_id}_transcript.txt"
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved transcript: {transcript_path}")
    
    def load_transcript(self, video_id: str) -> Optional[str]:
        """
        Load existing transcript from file.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Transcript text or None
        """
        transcript_path = self.transcripts_dir / f"{video_id}_transcript.txt"
        
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        return None
    
    def has_transcript(self, video_id: str) -> bool:
        """
        Check if transcript file exists.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if transcript exists
        """
        transcript_path = self.transcripts_dir / f"{video_id}_transcript.txt"
        return transcript_path.exists()

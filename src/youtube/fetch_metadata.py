"""
YouTube Metadata Agent - Fetches video metadata using YouTube Data API v3.
"""
import os
import pandas as pd
from googleapiclient.discovery import build
from typing import Optional, List, Dict
import re


class YouTubeMetadataAgent:
    """Agent for fetching YouTube video metadata."""
    
    def __init__(self, topic: str = None, max_results: int = 5):
        """
        Initialize YouTube Metadata Agent.
        
        Args:
            topic: Search topic for videos
            max_results: Maximum number of videos to fetch
        """
        self.topic = topic
        self.max_results = max_results
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")
        
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.videos_data = []
    
    def run(self) -> pd.DataFrame:
        """
        Main method to fetch videos and return as DataFrame.
        
        Returns:
            DataFrame with video metadata
        """
        videos = self._fetch_videos_directly()
        
        if not videos:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(videos)
        
        # Add timestamp for historical tracking
        from datetime import datetime
        df['search_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['search_topic'] = self.topic
        
        # Save current batch (overwrites)
        current_csv_path = "data/video_links.csv"
        df.to_csv(current_csv_path, index=False)
        
        # Append to historical CSV (preserves all searches)
        history_csv_path = "data/video_links_history.csv"
        if os.path.exists(history_csv_path):
            # Append without header
            df.to_csv(history_csv_path, mode='a', header=False, index=False)
        else:
            # Create new file with header
            df.to_csv(history_csv_path, index=False)
        
        return df
    
    def _fetch_videos_directly(self) -> List[Dict]:
        """
        Fetch videos using YouTube API.
        
        Returns:
            List of video dictionaries
        """
        if not self.topic:
            return []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=self.topic,
                part='id,snippet',
                maxResults=self.max_results,
                type='video',
                order='relevance',
                videoCategoryId='27',  # Education category (optional)
                relevanceLanguage='en'
            ).execute()
            
            videos = []
            video_ids = []
            
            # Extract video IDs
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                video_ids.append(video_id)
            
            # Get detailed video information
            if video_ids:
                videos_response = self.youtube.videos().list(
                    part='snippet,contentDetails,statistics',
                    id=','.join(video_ids)
                ).execute()
                
                for item in videos_response.get('items', []):
                    video_id = item['id']
                    snippet = item['snippet']
                    content_details = item['contentDetails']
                    statistics = item.get('statistics', {})
                    
                    # Parse duration
                    duration_sec = self._parse_duration(content_details.get('duration', 'PT0S'))
                    
                    # Extract chapters from description
                    description = snippet.get('description', '')
                    chapters = self._extract_chapters_from_description(description, duration_sec)
                    
                    video_data = {
                        'video_id': video_id,
                        'title': snippet.get('title', ''),
                        'channel': snippet.get('channelTitle', ''),
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'description': description,
                        'published_at': snippet.get('publishedAt', ''),
                        'duration_sec': duration_sec,
                        'duration': self._format_duration(duration_sec),
                        'view_count': int(statistics.get('viewCount', 0)),
                        'like_count': int(statistics.get('likeCount', 0)),
                        'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                        'has_subtitles': False,  # Will be updated by SubtitleExtractor
                        'chapters': chapters,
                        'num_chapters': len(chapters) if chapters else 0
                    }
                    
                    videos.append(video_data)
            
            return videos
            
        except Exception as e:
            print(f"Error fetching YouTube videos: {e}")
            return []
    
    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration to seconds.
        
        Args:
            duration_str: Duration string like 'PT1H2M30S'
            
        Returns:
            Duration in seconds
        """
        # Parse PT1H2M30S format
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def _format_duration(self, seconds: int) -> str:
        """
        Format seconds to HH:MM:SS or MM:SS.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """
        Get detailed information for a single video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video details dictionary or None
        """
        try:
            response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()
            
            if not response.get('items'):
                return None
            
            item = response['items'][0]
            snippet = item['snippet']
            content_details = item['contentDetails']
            statistics = item.get('statistics', {})
            
            duration_sec = self._parse_duration(content_details.get('duration', 'PT0S'))
            
            description = snippet.get('description', '')
            chapters = self._extract_chapters_from_description(description, duration_sec)
            
            return {
                'video_id': video_id,
                'title': snippet.get('title', ''),
                'channel': snippet.get('channelTitle', ''),
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'description': description,
                'published_at': snippet.get('publishedAt', ''),
                'duration_sec': duration_sec,
                'duration': self._format_duration(duration_sec),
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'chapters': chapters,
                'num_chapters': len(chapters) if chapters else 0
            }
            
        except Exception as e:
            print(f"Error fetching video details: {e}")
            return None
    
    def _extract_chapters_from_description(self, description: str, duration_sec: int) -> Optional[List[Dict]]:
        """
        Extract chapter timestamps and titles from video description.
        Many creators put chapters in the description like:
        0:00 Introduction
        2:30 Getting Started
        5:15 Advanced Topics
        
        Args:
            description: Video description
            duration_sec: Total video duration in seconds
        
        Returns:
            List of chapter dictionaries or None
        """
        if not description:
            return None
        
        # Pattern for timestamps: 0:00, 00:00, 0:00:00
        timestamp_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s+(.+?)(?:\n|$)'
        matches = re.findall(timestamp_pattern, description, re.MULTILINE)
        
        if len(matches) < 2:  # Need at least 2 chapters to be valid
            return None
        
        chapters = []
        for match in matches:
            hours = 0
            if match[2]:  # Has hours (HH:MM:SS)
                hours = int(match[0])
                minutes = int(match[1])
                seconds = int(match[2])
            else:  # Only minutes:seconds (MM:SS)
                minutes = int(match[0])
                seconds = int(match[1])
            
            start_time = hours * 3600 + minutes * 60 + seconds
            chapter_title = match[3].strip()
            
            # Clean up title (remove common prefixes like bullets, numbers)
            chapter_title = re.sub(r'^[\-\–\—\•\*\#\d\.\)\]]+\s*', '', chapter_title)
            
            chapters.append({
                'title': chapter_title,
                'start': start_time,
                'end': None  # Will be set to next chapter's start
            })
        
        # Set end times
        for i in range(len(chapters) - 1):
            chapters[i]['end'] = chapters[i + 1]['start']
        
        # Last chapter ends at video end
        if chapters:
            chapters[-1]['end'] = duration_sec
        
        return chapters if chapters else None

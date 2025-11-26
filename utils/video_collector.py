"""
YouTube video metadata collection using YouTube Data API.
Supports playlist extraction for curated video collections.
"""
import pandas as pd
from pathlib import Path
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
import json

load_dotenv()


def extract_playlist_id(playlist_url):
    """
    Extract playlist ID from YouTube playlist URL.
    
    Args:
        playlist_url (str): YouTube playlist URL
    
    Returns:
        str: Playlist ID
    """
    if 'list=' in playlist_url:
        return playlist_url.split('list=')[1].split('&')[0]
    return playlist_url


def get_videos_from_playlist(playlist_url, max_results=None):
    """
    Extract all videos from a YouTube playlist.
    
    Args:
        playlist_url (str): YouTube playlist URL
        max_results (int): Maximum number of videos to fetch (None = all)
    
    Returns:
        pandas.DataFrame: DataFrame with video metadata
    """
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in .env file")
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    playlist_id = extract_playlist_id(playlist_url)
    print(f"Extracting videos from playlist: {playlist_id}")
    
    video_data = []
    next_page_token = None
    
    while True:
        # Get playlist items
        playlist_response = youtube.playlistItems().list(
            playlistId=playlist_id,
            part='snippet,contentDetails',
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        
        video_ids = []
        for item in playlist_response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            video_ids.append(video_id)
            
            # Store basic info
            video_data.append({
                'video_id': video_id,
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'description': item['snippet']['description'],
                'published_at': item['snippet']['publishedAt'],
                'position': item['snippet']['position']
            })
        
        # Check if we've reached max_results
        if max_results and len(video_data) >= max_results:
            video_data = video_data[:max_results]
            break
        
        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break
    
    print(f"✓ Found {len(video_data)} videos in playlist")
    
    # Get additional video details (duration, chapters, captions)
    print("Fetching additional video details...")
    video_data = enrich_video_metadata(youtube, video_data)
    
    # Save to CSV
    df = pd.DataFrame(video_data)
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/video_links.csv', index=False)
    
    print(f"✓ Saved {len(video_data)} videos to data/video_links.csv")
    print(f"\nPreview:")
    print(df[['title', 'channel', 'topic_tag']].head(10))
    
    return df


def enrich_video_metadata(youtube, video_data):
    """
    Enrich video data with duration, chapters, captions, and topic tags.
    
    Args:
        youtube: YouTube API client
        video_data (list): List of video dictionaries
    
    Returns:
        list: Enriched video data
    """
    # Process in batches of 50 (API limit)
    for i in range(0, len(video_data), 50):
        batch = video_data[i:i+50]
        video_ids = [v['video_id'] for v in batch]
        
        # Get video details
        videos_response = youtube.videos().list(
            id=','.join(video_ids),
            part='contentDetails,snippet'
        ).execute()
        
        for idx, item in enumerate(videos_response['items']):
            video_id = item['id']
            
            # Find matching video in our data
            video = next((v for v in batch if v['video_id'] == video_id), None)
            if not video:
                continue
            
            # Parse duration
            duration = item['contentDetails']['duration']
            video['duration_sec'] = parse_duration(duration)
            
            # Check for captions
            video['has_captions'] = item['contentDetails'].get('caption') == 'true'
            video['caption_langs'] = 'en' if video['has_captions'] else None
            
            # Extract chapters (try to get from description or will be added during download)
            video['chapters'] = extract_chapters_from_description(video.get('description', ''))
            video['num_chapters'] = len(video['chapters']) if video['chapters'] else 0
            
            # Extract topic tag - NOW USES CHAPTERS!
            video['topic_tag'] = extract_topic_tag(
                video['title'], 
                video['description'],
                chapters=video['chapters']  # Pass chapters for better detection
            )
            
            # Placeholders for file paths
            video['local_audio_path'] = None
            video['local_transcript_path'] = None
            video['embedding_status'] = 'not-ready'
        
        print(f"Processed {min(i+50, len(video_data))}/{len(video_data)} videos...")
    
    return video_data


def extract_chapters_from_description(description):
    """
    Extract chapter timestamps and titles from video description.
    Many creators put chapters in the description like:
    0:00 Introduction
    2:30 Getting Started
    5:15 Advanced Topics
    
    Args:
        description (str): Video description
    
    Returns:
        list: List of chapter dictionaries or None
    """
    if not description:
        return None
    
    import re
    
    # Pattern for timestamps: 0:00, 00:00, 0:00:00
    timestamp_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s+(.+?)(?:\n|$)'
    matches = re.findall(timestamp_pattern, description, re.MULTILINE)
    
    if len(matches) < 2:  # Need at least 2 chapters to be valid
        return None
    
    chapters = []
    for match in matches:
        hours = 0
        if match[2]:  # Has hours
            hours = int(match[0])
            minutes = int(match[1])
            seconds = int(match[2])
        else:  # Only minutes:seconds
            minutes = int(match[0])
            seconds = int(match[1])
        
        start_time = hours * 3600 + minutes * 60 + seconds
        chapter_title = match[3].strip()
        
        # Clean up title (remove common prefixes)
        chapter_title = re.sub(r'^[\-\–\—\•\*\#\d\.\)\]]+\s*', '', chapter_title)
        
        chapters.append({
            'title': chapter_title,
            'start': start_time,
            'end': None  # Will be set to next chapter's start
        })
    
    # Set end times
    for i in range(len(chapters) - 1):
        chapters[i]['end'] = chapters[i + 1]['start']
    
    # Last chapter ends at video end (will be updated with actual duration later)
    if chapters:
        chapters[-1]['end'] = None
    
    return chapters if chapters else None


def is_lesson_video(title, description=''):
    """
    Determine if a video is an actual lesson or administrative/non-lesson content.
    Filters out: intros, portfolios, interviews, congratulations, Q&A sessions, etc.
    
    Args:
        title (str): Video title
        description (str): Video description
    
    Returns:
        bool: True if lesson video, False if non-lesson
    """
    text = (title + ' ' + description).lower()
    
    # Keywords that indicate non-lesson content
    non_lesson_keywords = [
        'free data analyst bootcamp',
        'how to become a data analyst',
        'portfolio project',
        'portfolio website',
        'web scraping using python | data analyst portfolio',
        'automating crypto',
        'api pull using python | data analyst project',
        'interview prep',
        'interview tips',
        'job interview',
        'interview questions on analyst builder',  # SQL interview practice
        'solving easy sql',
        'solving medium sql',
        'solving hard sql',
        'solving very hard sql',
        'congratulation',
        'congrats',
        'course completion',
        'certification (congrats',
        'download your',
        'resume',
        'cv tips',
        'linkedin to land a job',
        # Cloud setup videos
        'azure account setup',
        'azure account setup +',
        '$200 free credits',
        'blob storage and storage accounts in azure',
        'azure sql databases',
        'azure data factory',
        'azure synapse analytics',
        'aws setup and ui walkthrough',
        's3 storage in aws',
        'amazon athena in aws',
        'aws glue databrew',
        'aws quicksight |',
        # Project videos (too long/mixed for Q&A)
        'full project in excel',
        'full beginner project in tableau',
        'full power bi guided project',
        'data cleaning in mysql | full project',
        'mysql exploratory data analysis | full project',
        'building a bmi calculator',
        'building an automated file sorter',
        'scraping data from a real website',
    ]
    
    # Check if it's a non-lesson video
    for keyword in non_lesson_keywords:
        if keyword in text:
            return False
    
    # Additional check: very short titles that are just "Introduction" or "Welcome"
    title_words = title.lower().strip().split()
    if len(title_words) <= 2 and any(word in ['introduction', 'welcome', 'intro', 'outro', 'conclusion'] for word in title_words):
        # But allow "Introduction to Python", "Introduction to SQL" etc.
        if 'to' not in title_words:
            return False
    
    return True


def extract_topic_tag(title, description, chapters=None):
    """
    Extract topic tag from video title (main topic) and chapters (subtopics).
    Title determines main topic, chapters provide additional context.
    
    Args:
        title (str): Video title (primary source for main topic)
        description (str): Video description
        chapters (list): List of chapter dictionaries with 'title' keys
    
    Returns:
        str: Topic tag (e.g., 'Python', 'SQL', 'Python|Statistics')
    """
    title_lower = title.lower()
    
    # First, check if it's a non-lesson video
    if not is_lesson_video(title, description):
        return 'Non-Lesson'
    
    # ONLY use these 5 core topics for Data Analytics Bootcamp
    # Check title for the main topic keyword
    if 'sql' in title_lower or 'mysql' in title_lower:
        return 'SQL'
    elif 'excel' in title_lower:
        return 'Excel'
    elif 'tableau' in title_lower:
        return 'Tableau'
    elif 'power bi' in title_lower or 'powerbi' in title_lower:
        return 'Power BI'
    elif 'python' in title_lower or 'pandas' in title_lower or 'jupyter' in title_lower:
        return 'Python'
    
    # If none of the core topics found, mark as Non-Lesson
    return 'Non-Lesson'


def parse_duration(duration_str):
    """
    Parse ISO 8601 duration to seconds.
    
    Args:
        duration_str (str): ISO 8601 duration (e.g., 'PT15M33S')
    
    Returns:
        int: Duration in seconds
    """
    import re
    
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


if __name__ == "__main__":
    print("=" * 70)
    print("YOUTUBE VIDEO METADATA COLLECTOR")
    print("=" * 70)
    
    # Data Analytics Bootcamp Playlist URL
    PLAYLIST_URL = "https://youtube.com/playlist?list=PLUaB-1hjhk8FE_XZ87vPPSfHqb6OcM0cF"
    
    print(f"\nExtracting videos from Data Analytics Bootcamp playlist...")
    df = get_videos_from_playlist(PLAYLIST_URL)
    
    print(f"\n✓ Created CSV with {len(df)} videos")
    print(f"\nTopic distribution:")
    print(df['topic_tag'].value_counts())
    
    print(f"\nLesson vs Non-Lesson breakdown:")
    lesson_count = (df['topic_tag'] != 'Non-Lesson').sum()
    non_lesson_count = (df['topic_tag'] == 'Non-Lesson').sum()
    print(f"  Lesson videos: {lesson_count}")
    print(f"  Non-lesson videos: {non_lesson_count}")
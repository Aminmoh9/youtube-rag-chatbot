"""
Extract chapters from YouTube videos.
"""
from typing import List, Dict, Optional
import json
import re

def extract_chapters_from_description(description: str) -> Optional[List[Dict]]:
    """
    Extract chapter timestamps and titles from video description.
    
    Args:
        description (str): Video description
    
    Returns:
        list: List of chapter dictionaries or None
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
    
    # Last chapter ends at video end
    if chapters:
        chapters[-1]['end'] = None
    
    return chapters if chapters else None

def format_chapters_for_display(chapters: List[Dict]) -> str:
    """Format chapters for display in UI."""
    if not chapters:
        return "No chapters available"
    
    formatted = []
    for chapter in chapters:
        start_sec = chapter['start']
        minutes = start_sec // 60
        seconds = start_sec % 60
        time_str = f"{minutes}:{seconds:02d}"
        formatted.append(f"{time_str} - {chapter['title']}")
    
    return "\n".join(formatted)
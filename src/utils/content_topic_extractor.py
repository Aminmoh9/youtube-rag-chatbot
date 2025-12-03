"""
Extracts topics from user-provided content (videos, files, transcripts).
"""
from typing import Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
import hashlib
import json
import re

class ContentTopicExtractor:
    """Extracts topics from various content types."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    def extract_from_youtube_metadata(self, title: str, description: str = "", 
                                     channel: str = "") -> Dict:
        """Extract topic from YouTube video metadata."""
        prompt = f"""
        Analyze this YouTube video and extract its main topic:
        
        Title: {title}
        Channel: {channel}
        Description: {description[:500]}...
        
        Provide:
        1. Main topic (1-3 words)
        2. Subtopic/category
        3. 3-5 keywords
        4. Difficulty level (Beginner/Intermediate/Advanced)
        
        Format as JSON with keys: main_topic, subtopic, keywords, difficulty, content_type
        """
        
        try:
            response = self.llm.invoke(prompt)
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                topic_data = json.loads(json_match.group())
                topic_data['source'] = 'youtube_metadata'
                return topic_data
        except:
            pass
        
        # Fallback: Use title as topic
        return {
            'main_topic': title.split('|')[0].split('-')[0].strip()[:50],
            'subtopic': 'General',
            'keywords': title.split()[:5],
            'difficulty': 'Intermediate',
            'content_type': 'YouTube Video',
            'source': 'fallback'
        }
    
    def extract_from_transcript(self, transcript: str) -> Dict:
        """Extract topic from transcript content."""
        prompt = f"""
        Analyze this transcript content and extract its main topic:
        
        Transcript excerpt: {transcript[:1000]}...
        
        Provide:
        1. Main topic (what is this content about?)
        2. Key themes (3-5 main themes covered)
        3. Content type (Tutorial, Lecture, Interview, Presentation, etc.)
        4. Target audience (Beginner, Intermediate, Expert, General)
        
        Format as JSON.
        """
        
        try:
            response = self.llm.invoke(prompt)
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                topic_data = json.loads(json_match.group())
                topic_data['source'] = 'transcript_analysis'
                return topic_data
        except:
            pass
        
        return {
            'main_topic': 'User Uploaded Content',
            'key_themes': ['Custom Content'],
            'content_type': 'Uploaded Content',
            'target_audience': 'General',
            'source': 'transcript_fallback'
        }
    
    def extract_from_file_metadata(self, filename: str, file_type: str, 
                                 content_preview: str = "") -> Dict:
        """Extract topic from file metadata."""
        prompt = f"""
        Analyze this file and suggest a topic:
        
        Filename: {filename}
        File Type: {file_type}
        Content Preview: {content_preview[:500]}...
        
        Suggest an appropriate topic name for this content.
        Format as JSON with: suggested_topic, content_category, estimated_duration
        """
        
        try:
            response = self.llm.invoke(prompt)
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                topic_data = json.loads(json_match.group())
                topic_data['source'] = 'file_metadata'
                return topic_data
        except:
            pass
        
        # Extract from filename
        clean_name = filename.rsplit('.', 1)[0]  # Remove extension
        clean_name = clean_name.replace('_', ' ').replace('-', ' ')
        
        return {
            'suggested_topic': clean_name[:50],
            'content_category': 'User Upload',
            'estimated_duration': 'Unknown',
            'source': 'filename'
        }
    
    def generate_content_id(self, source: str, identifier: str) -> str:
        """Generate unique ID for user-provided content."""
        content_key = f"{source}:{identifier}"
        return f"user-content-{hashlib.md5(content_key.encode()).hexdigest()[:12]}"
    
    def create_content_session(self, content_data: Dict) -> Dict:
        """Create a session for user-provided content."""
        content_id = self.generate_content_id(
            content_data.get('source_type', 'unknown'),
            content_data.get('identifier', 'unknown')
        )
        
        # Extract topic
        if content_data.get('transcript'):
            topic_info = self.extract_from_transcript(content_data['transcript'])
        elif content_data.get('title'):
            topic_info = self.extract_from_youtube_metadata(
                title=content_data['title'],
                description=content_data.get('description', ''),
                channel=content_data.get('channel', '')
            )
        else:
            topic_info = self.extract_from_file_metadata(
                filename=content_data.get('filename', ''),
                file_type=content_data.get('file_type', ''),
                content_preview=content_data.get('preview', '')
            )
        
        # Create session
        session = {
            'content_id': content_id,
            'topic_info': topic_info,
            'content_type': content_data.get('content_type', 'unknown'),
            'source': content_data.get('source_type', 'unknown'),
            'created_at': content_data.get('created_at'),
            'content_length': content_data.get('content_length', 0),
            'status': 'processed'
        }
        
        return session
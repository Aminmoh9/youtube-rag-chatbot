"""
YouTube audio downloader using yt-dlp.
Downloads audio when subtitles are unavailable (with user consent).
"""
import os
from pathlib import Path
from typing import Optional


def download_audio(youtube_url: str, output_dir: str = "data/videos") -> Optional[str]:
    """
    Download audio from YouTube video.
    
    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save audio file
        
    Returns:
        Path to downloaded audio file, or None if failed
    """
    try:
        import yt_dlp
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure yt-dlp options - keep original format (webm/m4a), no conversion
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_id = info['id']
            file_ext = info.get('ext', 'webm')  # Usually webm or m4a
            
            # Construct output path with actual extension
            audio_path = os.path.join(output_dir, f"{video_id}.{file_ext}")
            
            if os.path.exists(audio_path):
                return audio_path
            else:
                return None
                
    except ImportError:
        print("Error: yt-dlp not installed. Install with: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None


def download_single_video_with_consent(youtube_url: str, consent_given: bool = False) -> Optional[str]:
    """
    Download audio with user consent check.
    
    Args:
        youtube_url: YouTube video URL
        consent_given: Whether user has given consent
        
    Returns:
        Path to downloaded audio file, or None if no consent or failed
    """
    if not consent_given:
        return None
    
    return download_audio(youtube_url)

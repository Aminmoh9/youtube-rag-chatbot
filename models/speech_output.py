"""
Speech Output Module
Converts text answers to speech and plays audio.
Uses OpenAI TTS for natural-sounding speech.
"""

import os
import platform
import tempfile
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def speak_answer(text: str, voice: str = 'alloy', auto_play: bool = False) -> Optional[str]:
    """
    Convert text to speech and optionally play the audio using OpenAI TTS.
    
    Args:
        text (str): Text to convert to speech
        voice (str): OpenAI voice ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
        auto_play (bool): If True, play audio immediately (not recommended for Streamlit)
    
    Returns:
        str: Path to generated audio file, or None if failed
    """
    try:
        # Generate speech with OpenAI
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_path = temp_audio.name
            temp_audio.write(response.content)
        
        # Only play audio if explicitly requested (not for Streamlit usage)
        if auto_play:
            play_audio(audio_path)
        
        return audio_path
            
    except Exception as e:
        print(f"✗ Error generating speech: {e}")
        return None


def play_audio(audio_path: str):
    """
    Play audio file using platform-specific command.
    
    Args:
        audio_path (str): Path to audio file
    """
    try:
        if not os.path.exists(audio_path):
            return
        
        system = platform.system()
        
        if system == "Windows":
            os.startfile(audio_path)
            
        elif system == "Darwin":  # macOS
            os.system(f'afplay "{audio_path}"')
            
        else:  # Linux
            players = ['mpg123', 'ffplay', 'cvlc', 'play']
            for player in players:
                if os.system(f'which {player} > /dev/null 2>&1') == 0:
                    os.system(f'{player} "{audio_path}" > /dev/null 2>&1')
                    break
                
    except Exception as e:
        print(f"⚠️ Could not play audio: {e}")


def text_to_audio_file(text: str, output_path: str, voice: str = 'alloy') -> bool:
    """
    Convert text to speech and save to specific file path using OpenAI TTS.
    
    Args:
        text (str): Text to convert
        output_path (str): Where to save the audio file
        voice (str): OpenAI voice name
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"✗ Error saving audio: {e}")
        return False
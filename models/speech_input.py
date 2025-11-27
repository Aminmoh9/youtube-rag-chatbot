"""
Speech Input Module
Captures user questions via microphone and converts speech to text.
Uses SpeechRecognition with OpenAI Whisper API.
"""

import speech_recognition as sr
from typing import Optional
import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def listen_to_question(timeout: int = 10, phrase_time_limit: int = 15) -> Optional[str]:
    """
    Capture and transcribe user's spoken question via microphone.
    
    Args:
        timeout (int): Maximum seconds to wait for speech to start
        phrase_time_limit (int): Maximum seconds for the phrase
    
    Returns:
        str: Transcribed question text, or error message if failed
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("🎤 Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("🎤 Listening... Speak your question now.")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("🔄 Processing speech...")
            
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio.get_wav_data())
            temp_audio_path = temp_audio.name
        
        try:
            # Convert speech to text using OpenAI Whisper API
            with open(temp_audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"  # Force English transcription
                )
            
            question = transcript.text
            print(f"✓ You asked: {question}")
            return question
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        
    except sr.WaitTimeoutError:
        error_msg = "ERROR: No speech detected. Please try again."
        print(error_msg)
        return None
        
    except sr.UnknownValueError:
        error_msg = "ERROR: Could not understand the audio. Please speak clearly."
        print(error_msg)
        return None
        
    except sr.RequestError as e:
        error_msg = f"ERROR: Could not request results from speech service: {e}"
        print(error_msg)
        return None
        
    except Exception as e:
        error_msg = f"ERROR: Unexpected error: {e}"
        print(error_msg)
        return None


def test_microphone() -> bool:
    """
    Test if microphone is available and working.
    
    Returns:
        bool: True if microphone is accessible, False otherwise
    """
    try:
        with sr.Microphone() as source:
            print("✓ Microphone detected and accessible.")
            return True
    except Exception as e:
        print(f"✗ Microphone error: {e}")
        return False
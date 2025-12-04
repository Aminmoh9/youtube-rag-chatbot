"""
Voice utilities for speech-to-text and text-to-speech.
"""
import os
import tempfile
import streamlit as st
from openai import OpenAI

# Check TTS availability
TTS_AVAILABLE = False
try:
    if os.getenv('OPENAI_API_KEY'):
        TTS_AVAILABLE = True
except:
    pass


def listen_to_question(timeout=5, phrase_time_limit=20, use_whisper_fallback=True):
    """
    Listen to microphone input and return transcribed text.
    Uses SpeechRecognition library and prefers a local Whisper transcription agent.
    Falls back to the OpenAI Whisper API if local Whisper is not available.
    
    Args:
        timeout: Time to wait for speech to start (seconds)
        phrase_time_limit: Maximum time for a phrase (seconds)
        use_whisper_fallback: If True, use Whisper API when Google fails
    
    Returns:
        Transcribed text or None if failed
    """
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        # Sensitivity settings
        recognizer.energy_threshold = 1000
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        with sr.Microphone(sample_rate=16000) as source:
            with st.spinner("🎙️ Adjusting for background noise..."):
                recognizer.adjust_for_ambient_noise(source, duration=1.5)
            with st.spinner("✅ Listening... Speak clearly into your microphone!"):
                audio = recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

        # Prefer local Whisper (re-uses WhisperTranscriptionAgent if available)
        wav_data = audio.get_wav_data()

        try:
            from src.transcription.whisper_agent import WhisperTranscriptionAgent

            agent = WhisperTranscriptionAgent(model="small", use_local=True)
            result = agent.transcribe_bytes(
                audio_bytes=wav_data,
                filename="microphone.wav",
                language="en",
            )

            if result.get("success") and result.get("text"):
                return result.get("text")
            # If local whisper didn't succeed, fall through to fallbacks
        except Exception:
            # Could not load or run local Whisper — we'll try other methods below
            pass

        # If local Whisper didn't produce results, try OpenAI Whisper API (if available)
        if use_whisper_fallback and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                import tempfile

                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(wav_data)
                    tmp_file_path = tmp_file.name

                try:
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="en",
                        )
                    return getattr(transcript, "text", None) or transcript.text
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except Exception:
                        pass
            except Exception:
                return None
        else:
            return None

    except sr.WaitTimeoutError:
        st.warning("No speech detected. Please speak after clicking the button.")
        return None
    except ImportError:
        st.error("SpeechRecognition not installed.")
        return None
    except OSError as e:
        st.error("Microphone access denied. Check Windows permissions.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def transcribe_audio(audio_bytes):
    """
    Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_bytes: Audio data as bytes
    
    Returns:
        Transcribed text or None if failed
    """
    if not TTS_AVAILABLE or not audio_bytes:
        return None
    
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe
            with open(tmp_file_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return transcript.text
        finally:
            # Clean up
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None
    
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None


def text_to_speech(text, voice="nova"):
    """
    Convert text to speech using OpenAI TTS API.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, nova, onyx, shimmer)
    
    Returns:
        Audio bytes or None if failed
    """
    if not TTS_AVAILABLE or not text:
        return None
    
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text[:4096]  # Limit text length
        )
        
        return response.content
    
    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return None


def render_voice_settings():
    """Render voice settings in sidebar."""
    if not TTS_AVAILABLE:
        st.sidebar.warning("⚠️ Voice features require OpenAI API key")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎤 Voice Settings")
    
    # Voice selection
    voices = ['alloy', 'echo', 'fable', 'nova', 'onyx', 'shimmer']
    st.session_state.selected_voice = st.sidebar.selectbox(
        "TTS Voice",
        voices,
        index=voices.index(st.session_state.get('selected_voice', 'nova'))
    )
    
    # Keep a deterministic `auto_play` flag set to False so playback never auto-triggers
    st.session_state['auto_play'] = False
    
    # Optional: generate audio for summaries (disabled by default to control cost)
    # Stored in session state so other parts of the app can read it.
    st.session_state['generate_summary_audio'] = st.sidebar.checkbox(
        "Generate audio for summaries (may incur cost)",
        value=st.session_state.get('generate_summary_audio', False),
        help="When enabled, the app will generate downloadable audio files for summaries. Disabled by default to avoid extra API usage."
    )
    

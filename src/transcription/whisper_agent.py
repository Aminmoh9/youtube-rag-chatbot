"""
Whisper Transcription Agent for audio/video files.
Supports both local Whisper and OpenAI Whisper API.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class WhisperTranscriptionAgent:
    """Agent for transcribing audio/video files using local Whisper or OpenAI API."""
    
    def __init__(self, model: str = "small", use_local: bool = True):
        """
        Initialize Whisper transcription agent.
        
        Args:
            model: Whisper model to use
                   - Local: "tiny", "base", "small", "medium", "large" (default: "small")
                   - API: "whisper-1" (only option)
            use_local: If True, use local Whisper. If False, use OpenAI API (default: True)
        """
        self.model_name = model
        self.use_local = use_local
        self._client = None
        self._local_model = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client
    
    @property
    def local_model(self):
        """Lazy load local Whisper model."""
        if self._local_model is None:
            try:
                import whisper
                print(f"Loading Whisper model '{self.model_name}'... (first time may take a few minutes to download)")
                self._local_model = whisper.load_model(self.model_name)
                print(f"✓ Whisper model '{self.model_name}' loaded")
            except ImportError:
                raise ImportError("whisper package not installed. Install with: pip install openai-whisper")
        return self._local_model
    
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Transcribe an audio/video file.
        
        Args:
            file_path: Path to audio/video file
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional prompt to guide transcription
            temperature: Sampling temperature (0-1)
        
        Returns:
            Dictionary with transcription results
        """
        if self.use_local:
            return self._transcribe_local(file_path, language, temperature)
        else:
            return self._transcribe_api(file_path, language, prompt, temperature)
    
    def _transcribe_local(
        self,
        file_path: str,
        language: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': f"File not found: {file_path}"
                }
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                return {
                    'success': False,
                    'error': f"File is empty: {file_path}"
                }
            
            print(f"Transcribing file: {file_path} ({file_size} bytes)")
            
            # Transcribe with local Whisper
            result = self.local_model.transcribe(
                str(file_path),
                language=language,
                temperature=temperature,
                verbose=False,
                fp16=False  # Use FP32 on CPU
            )
            
            return {
                'success': True,
                'text': result['text'].strip(),
                'language': result.get('language', language or 'unknown'),
                'segments': result.get('segments', [])
            }
            
        except FileNotFoundError as e:
            return {
                'success': False,
                'error': f"File not found during transcription: {str(e)}"
            }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f"Local transcription failed: {str(e)}\n{traceback.format_exc()}"
            }
    
    def _transcribe_api(
        self,
        file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': f"File not found: {file_path}"
                }
            
            # Check file size (max 25MB for Whisper API)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 25:
                return {
                    'success': False,
                    'error': f"File too large: {file_size_mb:.1f}MB (max 25MB). Consider splitting the file."
                }
            
            # Transcribe with Whisper API
            with open(file_path, 'rb') as audio_file:
                params = {
                    'model': 'whisper-1',
                    'file': audio_file,
                    'response_format': 'verbose_json',
                    'temperature': temperature
                }
                
                if language:
                    params['language'] = language
                if prompt:
                    params['prompt'] = prompt
                
                transcript = self.client.audio.transcriptions.create(**params)
            
            return {
                'success': True,
                'text': transcript.text,
                'language': getattr(transcript, 'language', language or 'unknown'),
                'duration': getattr(transcript, 'duration', None),
                'segments': getattr(transcript, 'segments', [])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Transcription failed: {str(e)}"
            }
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Raw audio bytes
            filename: Filename with extension (for format detection)
            language: Optional language code
            prompt: Optional prompt
            temperature: Sampling temperature
        
        Returns:
            Dictionary with transcription results
        """
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix,
                delete=False
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Transcribe
            result = self.transcribe_file(
                file_path=temp_path,
                language=language,
                prompt=prompt,
                temperature=temperature
            )
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Transcription from bytes failed: {str(e)}"
            }
    
    def transcribe_with_chunks(
        self,
        file_path: str,
        chunk_duration: int = 600,  # 10 minutes
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe large files by splitting into chunks.
        
        Args:
            file_path: Path to audio/video file
            chunk_duration: Duration of each chunk in seconds
            language: Optional language code
        
        Returns:
            Dictionary with combined transcription results
        """
        try:
            from pydub import AudioSegment
            
            file_path = Path(file_path)
            
            # Load audio
            if file_path.suffix.lower() in ['.mp3']:
                audio = AudioSegment.from_mp3(file_path)
            elif file_path.suffix.lower() in ['.wav']:
                audio = AudioSegment.from_wav(file_path)
            elif file_path.suffix.lower() in ['.m4a']:
                audio = AudioSegment.from_file(file_path, format='m4a')
            else:
                audio = AudioSegment.from_file(file_path)
            
            # Split into chunks
            chunk_length_ms = chunk_duration * 1000
            chunks = [audio[i:i + chunk_length_ms] 
                     for i in range(0, len(audio), chunk_length_ms)]
            
            # Transcribe each chunk
            all_text = []
            all_segments = []
            
            for i, chunk in enumerate(chunks):
                # Export chunk to temp file
                with tempfile.NamedTemporaryFile(
                    suffix='.wav',
                    delete=False
                ) as temp_file:
                    chunk.export(temp_file.name, format='wav')
                    temp_path = temp_file.name
                
                # Transcribe chunk
                result = self.transcribe_file(
                    file_path=temp_path,
                    language=language
                )
                
                # Clean up
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                if result['success']:
                    all_text.append(result['text'])
                    if result.get('segments'):
                        # Adjust segment timestamps
                        offset = i * chunk_duration
                        for seg in result['segments']:
                            seg['start'] += offset
                            seg['end'] += offset
                        all_segments.extend(result['segments'])
                else:
                    return result
            
            return {
                'success': True,
                'text': ' '.join(all_text),
                'language': language or 'unknown',
                'duration': len(audio) / 1000.0,
                'segments': all_segments,
                'chunks_processed': len(chunks)
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "pydub not installed. Install with: pip install pydub"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Chunked transcription failed: {str(e)}"
            }
    
    def batch_transcribe(
        self,
        file_paths: Optional[list] = None,
        audio_dir: Optional[str] = None,
        require_consent: bool = True,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Batch transcribe multiple audio files.
        
        Args:
            file_paths: List of file paths to transcribe
            audio_dir: Directory containing audio files (alternative to file_paths)
            require_consent: Whether user consent is required
            language: Optional language code
        
        Returns:
            Dictionary with batch transcription results
        """
        try:
            if require_consent:
                # In a real implementation, you'd check for user consent
                pass
            
            # Get list of files
            if file_paths:
                files = [Path(fp) for fp in file_paths]
            elif audio_dir:
                audio_dir = Path(audio_dir)
                if not audio_dir.exists():
                    return {
                        'success': False,
                        'error': f"Directory not found: {audio_dir}"
                    }
                # Common audio/video extensions
                extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.flac', '.ogg']
                files = [f for f in audio_dir.iterdir() 
                        if f.suffix.lower() in extensions]
            else:
                return {
                    'success': False,
                    'error': "Either file_paths or audio_dir must be provided"
                }
            
            if not files:
                return {
                    'success': False,
                    'error': "No audio files found"
                }
            
            # Transcribe each file
            results = []
            successful = 0
            failed = 0
            
            for file_path in files:
                result = self.transcribe_file(str(file_path), language=language)
                
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                
                results.append({
                    'file': str(file_path),
                    'filename': file_path.name,
                    'success': result['success'],
                    'text': result.get('text', ''),
                    'error': result.get('error', None),
                    'duration': result.get('duration', None)
                })
            
            return {
                'success': True,
                'total_files': len(files),
                'successful': successful,
                'failed': failed,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Batch transcription failed: {str(e)}"
            }


# Convenience function
def transcribe_audio(
    file_path: str,
    language: Optional[str] = None,
    auto_chunk: bool = True
) -> Dict[str, Any]:
    """
    Transcribe audio file (auto-chunks if needed).
    
    Args:
        file_path: Path to audio/video file
        language: Optional language code
        auto_chunk: Automatically chunk large files
    
    Returns:
        Dictionary with transcription results
    """
    agent = WhisperTranscriptionAgent(model="small", use_local=True)
    
    # Check file size
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    
    if auto_chunk and file_size_mb > 20:
        return agent.transcribe_with_chunks(file_path, language=language)
    else:
        return agent.transcribe_file(file_path, language=language)

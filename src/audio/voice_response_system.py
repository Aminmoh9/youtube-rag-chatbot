"""
Unified voice response system for all 4 input methods.
Generates and manages voice responses for answers and summaries.
"""
import os
import tempfile
from typing import Optional, Dict
from datetime import datetime
import streamlit as st

# Import from your existing speech_output.py
from models.speech_output import speak_answer, text_to_audio_file


class VoiceResponseSystem:
    """Manages voice generation and playback for all content types."""
    
    def __init__(self, cache_dir: str = "data/voice_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Voice options (OpenAI TTS voices)
        self.voice_options = {
            'nova': 'Clear, expressive female voice',
            'alloy': 'Balanced, versatile voice',
            'echo': 'Warm, comforting male voice',
            'fable': 'Storytelling voice',
            'onyx': 'Deep, authoritative voice',
            'shimmer': 'Energetic, upbeat voice'
        }
        
        # Default voice per content type (simplified)
        # Use a single default voice for all input methods. The UI
        # selector or an explicit `voice` argument will override this.
        self.default_voices = {
            'topic_search': 'nova',
            'youtube_link': 'nova',
            'audio_video_upload': 'nova',
            'script_upload': 'nova'
        }
    
    def generate_voice_response(self, text: str, content_type: str, 
                              session_id: str = None, voice: str = None) -> Dict:
        """
        Generate voice response for text.
        
        Args:
            text: Text to convert to speech
            content_type: One of the 4 input methods
            session_id: Optional session ID for caching
            voice: Specific voice to use (optional)
        
        Returns:
            Dict with audio file info and metadata
        """
        # Choose voice
        if not voice:
            voice = self.default_voices.get(content_type, 'nova')
        
        # Create cache key
        if session_id:
            cache_key = f"{session_id}_{hash(text) & 0xFFFFFFFF}"
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")
            
            # Check cache
            if os.path.exists(cache_file):
                return {
                    'success': True,
                    'audio_path': cache_file,
                    'cached': True,
                    'voice': voice,
                    'content_type': content_type,
                    'text_length': len(text)
                }
        
        # Generate new audio
        try:
            # Use temp file first
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                temp_path = tmp.name
            
            success = text_to_audio_file(text, temp_path, voice=voice)
            
            if success:
                # Move to cache if session ID provided
                if session_id:
                    import shutil
                    shutil.move(temp_path, cache_file)
                    audio_path = cache_file
                else:
                    audio_path = temp_path
                
                return {
                    'success': True,
                    'audio_path': audio_path,
                    'cached': False,
                    'voice': voice,
                    'content_type': content_type,
                    'text_length': len(text),
                    'generated_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate audio',
                    'content_type': content_type
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'content_type': content_type
            }
    
    def get_answer_audio(self, answer_result: Dict, session_info: Dict) -> Optional[str]:
        """
        Get audio for an answer result.
        Includes intelligent formatting for better speech.
        """
        # Format answer for better speech
        formatted_answer = self._format_for_speech(
            answer_result.get('answer', ''),
            session_info.get('content_type', 'unknown')
        )
        
        # Generate audio
        voice_response = self.generate_voice_response(
            text=formatted_answer,
            content_type=session_info.get('input_method', 'unknown'),
            session_id=session_info.get('id'),
            voice=self._select_voice_for_content(session_info)
        )
        
        if voice_response['success']:
            return voice_response['audio_path']
        return None
    
    def get_summary_audio(self, summary_data: Dict, session_info: Dict) -> Optional[str]:
        """
        Get audio for a summary.
        """
        # Extract summary text
        if isinstance(summary_data, dict):
            if 'short_summary' in summary_data:
                summary_text = summary_data['short_summary']
            elif 'overall_summary' in summary_data:
                summary_text = summary_data['overall_summary']
            else:
                # Try to get first text value
                for value in summary_data.values():
                    if isinstance(value, str) and len(value) > 50:
                        summary_text = value
                        break
                else:
                    summary_text = str(summary_data)
        else:
            summary_text = str(summary_data)
        
        # Add intro based on content type
        intro = self._get_summary_intro(session_info)
        full_text = f"{intro} {summary_text}"
        
        # Generate audio (return full response dict so callers can inspect cached flag)
        voice_response = self.generate_voice_response(
            text=full_text,
            content_type=session_info.get('input_method', 'unknown'),
            session_id=f"{session_info.get('id')}_summary",
            voice=self._select_voice_for_content(session_info, is_summary=True)
        )

        return voice_response
    
    def _format_for_speech(self, text: str, content_type: str) -> str:
        """Format text for better speech synthesis."""
        import re
        
        # Clean up markdown and special characters
        text = re.sub(r'#+\s*', '', text)  # Remove headings
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Remove code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links
        
        # Add pauses for better rhythm
        text = text.replace('. ', '. \n\n')
        text = text.replace('? ', '? \n\n')
        text = text.replace('! ', '! \n\n')
        
        # Add content-specific formatting
        if content_type == 'topic_search':
            text = f"Here's what I found about this topic. {text}"
        elif content_type == 'youtube_link':
            text = f"Based on this video, {text}"
        elif content_type in ['audio_video_upload', 'script_upload']:
            text = f"Based on your uploaded content, {text}"
        
        return text
    
    def _select_voice_for_content(self, session_info: Dict, is_summary: bool = False) -> str:
        """Select appropriate voice for content type."""
        content_type = session_info.get('input_method', 'unknown')
        
        if is_summary:
            # Use different voice for summaries vs answers
            if content_type == 'topic_search':
                return 'shimmer'  # Energetic for learning summaries
            elif content_type == 'youtube_link':
                return 'echo'     # Warm for video summaries
            else:
                return 'alloy'    # Balanced for uploaded content
        else:
            return self.default_voices.get(content_type, 'nova')
    
    def _get_summary_intro(self, session_info: Dict) -> str:
        """Get appropriate intro for summary audio."""
        content_type = session_info.get('input_method', 'unknown')
        topic = session_info.get('topic', 'this content')
        
        intros = {
            'topic_search': f"Here's a summary of what I found about {topic}.",
            'youtube_link': f"Here are the key points from the video about {topic}.",
            'audio_video_upload': f"Here's a summary of your recording about {topic}.",
            'script_upload': f"Here's a summary of your document about {topic}."
        }
        
        return intros.get(content_type, f"Here's a summary of {topic}.")
    
    def create_voice_settings_ui(self):
        """Create UI for voice settings."""
        st.markdown("#### 🔊 Voice Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_voice = st.selectbox(
                "Choose Voice",
                options=list(self.voice_options.keys()),
                format_func=lambda x: f"{x} - {self.voice_options[x]}",
                index=list(self.voice_options.keys()).index('nova')
            )
        
        with col2:
            playback_speed = st.slider(
                "Playback Speed",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Note: Speed adjustment requires audio processing"
            )
        
        # Preview button
        if st.button("🎵 Preview Voice", use_container_width=True):
            preview_text = "This is what the selected voice sounds like."
            preview_audio = self.generate_voice_response(
                text=preview_text,
                content_type='preview',
                voice=selected_voice
            )
            
            if preview_audio['success']:
                st.audio(preview_audio['audio_path'], format='audio/mp3')
        
        return selected_voice


# Global instance
voice_system = VoiceResponseSystem()
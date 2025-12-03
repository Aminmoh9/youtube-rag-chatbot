"""
Audio/Video upload processor - handles local media file transcription.
"""
import os
import tempfile
import hashlib
from typing import Dict


class AudioVideoProcessor:
    """Process uploaded audio/video files."""
    
    def __init__(self, chunker, embedder, session_manager, summarizer,
                 isolation_manager, topic_extractor):
        """Initialize with shared components."""
        self.chunker = chunker
        self.embedder = embedder
        self.session_manager = session_manager
        self.summarizer = summarizer
        self.isolation_manager = isolation_manager
        self.topic_extractor = topic_extractor
    
    def process(self, file_bytes: bytes, filename: str, file_type: str,
                consent_given: bool = True) -> Dict:
        """
        Process audio/video file upload.
        
        Args:
            file_bytes: File content
            filename: Original filename
            file_type: File type (audio/video)
            consent_given: User consent for processing
        
        Returns:
            Processing result dictionary
        """
        if not consent_given:
            return {'success': False, 'error': 'Consent required for audio/video processing'}
        
        session_id = self.session_manager.generate_session_id(
            f"upload:{hashlib.md5(file_bytes).hexdigest()[:12]}"
        )
        
        # Save to temp file (keep it open until transcription is done)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        tmp_path = tmp_file.name
        tmp_file.write(file_bytes)
        tmp_file.flush()  # Ensure all data is written
        tmp_file.close()  # Close the file so Whisper can read it
        
        # Verify file was created
        if not os.path.exists(tmp_path):
            return {'success': False, 'error': f'Failed to create temp file: {tmp_path}'}
        
        print(f"Created temp file: {tmp_path} ({len(file_bytes)} bytes)")
        
        try:
            # Transcribe with local Whisper (faster and free)
            from src.transcription.whisper_agent import WhisperTranscriptionAgent
            whisper_agent = WhisperTranscriptionAgent(model="small", use_local=True)
            result = whisper_agent.transcribe_file(tmp_path)
            
            if not result.get('success'):
                return {'success': False, 'error': f'Transcription failed: {result.get("error", "Unknown error")}'}
            
            transcript = result.get('text', '')
            
            if not transcript:
                return {'success': False, 'error': f'Transcription failed: {transcript}'}
            
            # Use filename as topic (cleaner than AI extraction)
            # Remove file extension and clean up
            clean_filename = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').strip()
            main_topic = clean_filename if clean_filename else f"Uploaded {file_type}: {filename}"
            
            # Generate summary
            summary_result = self.summarizer.summarize(transcript, "standard")
            summary = summary_result.get('summary', '') if isinstance(summary_result, dict) else str(summary_result)
            
            # Chunk content
            chunks = self.chunker.chunk(transcript, session_id, {
                'filename': filename,
                'file_type': file_type,
                'input_method': 'audio_video_upload'
            })
            
            # Embed and store
            embeddings = self.embedder.generate_embeddings(chunks, {
                'topic': main_topic,
                'input_method': 'audio_video_upload',
                'filename': filename
            })
            
            pinecone_result = self.isolation_manager.upsert_with_isolation(
                vectors=embeddings,
                topic=main_topic
            )
            
            # Create session
            session = {
                'session_id': session_id,
                'input_method': 'audio_video_upload',
                'filename': filename,
                'file_type': file_type,
                'topic': main_topic,
                'summary': summary,
                'chunk_count': len(chunks),
                'pinecone_result': pinecone_result,
                'content_type': file_type,
                'status': 'processed',
                'transcript_preview': transcript[:500]
            }
            
            self.session_manager.save_session(session_id, session)
            
            # Get namespace for immediate Q&A
            namespace = self.isolation_manager.get_topic_namespace(main_topic)
            
            return {
                'success': True,
                'session_id': session_id,
                'input_method': 'audio_video_upload',
                'topic': main_topic,
                'summary': summary,
                'can_query': True,
                'content_type': f'Uploaded {file_type}',
                'namespace': namespace,  # Store for immediate Q&A
                'loaded_from_pinecone': False
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

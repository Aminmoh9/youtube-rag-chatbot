"""
Script upload processor - handles text/transcript file uploads.
"""
import hashlib
from typing import Dict


class ScriptUploadProcessor:
    """Process uploaded script/transcript text files."""
    
    def __init__(self, chunker, embedder, session_manager, summarizer,
                 isolation_manager, topic_extractor):
        """Initialize with shared components."""
        self.chunker = chunker
        self.embedder = embedder
        self.session_manager = session_manager
        self.summarizer = summarizer
        self.isolation_manager = isolation_manager
        self.topic_extractor = topic_extractor
    
    def process(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Process script/transcript file upload.
        
        Args:
            file_bytes: File content
            filename: Original filename
        
        Returns:
            Processing result dictionary
        """
        session_id = self.session_manager.generate_session_id(
            f"script:{hashlib.md5(file_bytes).hexdigest()[:12]}"
        )
        
        # Read text
        try:
            transcript = file_bytes.decode('utf-8', errors='ignore')
        except:
            return {'success': False, 'error': 'Cannot decode file as text'}
        
        if len(transcript) < 50:
            return {'success': False, 'error': 'File too short or empty'}
        
        # Use filename as topic (cleaner than AI extraction which can be inconsistent)
        # Remove file extension and clean up
        clean_filename = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').strip()
        main_topic = clean_filename if clean_filename else f"Uploaded Script: {filename}"
        
        # Generate summary
        summary_result = self.summarizer.summarize(transcript, "standard")
        summary = summary_result.get('summary', '') if isinstance(summary_result, dict) else str(summary_result)
        
        # Chunk content
        chunks = self.chunker.chunk(transcript, session_id, {
            'filename': filename,
            'input_method': 'script_upload'
        })
        
        # Embed and store
        embeddings = self.embedder.generate_embeddings(chunks, {
            'topic': main_topic,
            'input_method': 'script_upload',
            'filename': filename
        })
        
        pinecone_result = self.isolation_manager.upsert_with_isolation(
            vectors=embeddings,
            topic=main_topic
        )
        
        # Get namespace for immediate Q&A
        namespace = self.isolation_manager.get_topic_namespace(main_topic)
        
        # Create session
        session = {
            'session_id': session_id,
            'input_method': 'script_upload',
            'filename': filename,
            'topic': main_topic,
            'summary': summary,
            'chunk_count': len(chunks),
            'pinecone_result': pinecone_result,
            'content_type': 'text',
            'status': 'processed',
            'transcript_preview': transcript[:500]
        }
        
        self.session_manager.save_session(session_id, session)
        
        return {
            'success': True,
            'session_id': session_id,
            'input_method': 'script_upload',
            'topic': main_topic,
            'summary': summary,
            'can_query': True,
            'content_type': 'Uploaded Script',
            'namespace': namespace,
            'loaded_from_pinecone': False
        }

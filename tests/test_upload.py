"""Test file upload processing."""
import pytest
import os
from dotenv import load_dotenv

load_dotenv()

class TestUploadProcessing:
    """Test upload processing functionality."""
    
    def test_text_file_processing(self):
        """Test processing of uploaded text files."""
        from src.processors.chunking.character_chunker import CharacterChunker
        
        chunker = CharacterChunker()
        test_content = "This is test content for upload processing. " * 20
        
        chunks = chunker.chunk(
            content=test_content,
            source_id="upload_test",
            metadata={'source': 'file://test.txt', 'title': 'Test Upload'}
        )
        
        assert len(chunks) > 0
        assert chunks[0]['text']
        assert chunks[0]['metadata']['source'] == "file://test.txt"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_summarization(self):
        """Test summarization of uploaded content."""
        from src.qa.summarization_agent import SummarizationAgent
        
        summarizer = SummarizationAgent()
        test_text = "Artificial intelligence is transforming industries. " * 10
        
        result = summarizer.summarize(test_text, "standard")
        
        assert result is not None
        assert isinstance(result, (str, dict))

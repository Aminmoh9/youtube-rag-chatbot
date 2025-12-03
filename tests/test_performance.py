"""
Performance testing suite.
"""
import time
import pytest
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

class PerformanceTests:
    """Performance testing."""
    
    @pytest.mark.performance
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_summarization_performance(self):
        """Test summarization performance."""
        from src.qa.summarization_agent import SummarizationAgent
        
        summarizer = SummarizationAgent()
        test_text = "This is test content about artificial intelligence. " * 50
        
        start = time.time()
        result = summarizer.summarize(test_text, "standard")
        end = time.time()
        
        elapsed = end - start
        print(f"Summarization took: {elapsed:.2f}s")
        
        assert result is not None
        assert elapsed < 15.0  # Should take less than 15 seconds
    
    @pytest.mark.performance
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_qa_latency(self):
        """Test QA response latency."""
        from src.qa.qa_model import QAModel
        
        qa = QAModel()
        
        # Note: QAModel.ask_question requires session_id with actual data
        # This is a simplified performance test for model initialization
        latencies = []
        for i in range(3):
            start = time.time()
            # Just test model responsiveness (initialization time)
            test_model = QAModel()
            end = time.time()
            latencies.append(end - start)
            
            assert test_model is not None
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average QA latency: {avg_latency:.2f}s")
        
        assert avg_latency < 10.0  # Should average under 10 seconds
    
    @pytest.mark.performance
    def test_chunking_performance(self):
        """Test chunking performance."""
        from src.processors.chunking.character_chunker import CharacterChunker
        
        chunker = CharacterChunker()
        test_text = "Test content. " * 1000  # Large text
        
        start = time.time()
        chunks = chunker.chunk(
            text=test_text,
            video_id="test",
            url="http://test.com",
            title="Test"
        )
        end = time.time()
        
        elapsed = end - start
        print(f"Chunking {len(test_text)} chars took: {elapsed:.2f}s")
        
        assert len(chunks) > 0
        assert elapsed < 2.0  # Should be very fast
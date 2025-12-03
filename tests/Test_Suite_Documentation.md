# Test Suite Documentation

## Overview
This directory contains automated tests for the YouTube AI Research Assistant. Tests verify core functionality, performance, and integration points.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_comprehensive.py

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run only performance tests
pytest tests/ -m performance
```

## Test Results
Results are automatically saved to `test_results/`:
- **junit.xml** - Machine-readable test results (for CI/CD)
- **report.html** - Interactive HTML report (open in browser)
- **test.log** - Detailed execution logs

## Test Files

### test_comprehensive.py
Core functionality tests covering the main features of the application.

**TestYouTubeAI Class:**

1. **test_youtube_metadata_fetch**
   - **What it tests:** YouTube API integration and video metadata retrieval
   - **How it works:** Creates a YouTubeMetadataAgent and fetches video info for "python programming" topic
   - **Why it matters:** Confirms YouTube API key is valid and API calls work
   - **Requires:** YOUTUBE_API_KEY environment variable

2. **test_subtitle_extraction**
   - **What it tests:** YouTube subtitle/transcript downloading
   - **How it works:** Uses SubtitleExtractor to get subtitles from a known public video
   - **Why it matters:** Ensures transcript extraction works (critical for processing videos)
   - **Requires:** Working internet connection

3. **test_qa_pipeline**
   - **What it tests:** Question-answering model initialization
   - **How it works:** Creates QAModel instance and verifies OpenAI LLM is configured
   - **Why it matters:** Confirms OpenAI API key is valid and model can be initialized
   - **Requires:** OPENAI_API_KEY environment variable

4. **test_chunking_strategies**
   - **What it tests:** Text chunking for embeddings
   - **How it works:** Uses CharacterChunker to split long text into overlapping chunks
   - **Why it matters:** Ensures content can be properly split for vector storage
   - **No API keys required**

**TestLangSmithIntegration Class:**

5. **test_tracing_enabled**
   - **What it tests:** LangSmith monitoring configuration
   - **How it works:** Checks if LangSmithManager detects LANGSMITH_API_KEY and enables tracing
   - **Why it matters:** Verifies observability/debugging tools are properly configured
   - **Optional:** Works without LangSmith API key (tracing just disabled)

6. **test_feedback_ui**
   - **What it tests:** Feedback widget component
   - **How it works:** Instantiates LangSmithFeedbackUI and checks for required methods
   - **Why it matters:** Ensures user feedback collection UI component exists
   - **No API keys required**

### test_upload.py
Tests for file upload and processing functionality.

**TestUploadProcessing Class:**

7. **test_text_file_processing**
   - **What it tests:** File upload chunking
   - **How it works:** Creates fake uploaded text content and chunks it using CharacterChunker
   - **Why it matters:** Ensures uploaded files can be processed correctly
   - **No API keys required**

8. **test_summarization**
   - **What it tests:** AI-powered content summarization
   - **How it works:** Sends test text to SummarizationAgent and verifies it returns a summary
   - **Why it matters:** Confirms OpenAI can generate summaries of content
   - **Requires:** OPENAI_API_KEY environment variable

### test_performance.py
Performance and load testing for critical operations.

**PerformanceTests Class:**

9. **test_summarization_performance**
   - **What it tests:** Summarization speed
   - **How it works:** Times how long it takes to summarize 50 repeated sentences
   - **Why it matters:** Ensures summarization completes in reasonable time (<15 seconds)
   - **Requires:** OPENAI_API_KEY
   - **Run with:** `pytest tests/ -m performance`

10. **test_qa_latency**
    - **What it tests:** Model initialization time
    - **How it works:** Times how long QAModel takes to initialize (3 iterations)
    - **Why it matters:** Ensures app startup isn't too slow (<10 seconds average)
    - **Requires:** OPENAI_API_KEY
    - **Run with:** `pytest tests/ -m performance`

11. **test_chunking_performance**
    - **What it tests:** Chunking speed for large content
    - **How it works:** Times chunking of 1000 repeated sentences
    - **Why it matters:** Ensures large files can be processed quickly (<2 seconds)
    - **No API keys required**
    - **Run with:** `pytest tests/ -m performance`

## Test Markers

Tests are marked for selective execution:

- `@pytest.mark.integration` - Full pipeline tests (may be slow)
- `@pytest.mark.performance` - Performance/timing tests
- `@pytest.mark.load` - Load/stress tests (not yet implemented)

Run specific markers:
```bash
pytest tests/ -m performance  # Only performance tests
pytest tests/ -m "not performance"  # Skip performance tests
```

## Environment Variables Required

For full test coverage, set these in your `.env` file:

```env
OPENAI_API_KEY=your_key          # Required for QA and summarization tests
YOUTUBE_API_KEY=your_key         # Required for YouTube metadata tests
LANGSMITH_API_KEY=your_key       # Optional for tracing tests
PINECONE_API_KEY=your_key        # Not directly tested (used in integration)
```

## Understanding Test Results

### Passing Tests ✅
All core functionality works correctly. Your app should run without errors.

### Failing Tests ❌
Check the HTML report (`test_results/report.html`) for:
- **TypeError/AttributeError** - Code signature changed, update tests
- **AssertionError** - Expected behavior not matching actual behavior
- **ConnectionError** - API keys invalid or network issues
- **TimeoutError** - Performance degraded, operations too slow

## Adding New Tests

When adding features, add corresponding tests:

1. **Unit tests** in `test_comprehensive.py` - Test individual components
2. **Integration tests** in `test_comprehensive.py` - Test component interactions
3. **Performance tests** in `test_performance.py` - Test timing/speed
4. **Upload tests** in `test_upload.py` - Test file handling

Example test structure:
```python
def test_new_feature(self):
    """Test description of what this verifies."""
    # Arrange - Set up test data
    test_input = "sample data"
    
    # Act - Execute the code
    result = your_function(test_input)
    
    # Assert - Verify expected behavior
    self.assertTrue(result['success'])
    self.assertEqual(result['output'], "expected")
```

## Continuous Integration

Tests are designed to work in CI/CD pipelines:
- JUnit XML output for CI systems
- Exit code 0 (pass) or 1 (fail)
- Can run without interactive input
- Skip tests when API keys unavailable

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
- Run: `pytest tests/` from project root
- `conftest.py` should auto-configure paths

**"No API key found"**
- Tests skip if keys unavailable (not a failure)
- Set keys in `.env` for full coverage

**"Tests too slow"**
- Performance tests have generous time limits
- If failing, check your internet connection
- Consider running with `-m "not performance"` to skip

**"Import errors"**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Activate conda environment: `conda activate youtube_ai`

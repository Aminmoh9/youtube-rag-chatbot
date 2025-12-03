"""
Unified processor for all 4 input methods with consistent output.
DEPRECATED: Use ContentProcessor from content_processor.py instead.
This file maintained for backward compatibility.
"""
from .content_processor import ContentProcessor, content_processor

# Backward compatibility aliases
UnifiedContentProcessor = ContentProcessor
unified_content_processor = content_processor  # Alternative import name

# Export for imports
__all__ = ['UnifiedContentProcessor', 'content_processor', 'unified_content_processor']

"""
Input method processors for the 4 different content types.
"""
from .topic_search_processor import TopicSearchProcessor
from .youtube_link_processor import YouTubeLinkProcessor
from .audio_video_processor import AudioVideoProcessor
from .script_upload_processor import ScriptUploadProcessor

__all__ = [
    'TopicSearchProcessor',
    'YouTubeLinkProcessor',
    'AudioVideoProcessor',
    'ScriptUploadProcessor'
]

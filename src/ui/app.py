"""
YouTube AI Research Assistant - Main Streamlit Application.
Modular version with clean separation of concerns.
"""
import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import importlib

# Load environment variables
load_dotenv()

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports with reload to get latest changes
from src.auth.api_key_manager import api_key_manager
from src.processors.unified_content_processor import content_processor
from src.ui.langsmith_feedback import feedback_ui

# Force reload QA model to get latest changes
import src.qa.qa_model
importlib.reload(src.qa.qa_model)
from src.qa.qa_model import QAModel

# UI Components
from src.ui.voice_utils import render_voice_settings, TTS_AVAILABLE
from src.ui.display_components import display_topic_search_summary, display_single_content_summary
from src.ui.input_tabs import (
    render_topic_search_tab,
    render_youtube_link_tab,
    render_audio_video_upload_tab,
    render_script_upload_tab
)
from src.ui.qa_interface import render_qa_interface
from src.ui.styles import load_css

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="YouTube AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize QA model
@st.cache_resource
def get_qa_model():
    """Get cached QA model instance."""
    return QAModel(enable_tracing=True)


@st.cache_resource
def get_content_processor():
    """Get cached ContentProcessor instance."""
    from src.processors.unified_content_processor import content_processor
    return content_processor


@st.cache_resource
def get_pinecone_manager():
    """Get cached PineconeDataManager instance."""
    from src.utils.pinecone_manager import PineconeDataManager
    return PineconeDataManager()


@st.cache_data(ttl=60)
def get_topics_list():
    """Get cached list of topics (refreshes every 60 seconds)."""
    manager = get_pinecone_manager()
    return manager.list_all_topics()


# Load CSS
load_css()


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'current_session': None,
        'session_history': [],
        'voice_enabled': True,
        'selected_voice': 'nova',
        'auto_play': False,
        'generate_summary_audio': False,
        'used_voice_input': False,
        'voice_question_text': '',
        'last_audio_bytes': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# Normalize any existing session_history entries so older saved JSON files
# (which may be top-level objects) get a consistent `data` wrapper.
def _normalize_session_entry(sess):
    # If session already has nested `data`, assume it's normalized
    if sess is None:
        return sess
    if 'data' in sess and isinstance(sess.get('data'), dict):
        return sess

    # Otherwise wrap top-level keys under `data`
    normalized = {
        'session_id': sess.get('session_id') or sess.get('id') or None,
        'type': sess.get('type') or sess.get('input_method') or 'topic_search',
        'topic': sess.get('topic') or (sess.get('topic_summary', {}) and sess.get('topic_summary').get('topic')) or None,
        'data': sess
    }
    return normalized


def _truncate_text(text: str, length: int = 34) -> str:
    """Return a single-line truncated string with an ellipsis if too long."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= length:
        return text
    return text[: max(0, length - 1)].rstrip() + "…"


def _session_icon(sess: dict) -> str:
    """Choose an icon emoji based on session metadata to make sessions distinguishable.

    Mapping is conservative and based on `type`, `input_method`, and common keywords
    found in the session topic/title.
    """
    if not sess:
        return "📄"
    data = sess.get('data') or {}

    # Prefer explicit fields first: `input_method`, `content_type`, `type`.
    input_method = str(sess.get('input_method') or data.get('input_method') or '').lower()
    content_type = str(sess.get('content_type') or data.get('content_type') or '').lower()
    stype = str(sess.get('type') or '').lower()

    # Exact mappings (explicit)
    # Topic search (multiple videos)
    if 'topic_search' in input_method or 'topic_search' in stype:
        return "🔍"

    # YouTube link / single video -> use chain/link icon as requested
    if 'youtube_link' in input_method or input_method == 'youtube' or 'youtube' in content_type or 'youtube' in stype:
        return "🔗"

    # Audio / audio upload -> musical note
    if ('audio' in input_method) or ('audio' in content_type) or ('audio' in stype) or 'audio' in str(data.get('content_type', '')).lower():
        return "🎵"

    # Script / text upload
    if 'script' in input_method or 'script' in content_type or 'script' in stype or 'text' in content_type:
        return "📄"

    # Video uploads (generic) -> clapper if explicit
    if 'video' in input_method or 'video' in content_type or 'youtube_videos' in content_type:
        return "🎬"

    # Do not override the icon just because a session was loaded from Pinecone.
    # The original input method (what created the session) is the primary signal.
    # Fallback: document icon
    return "📄"

# Apply normalization in-place for any pre-existing session history
if st.session_state.get('session_history'):
    normalized_history = []
    for s in st.session_state.session_history:
        normalized_history.append(_normalize_session_entry(s))
    st.session_state.session_history = normalized_history


# Sidebar
with st.sidebar:
    st.markdown("# 🔍 YouTube AI Research")
    st.markdown("---")
    
    # API Key Management
    api_key_manager.render_ui()
    
    # Voice Settings
    if TTS_AVAILABLE:
        render_voice_settings()
    
    st.markdown("---")
    st.markdown("### 🎯 Research Methods")
    st.markdown("""
    1. **Topic Search** - Multiple videos
    2. **YouTube Link** - Single video
    3. **Upload Audio/Video** - Local file
    4. **Upload Script** - Text file
    """)
    
    # Load from Pinecone
    st.markdown("---")
    st.markdown("### 🗄️ Load from Database")
    
    with st.expander("📂 Browse Stored Topics", expanded=False):
        try:
            # Use cached topics list (refreshes every 60 seconds)
            topics = get_topics_list()
            
            if topics:
                st.markdown(f"**{len(topics)} topics in database**")
                st.markdown("*Click a topic to load it:*")
                
                for i, topic in enumerate(topics[:10]):  # Show first 10
                    topic_name = topic.get('topic_name', 'Unknown')
                    vector_count = topic['vector_count']
                    topic_hash = topic.get('topic_hash')  # Get actual hash from Pinecone
                    
                    # Make each topic clickable - use unique key based on hash
                    if st.button(
                        f"📁 {topic_name} ({vector_count} vectors)",
                        key=f"load_topic_{topic_hash}",  # Use hash instead of index for unique key
                        use_container_width=True
                    ):
                        # Load this specific topic using its hash
                        import uuid
                        new_session = {
                            'session_id': str(uuid.uuid4()),
                            'type': 'pinecone_existing',
                            'topic': topic_name,
                            'data': {
                                'topic': topic_name,
                                'topic_hash': topic_hash,  # Store actual hash
                                'namespace': f"topic-{topic_hash}",  # Store full namespace
                                'success': True,
                                'loaded_from_pinecone': True
                            },
                            'time': 'loaded from database'
                        }
                        # Try to load summary from disk
                        import os, json
                        session_dir = os.path.join(os.path.dirname(__file__), '../../data/content_sessions')
                        # Find file by topic name or namespace
                        for fname in os.listdir(session_dir):
                            if fname.endswith('.json'):
                                fpath = os.path.join(session_dir, fname)
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    session_json = json.load(f)
                                    # Check multiple possible locations for topic and summary
                                    sess_topic = session_json.get('topic') or session_json.get('data', {}).get('topic')
                                    sess_namespace = None
                                    if session_json.get('pinecone_result'):
                                        sess_namespace = session_json.get('pinecone_result', {}).get('namespace')
                                    elif isinstance(session_json.get('data'), dict) and session_json['data'].get('pinecone_result'):
                                        sess_namespace = session_json['data'].get('pinecone_result', {}).get('namespace')

                                    if (sess_topic and str(sess_topic).lower() == str(topic_name).lower()) or (sess_namespace and sess_namespace == f"topic-{topic_hash}"):
                                        # Prefer data.summary, then top-level summary, then topic_summary
                                        summary = None
                                        if isinstance(session_json.get('data'), dict) and session_json['data'].get('summary'):
                                            summary = session_json['data']['summary']
                                        elif session_json.get('summary'):
                                            summary = session_json.get('summary')
                                        elif session_json.get('topic_summary'):
                                            summary = session_json.get('topic_summary')

                                        if summary:
                                            new_session['data']['summary'] = summary
                                            # if video summaries present, include them in session data for topic view
                                            if isinstance(summary, dict) and summary.get('video_summaries'):
                                                new_session['data']['video_summaries'] = summary.get('video_summaries')
                                                new_session['data']['video_count'] = len(summary.get('video_summaries', []))
                                            # also map top-level video_summaries if present
                                            if session_json.get('video_summaries') and not new_session['data'].get('video_summaries'):
                                                new_session['data']['video_summaries'] = session_json.get('video_summaries')
                                                new_session['data']['video_count'] = len(session_json.get('video_summaries', []))
                                            # Preserve original input method / content type when available so icons reflect creation method
                                            orig_input = None
                                            orig_content_type = None
                                            if session_json.get('input_method'):
                                                orig_input = session_json.get('input_method')
                                            elif session_json.get('type'):
                                                orig_input = session_json.get('type')
                                            elif isinstance(session_json.get('data'), dict) and session_json['data'].get('input_method'):
                                                orig_input = session_json['data'].get('input_method')

                                            if session_json.get('content_type'):
                                                orig_content_type = session_json.get('content_type')
                                            elif isinstance(session_json.get('data'), dict) and session_json['data'].get('content_type'):
                                                orig_content_type = session_json['data'].get('content_type')

                                            if orig_input:
                                                new_session['input_method'] = orig_input
                                                new_session['data']['input_method'] = orig_input
                                            if orig_content_type:
                                                new_session['content_type'] = orig_content_type
                                                new_session['data']['content_type'] = orig_content_type
                                            break
                        st.session_state.current_session = new_session
                        st.session_state.session_history.append(new_session)
                        st.success(f"✅ Loaded: {topic_name}")
                
                # Manual topic name entry removed — users can load topics by clicking the
                # buttons above. This keeps the sidebar compact and avoids ambiguous text input.
            else:
                st.info("No topics in database yet. Upload some content first!")
        except Exception as e:
            st.error(f"Error loading topics: {str(e)}")
    
    # Session History
    if st.session_state.session_history:
        st.markdown("---")
        st.markdown("### 📚 Recent Sessions")
        st.markdown("*Click to view summary & ask questions*")
        for i, sess in enumerate(reversed(st.session_state.session_history[-10:])):
            sess_type = sess.get('type', 'Unknown')
            sess_data = sess.get('data', {})
            topic = sess_data.get('topic', 'Unknown')
            session_id = sess.get('session_id', 'unknown')
            cols = st.columns([8,1])

            # Derive a compact label and an icon based on session metadata.
            icon = _session_icon(sess)
            truncated = _truncate_text(topic, length=34)

            # Left: clickable load button for the session (restores switching)
            with cols[0]:
                # Show a short label (truncated) and use the full topic as the button help/tooltip
                if st.button(f"{icon} {truncated}", key=f"load_session_{session_id}", use_container_width=True, help=topic):
                    # Normalize the session when loading so both top-level and nested forms work
                    loaded = _normalize_session_entry(sess)
                    st.session_state.current_session = loaded
                    st.success(f"Loaded session: {topic}")
                    st.rerun()

            # Right: small delete button
            with cols[1]:
                # Using a compact label and no container width to keep the button small
                if st.button("✖", key=f"delete_session_{session_id}", use_container_width=False, help="Delete session"):
                    st.session_state.session_history = [s for s in st.session_state.session_history if s.get('session_id') != session_id]
                    # Clear current_session if it was the one deleted
                    if st.session_state.get('current_session', {}).get('session_id') == session_id:
                        st.session_state.current_session = None
                    st.success(f"Session deleted: {topic}")
                    st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.session_history = []
        st.session_state.current_session = None
        # Removed st.rerun() to prevent repeated reruns and flicker


# Main content
st.title("🔍 YouTube AI Research Assistant")
st.markdown("Analyze YouTube videos, audio files, and scripts with AI-powered Q&A")

# Always show input methods at the top
st.markdown("### 📥 Input Methods")

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Topic Search",
    "🔗 YouTube Link",
    "🎵 Audio/Video Upload",
    "📝 Script Upload"
])

# Get cached processor
content_processor = get_content_processor()

# (Removed duplicate rendering of current session here to avoid duplicate widgets)

with tab1:
    render_topic_search_tab()

with tab2:
    render_youtube_link_tab()

with tab3:
    render_audio_video_upload_tab()

with tab4:
    render_script_upload_tab()# Help section
st.markdown("---")
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    ### Choose your research method:
    
    #### 1. 🔍 **Topic Search** 
    - Search YouTube for videos on any topic
    - Get summaries from multiple videos
    - Best for: Research, learning new topics
    
    #### 2. 🔗 **YouTube Link**
    - Analyze a specific YouTube video
    - Get detailed Q&A capability
    - Best for: Deep dives into specific content
    
    #### 3. 🎵 **Upload Audio/Video**
    - Upload local media files
    - Automatic transcription with Whisper
    - Best for: Personal recordings, lectures
    
    #### 4. 📝 **Upload Script**
    - Upload text transcripts
    - Direct text analysis
    - Best for: Written content, existing transcripts
    
    ---
    
    **💡 Tip:** You can use multiple tabs to add different types of content. Each will create a new session accessible from the sidebar.
    """)

st.markdown("---")

# Check for active session
if st.session_state.current_session:
    session = st.session_state.current_session
    
    # Display summary based on type
    topic_name = session['data'].get('topic', 'Unknown')
    st.markdown(f"### 📊 Current Session: **{topic_name}**")

    # Check if loaded from Pinecone
    if session.get('type') == 'pinecone_existing' or session['data'].get('loaded_from_pinecone'):
        st.info("🗄️ **Loaded from Database** - This content was previously uploaded and retrieved from Pinecone")

        from src.utils.pinecone_manager import PineconeDataManager
        manager = PineconeDataManager()

        # Use namespace from session data if available (correct hash)
        namespace = session.get('data', {}).get('namespace')

        if namespace:
            # Get stats for the actual namespace
            stats = manager.get_index_stats()
            namespace_stats = stats.get('namespaces', {}).get(namespace, {})
            vector_count = namespace_stats.get('vector_count', 0)
        else:
            # Fallback to computing from topic name (may be incorrect hash)
            metadata = manager.get_topic_metadata(topic_name)
            namespace = metadata['namespace']
            vector_count = metadata['vector_count']

        # Show database stats
        with st.expander("📊 Database Stats", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stored Vectors", vector_count)
            with col2:
                st.metric("Namespace", namespace[-8:])
            with col3:
                st.metric("Status", "✅ Active")

    # Show summary for loaded session (only once)
    # Treat Pinecone-existing records that contain video_summaries or an overall_summary as topic_search
    data = session.get('data', {})
    looks_like_topic_search = (
        session.get('type') == 'topic_search' or
        data.get('input_method') == 'topic_search' or
        isinstance(data.get('summary'), dict) and (data['summary'].get('overall_summary') or data['summary'].get('video_summaries')) or
        data.get('video_summaries')
    )

    # Wrap the content summary in an expander so users can hide/minimize it
    with st.expander("📋 Content Summary", expanded=True):
        if looks_like_topic_search:
            display_topic_search_summary(session)
        else:
            display_single_content_summary(session)
    
    st.markdown("---")
    
    # Q&A Interface
    qa_model = get_qa_model()
    render_qa_interface(qa_model, session)
    
    # Performance Metrics (if enabled)
    if st.session_state.get('show_metrics', False):
        st.markdown("---")
        feedback_ui.show_performance_metrics(days=7)

else:
    # No active session - show welcome message
    st.info("👆 **Get started by choosing an input method above!**")
    st.markdown("""
    - Use **Topic Search** to research multiple YouTube videos
    - Paste a **YouTube Link** to analyze a specific video
    - **Upload Audio/Video** files for transcription and analysis
    - **Upload Scripts** for direct text analysis
    
    All your sessions will appear in the sidebar for easy access.
    """)
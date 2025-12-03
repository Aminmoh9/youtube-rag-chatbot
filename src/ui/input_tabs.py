"""
Input method tabs for the 4 different content processing methods.
"""
import streamlit as st
from src.processors.unified_content_processor import content_processor
from src.ui.voice_utils import transcribe_audio, TTS_AVAILABLE


def render_topic_search_tab():
    # Restore summary for loaded topic_search session
    if 'current_session' in st.session_state and st.session_state.current_session and st.session_state.current_session.get('type') == 'topic_search':
        session_data = st.session_state.current_session.get('data', {})
        if 'summary' not in session_data:
            if 'summary' in st.session_state.current_session:
                session_data['summary'] = st.session_state.current_session['summary']
            elif 'topic_summary' in st.session_state.current_session:
                session_data['summary'] = st.session_state.current_session['topic_summary']
            st.session_state.current_session['data'] = session_data
    """Tab 1: Topic Search → Fetch multiple YouTube videos."""
    st.markdown("### 🔍 Search YouTube for Topics")
    st.markdown("Search for a topic and get summaries from multiple videos.")
    
    # Initialize voice input state
    if 'voice_topic_text' not in st.session_state:
        st.session_state.voice_topic_text = ""
    
    # Voice input for topic
    if TTS_AVAILABLE:
        col1, col2 = st.columns([4, 1])
        
        with col2:
            # Voice input button that listens when clicked
            if st.button("🎙️ Voice", key="voice_topic_btn", use_container_width=True):
                from src.ui.voice_utils import listen_to_question
                voice_topic = listen_to_question(timeout=5, phrase_time_limit=20)
                
                if voice_topic:
                    st.session_state.voice_topic_text = voice_topic
                st.rerun()
        
        with col1:
            topic = st.text_input(
                "Enter research topic or use voice",
                value=st.session_state.voice_topic_text,
                placeholder="e.g., 'New AI tools in 2025'",
                label_visibility="collapsed"
            )
    else:
        topic = st.text_input(
            "Enter research topic",
            placeholder="e.g., 'New AI tools in 2025'"
        )
    
    max_videos = st.slider("Number of videos", 2, 10, 5)
    
    with st.expander("💡 Tips", expanded=False):
        st.markdown("""
        - Use specific topics for better results
        - More videos = more comprehensive research
        - Processing takes ~30 seconds per video
        """)
    
    if st.button("🔍 Search & Analyze", type="primary", key="search_btn"):
        if not topic:
            st.error("Please enter a topic")
            return
        
        with st.spinner(f"Searching for videos about '{topic}'..."):
            result = content_processor.process_topic_search(
                topic=topic,
                max_videos=max_videos,
                require_consent=True
            )
        
        if result['success']:
            st.success(f"✅ Processed {result['video_count']} videos!")
            st.session_state.current_session = {
                'session_id': result['session_id'],
                'type': 'topic_search',
                'data': result,
                'time': st.session_state.get('time', 'now')
            }
            st.session_state.session_history.append(st.session_state.current_session)
            # Extract video titles and IDs for dropdown
            video_summaries = result.get('video_summaries', [])
            if video_summaries:
                video_options = {v['title']: v['video_id'] for v in video_summaries}
                selected_title = st.selectbox(
                    "Filter Q&A by video:",
                    options=list(video_options.keys()),
                    index=0
                )
                st.session_state.selected_video_id = video_options[selected_title]
            else:
                st.session_state.selected_video_id = None
            # Clear voice input after successful search
            st.session_state.voice_topic_text = ""
            st.rerun()
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")


def render_youtube_link_tab():
    # Restore summary for loaded YouTube link session
    if 'current_session' in st.session_state and st.session_state.current_session and st.session_state.current_session.get('type') == 'youtube_link':
        session_data = st.session_state.current_session.get('data', {})
        if 'summary' not in session_data:
            if 'summary' in st.session_state.current_session:
                session_data['summary'] = st.session_state.current_session['summary']
            st.session_state.current_session['data'] = session_data
    """Tab 2: Single YouTube Link."""
    st.markdown("### 🔗 Analyze Specific YouTube Video")
    st.markdown("Paste a YouTube URL to analyze a single video.")
    
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    consent = st.checkbox(
        "I consent to download audio if subtitles unavailable",
        value=False
    )
    
    if st.button("📊 Analyze Video", type="primary", key="youtube_btn"):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
            return
        
        with st.spinner("Processing video..."):
            result = content_processor.process_youtube_link(
                youtube_url=youtube_url,
                consent_given=consent
            )
        
        if result['success']:
            st.success("✅ Video processed!")
            st.session_state.current_session = {
                'session_id': result['session_id'],
                'type': 'youtube_link',
                'data': result,
                'time': st.session_state.get('time', 'now')
            }
            st.session_state.session_history.append(st.session_state.current_session)
            st.rerun()
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")


def render_audio_video_upload_tab():
    # Restore summary for loaded audio/video upload session
    if 'current_session' in st.session_state and st.session_state.current_session and st.session_state.current_session.get('type') == 'audio_video_upload':
        session_data = st.session_state.current_session.get('data', {})
        if 'summary' not in session_data:
            if 'summary' in st.session_state.current_session:
                session_data['summary'] = st.session_state.current_session['summary']
            st.session_state.current_session['data'] = session_data
    """Tab 3: Audio/Video Upload."""
    st.markdown("### 🎵 Upload Audio/Video File")
    st.markdown("Upload audio or video to transcribe and analyze.")
    
    uploaded_file = st.file_uploader(
        "Choose audio/video file",
        type=['mp3', 'mp4', 'wav', 'm4a', 'webm'],
        key="audio_upload"
    )
    
    consent = st.checkbox(
        "I consent to processing this file with OpenAI Whisper",
        value=False,
        key="audio_consent"
    )
    
    if uploaded_file and consent:
        # Show processing time estimate
        file_size_mb = uploaded_file.size / (1024 * 1024)
        estimated_minutes = max(1, int(file_size_mb / 5))
        
        if file_size_mb > 15:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Estimated processing time: **{estimated_minutes}+ minutes**. Once started, this cannot be cancelled.")
        else:
            st.info(f"ℹ️ File size: {file_size_mb:.1f} MB. Estimated time: ~{estimated_minutes} minute(s)")
        
        if st.button("🎵 Transcribe & Analyze", type="primary", key="audio_btn"):
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            # Progress tracking with percentage
            progress_bar = st.progress(0)
            status_text = st.empty()
            percent_text = st.empty()
            
            # Step 1: Reading file (0-10%)
            status_text.text("⏳ Step 1/3: Reading file...")
            percent_text.text("Progress: 0%")
            progress_bar.progress(0)
            file_bytes = uploaded_file.read()
            
            progress_bar.progress(10)
            percent_text.text("Progress: 10%")
            
            # Step 2: Transcribing (10-80% - longest step)
            if file_size_mb > 20:
                status_text.text(f"🎙️ Step 2/3: Transcribing with local Whisper ({file_size_mb:.1f} MB)... Est. {estimated_minutes}+ min")
            else:
                status_text.text(f"🎙️ Step 2/3: Transcribing audio with local Whisper AI...")
            percent_text.text("Progress: 10% (Transcription in progress...)")
            progress_bar.progress(10)
            
            result = content_processor.process_audio_video_upload(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                file_type='audio' if uploaded_file.type.startswith('audio') else 'video',
                consent_given=True
            )
            
            # Step 3: Post-processing (80-100%)
            if result['success']:
                progress_bar.progress(80)
                percent_text.text("Progress: 80%")
                status_text.text("📊 Step 3/3: Generating summary and embeddings...")
                
                progress_bar.progress(100)
                percent_text.text("Progress: 100% ✓")
                status_text.text("✅ Complete!")
                
                st.success(f"✅ File processed! Created {result.get('chunk_count', 0)} text chunks")
                
                # Clean up progress indicators after a moment
                import time
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                percent_text.empty()
                
                st.session_state.current_session = {
                    'session_id': result['session_id'],
                    'type': 'audio_video_upload',
                    'data': result,
                    'time': st.session_state.get('time', 'now')
                }
                st.session_state.session_history.append(st.session_state.current_session)
                st.rerun()
            else:
                progress_bar.empty()
                status_text.empty()
                percent_text.empty()
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")


def render_script_upload_tab():
    # Restore summary for loaded script upload session
    if 'current_session' in st.session_state and st.session_state.current_session and st.session_state.current_session.get('type') == 'script_upload':
        session_data = st.session_state.current_session.get('data', {})
        if 'summary' not in session_data:
            if 'summary' in st.session_state.current_session:
                session_data['summary'] = st.session_state.current_session['summary']
            st.session_state.current_session['data'] = session_data
    """Tab 4: Script Upload."""
    
    st.markdown("### 📝 Upload Script/Transcript")
    st.markdown("Upload a text file containing a transcript or script.")
    
    uploaded_file = st.file_uploader(
        "Choose text file",
        type=['txt', 'md', 'srt'],
        key="script_upload"
    )
    
    if uploaded_file:
        if st.button("📝 Process Script", type="primary", key="script_btn"):
            with st.spinner("Processing script..."):
                result = content_processor.process_script_upload(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name
                )
            
            if result['success']:
                st.success("✅ Script processed!")
                st.session_state.current_session = {
                    'session_id': result['session_id'],
                    'type': 'script_upload',
                    'data': result,
                    'time': st.session_state.get('time', 'now')
                }
                st.session_state.session_history.append(st.session_state.current_session)
                st.rerun()
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

"""
Q&A interface for asking questions about processed content.
"""
import streamlit as st
from src.ui.voice_utils import text_to_speech, TTS_AVAILABLE, listen_to_question


def render_qa_interface(qa_model, session):
    """
    Render the Q&A interface as a chat window with conversation history.
    
    Args:
        qa_model: QAModel instance
        session: Current session data
    """
    st.markdown("### 💬 Chat with Your Research")
    
    # Initialize chat history for this session
    session_id = session['session_id']
    chat_key = f"chat_history_{session_id}"
    
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    # Display chat history
    if st.session_state[chat_key]:
        st.markdown("#### Conversation History")
        chat_container = st.container()
        
        with chat_container:
            for i, msg in enumerate(st.session_state[chat_key]):
                # Question
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🙋 You:</strong> {msg['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                with st.container():
                    st.markdown("**🤖 Assistant:**")
                    st.markdown(msg['answer'])
                
                # Show sources with timestamps if available
                sources = msg.get('sources', [])
                if sources:
                    # Sort by score if available (higher is better)
                    sorted_sources = sorted(sources, key=lambda x: x.get('score', 0), reverse=True)
                    
                    with st.expander(f"📎 Sources ({len(sorted_sources)})", expanded=False):
                        for j, source in enumerate(sorted_sources, 1):
                            metadata = source.get('metadata', {})
                            video_url = metadata.get('video_url') or metadata.get('url') or source.get('url')
                            timestamp = metadata.get('timestamp')
                            video_id = metadata.get('video_id')
                            chapter_title = metadata.get('chapter_title')
                            chunk_type = metadata.get('chunk_type')
                            score = source.get('score', 0)
                            
                            # Check if this is YouTube content
                            if video_id and video_url:
                                # Build the source description
                                source_parts = ["**Source {j}:** 🎥"]
                                
                                # Add relevance indicator for top source
                                relevance_icon = "⭐ " if j == 1 else ""
                                
                                if timestamp is not None:
                                    ts_int = int(timestamp)
                                    
                                    # Format timestamp
                                    hours = ts_int // 3600
                                    minutes = (ts_int % 3600) // 60
                                    secs = ts_int % 60
                                    
                                    if hours > 0:
                                        time_str = f"{hours}:{minutes:02d}:{secs:02d}"
                                    else:
                                        time_str = f"{minutes}:{secs:02d}"
                                    
                                    # Create clickable link with timestamp
                                    if '?' in video_url:
                                        timestamp_url = f"{video_url}&t={ts_int}s"
                                    else:
                                        timestamp_url = f"{video_url}?t={ts_int}s"
                                    
                                    # Show chapter title if available, otherwise show time
                                    if chapter_title:
                                        st.markdown(f"**Source {j}:** {relevance_icon}🎥 **{chapter_title}** - [{time_str}]({timestamp_url}) `{score:.3f}`")
                                    elif ts_int > 0 or len(sources) > 1:
                                        # Only show timestamp if it's not 0:00 or if there are multiple chunks
                                        st.markdown(f"**Source {j}:** {relevance_icon}🎥 [{time_str}]({timestamp_url}) - Jump to video `{score:.3f}`")
                                    else:
                                        # Single chunk at 0:00 - just show video link
                                        st.markdown(f"**Source {j}:** {relevance_icon}🎥 [Full video]({video_url}) `{score:.3f}`")
                                else:
                                    # No timestamp available (old data)
                                    st.markdown(f"**Source {j}:** {relevance_icon}🎥 [Video Link]({video_url})")
                            elif source.get('url'):
                                st.markdown(f"**Source {j}:** {source.get('title', 'Unknown')}")
                            
                            # Show preview
                            preview = source.get('chunk_preview', '')
                            if preview:
                                # Show more context for verification (300 chars instead of 150)
                                preview_length = min(300, len(preview))
                                st.caption(f"📄 Transcript excerpt: {preview[:preview_length]}...")
                
                # TTS playback and download for each answer
                if TTS_AVAILABLE:
                    play_col, download_col = st.columns([1,1])
                    with play_col:
                        if st.button(f"🔊 Play Answer #{i+1}", key=f"play_{session_id}_{i}"):
                            with st.spinner("Generating speech..."):
                                audio_bytes = text_to_speech(
                                    msg['answer'],
                                    voice=st.session_state.get('selected_voice', 'nova')
                                )
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                    with download_col:
                        # Allow downloading the answer as an MP3 without relying on the browser's autogenerated file
                        if st.button(f"💾 Download Answer #{i+1}", key=f"dl_{session_id}_{i}"):
                            with st.spinner("Preparing download..."):
                                audio_bytes = text_to_speech(
                                    msg['answer'],
                                    voice=st.session_state.get('selected_voice', 'nova')
                                )
                                if audio_bytes:
                                    filename = f"answer_{session_id}_{i+1}.mp3"
                                    st.download_button(
                                        label="Download MP3",
                                        data=audio_bytes,
                                        file_name=filename,
                                        mime="audio/mpeg",
                                        key=f"download_btn_{session_id}_{i}"
                                    )
        
        st.markdown("---")
    
    # New question input section
    st.markdown("#### Ask a New Question")
    
    # Initialize session state for question input (before widget is created)
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ''
    if 'used_voice_input' not in st.session_state:
        st.session_state.used_voice_input = False
    if 'clear_question_flag' not in st.session_state:
        st.session_state.clear_question_flag = False
    
    # Clear the question input if flag is set (before widget renders)
    if st.session_state.clear_question_flag:
        st.session_state.question_input = ''
        st.session_state.clear_question_flag = False
    
    # Voice input button
    if TTS_AVAILABLE:
        col1, col2 = st.columns([5, 1])
        
        with col2:
            if st.button("🎙️ Voice", key=f"voice_btn_main_{session.get('session_id', '')}", use_container_width=True):
                with st.spinner("Speak now..."):
                    voice_question = listen_to_question(timeout=10, phrase_time_limit=30)
                
                if voice_question:
                    # Set widget state directly
                    st.session_state.question_input = voice_question
                    st.session_state.used_voice_input = True
                    st.rerun()
        
        with col1:
            st.markdown("**Type or use voice:**")
    
    # Question input - no value parameter, only key (managed via session state)
    question = st.text_input(
        "Your question:",
        placeholder="What are the main points discussed?",
        key="question_input",
        label_visibility="collapsed"
    )

    # Use selected video ID from topic search dropdown if available
    video_id_filter = st.session_state.get('selected_video_id', None)

    # Ask button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state[chat_key] = []
            st.session_state.clear_question_flag = True  # Set flag to clear on next render
            st.session_state.used_voice_input = False
            st.rerun()

    if ask_button and question:
        try:
            # Import metrics tracker
            from src.evaluation.metrics_tracker import get_metrics_tracker
            import time

            # Start timing
            start_time = time.time()
            
            # Query the session
            topic = session.get('data', {}).get('topic', session.get('topic', 'general'))
            # Get namespace from session data (if available from loaded topics)
            namespace = session.get('data', {}).get('namespace')

            # NOTE: filters support removed - we no longer apply metadata filters to retrieval

            # Display question immediately
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>🙋 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)

            # Stream the answer
            answer_placeholder = st.empty()
            full_answer = ""
            sources = []

            with st.spinner("🔍 Finding answer..."):
                for response in qa_model.ask_question_stream(
                    question=question,
                    session_id=session_id,
                    topic=topic,
                    namespace=namespace
                ):
                    if response.get('type') == 'token':
                        full_answer += response['content']
                        # Update display with streaming text - use container for styling
                        with answer_placeholder.container():
                            st.markdown("**🤖 Assistant:**")
                            st.markdown(full_answer + "▌")
                    elif response.get('type') == 'complete':
                        sources = response.get('sources', [])
                        # Remove cursor and show final answer with proper markdown
                        with answer_placeholder.container():
                            st.markdown("**🤖 Assistant:**")
                            st.markdown(full_answer)
                    elif response.get('type') == 'error':
                        st.error(response.get('message'))
                        st.stop()
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log metrics
            metrics_tracker = get_metrics_tracker()
            relevance_scores = [src.get('score', 0) for src in sources]
            metrics_tracker.log_qa_interaction(
                question=question,
                answer=full_answer,
                sources=sources,
                latency_ms=latency_ms,
                session_id=session_id,
                relevance_scores=relevance_scores if relevance_scores else None
            )
            
            # Add to chat history
            st.session_state[chat_key].append({
                'question': question,
                'answer': full_answer,
                'sources': sources,
                'latency_ms': latency_ms,
                'relevance_scores': relevance_scores
            })
            
            # Set flag to clear input on next render
            st.session_state.clear_question_flag = True
            was_voice_input = st.session_state.get('used_voice_input', False)
            st.session_state.used_voice_input = False
            
            # Auto-play if voice was used and auto-play is enabled
            if TTS_AVAILABLE and was_voice_input and st.session_state.get('auto_play'):
                with st.spinner("Generating speech..."):
                    audio_bytes = text_to_speech(
                        full_answer,
                        voice=st.session_state.get('selected_voice', 'nova')
                    )
                    if audio_bytes:
                        # Display audio player - it will autoplay before rerun
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Always rerun to show updated chat and clear input
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

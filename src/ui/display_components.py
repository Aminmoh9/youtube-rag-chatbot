"""
UI components for displaying content summaries and video cards.
"""
import streamlit as st
from src.processors.content_processor import content_processor
from src.audio.voice_response_system import voice_system
import os


def _safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        import importlib
        # Try the modern runtime exception class if available
        try:
            mod = importlib.import_module("streamlit.runtime.scriptrunner.script_runner")
            raise getattr(mod, "RerunException")()
        except Exception:
            # Older Streamlit used streamlit.report_thread; try importing at runtime if present
            try:
                mod = importlib.import_module("streamlit.report_thread")
                raise getattr(mod, "RerunException")()
            except Exception:
                # As a last resort, stop the app
                st.stop()


def display_topic_search_summary(session):
    """Display summary for topic search results."""
    data = session['data']
    
    st.markdown("#### 📋 Topic Overview")
    st.markdown(f'<div class="summary-card">{data["summary"]["overall_summary"]}</div>', 
                unsafe_allow_html=True)
    # Button to generate audio for the overall summary on demand
    if st.session_state.get('generate_summary_audio', False):
        if st.button("🔊 Generate audio for topic summary", key=f"gen_summary_topic_{session.get('session_id')}"):
            with st.spinner("Generating summary audio..."):
                try:
                    session_info = {
                        'id': session.get('session_id'),
                        'topic': data.get('topic'),
                        'input_method': 'topic_search'
                    }
                    summary_data = {'short_summary': data['summary']['overall_summary']}
                    resp = voice_system.get_summary_audio(summary_data, session_info)
                    if resp.get('success'):
                        audio_path = resp.get('audio_path')
                        cached = resp.get('cached', False)
                        st.success("✅ Summary audio generated")
                        try:
                            st.audio(audio_path, format='audio/mp3')
                        except Exception:
                            pass
                        # Provide a download button
                        if os.path.exists(audio_path):
                            with open(audio_path, 'rb') as f:
                                audio_bytes = f.read()
                            download_label = "📥 Download summary (MP3)" + (" — cached" if cached else "")
                            st.download_button(download_label, data=audio_bytes, file_name=f"summary_{session.get('session_id')}.mp3", mime='audio/mpeg')
                    else:
                        st.error(f"Failed to generate summary audio: {resp.get('error')}")
                except Exception as e:
                    st.error(f"Error generating summary audio: {e}")
    else:
        # Keep UI minimal when audio generation is disabled
        st.info("Audio generation disabled. Enable 'Generate audio for summaries' in Voice Settings to show audio buttons.")
    
    st.markdown(f"#### 🎥 {data['video_count']} Videos Found")
    
    for i, video in enumerate(data['video_summaries'], 1):
        display_video_card(i, video)


def display_video_card(index, video):
    """Display a single video card with metadata and summaries."""
    with st.expander(f"Video {index}: {video.get('title', 'Unknown')}", expanded=False):
        # Video metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration = video.get('duration', 0)
            if duration:
                mins = duration // 60
                secs = duration % 60
                st.metric("Duration", f"{mins}:{secs:02d}")
            else:
                st.metric("Duration", "N/A")
        
        with col2:
            video_id = video.get('video_id', '')
            if video_id:
                st.markdown(f"**Video ID:** `{video_id}`")
        
        with col3:
            if video_id:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                st.markdown(f"[🔗 Watch on YouTube]({video_url})")
        
        # Display chapters if available
        chapters = video.get('chapters', [])
        num_chapters = video.get('num_chapters', 0)
        if chapters and num_chapters > 0:
            video_id = video.get('video_id', '')
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            with st.expander(f"📑 Chapters ({num_chapters})", expanded=False):
                for idx, chapter in enumerate(chapters, 1):
                    start_min = chapter['start'] // 60
                    start_sec = chapter['start'] % 60
                    chapter_link = f"{video_url}&t={chapter['start']}s"
                    st.markdown(f"{idx}. [{start_min}:{start_sec:02d}]({chapter_link}) **{chapter['title']}**")
        
        st.markdown("---")
        
        # Summary
        summary = video.get('summary')
        if isinstance(summary, dict):
            # If summarization failed, show error info
            if summary.get('success') is False:
                err = summary.get('error') or 'Unknown error while generating summary.'
                st.error(f"Summary generation failed: {err}")
                # Explain short reels may not have enough transcribed content
                duration = video.get('duration', 0)
                if duration and duration < 30:
                    st.info("This appears to be a very short clip (under 30s). Short reels sometimes don't produce transcriptions or meaningful chunks, so the automatic summarizer may skip them.")
                # Offer to re-run summarization for this single video (summary-only, no upsert)
                video_id = video.get('video_id')
                if video_id:
                    if st.button("🔁 Regenerate summary", key=f"regen_{video_id}"):
                        with st.spinner("Regenerating summary (summary-only, no new embeddings)..."):
                            try:
                                # Persist into the current parent session without creating a new session or namespace
                                parent = st.session_state.get('current_session')
                                parent_id = parent.get('session_id') if parent else None
                                if not parent_id:
                                    st.error("No parent session available to persist the regenerated summary.")
                                else:
                                    result = content_processor.regenerate_video_summary(parent_id, video_id, consent_given=True)
                                    if result.get('success'):
                                        # Update local view
                                        video['summary'] = result.get('summary')
                                        st.success("Summary regenerated and persisted to the parent session.")
                                        _safe_rerun()
                                    else:
                                        st.error(f"Regeneration failed: {result.get('error')}")
                            except Exception as e:
                                st.error(f"Error regenerating summary: {e}")
            else:
                st.markdown("**Short Summary:**")
                st.markdown(summary.get('short_summary', ''))

                if summary.get('detailed_summary'):
                    with st.expander("📖 Detailed Summary", expanded=False):
                        st.markdown(summary.get('detailed_summary', ''))

                if summary.get('key_concepts'):
                    st.markdown("**Key Concepts:**")
                    st.markdown(summary.get('key_concepts', ''))

                # Add a per-video summary audio button (only if enabled)
                if st.session_state.get('generate_summary_audio', False):
                    if st.button("🔊 Generate audio for this video summary", key=f"gen_summary_video_{video.get('video_id')}"):
                        with st.spinner("Generating video summary audio..."):
                            try:
                                session_info = {
                                    'id': st.session_state.get('current_session', {}).get('session_id'),
                                    'topic': st.session_state.get('current_session', {}).get('data', {}).get('topic'),
                                    'input_method': 'topic_search'
                                }
                                summary_data = {'short_summary': summary.get('short_summary', '')}
                                resp = voice_system.get_summary_audio(summary_data, session_info)
                                if resp.get('success'):
                                    audio_path = resp.get('audio_path')
                                    cached = resp.get('cached', False)
                                    st.success("✅ Video summary audio generated")
                                    try:
                                        st.audio(audio_path, format='audio/mp3')
                                    except Exception:
                                        pass
                                    if os.path.exists(audio_path):
                                        with open(audio_path, 'rb') as f:
                                            audio_bytes = f.read()
                                        download_label = "📥 Download video summary (MP3)" + (" — cached" if cached else "")
                                        st.download_button(download_label, data=audio_bytes, file_name=f"video_summary_{video.get('video_id')}.mp3", mime='audio/mpeg')
                                else:
                                    st.error(f"Failed to generate video summary audio: {resp.get('error')}")
                            except Exception as e:
                                st.error(f"Error generating video summary audio: {e}")
        else:
            st.markdown(f"**Summary:** {video.get('summary', '')}")


def display_single_content_summary(session):
    """Display summary for single content (YouTube link, upload, script)."""
    data = session['data']

    st.markdown("#### 📋 Content Summary")
    if 'summary' in data:
        if isinstance(data['summary'], dict):
            st.markdown(f"**Short Summary:** {data['summary'].get('short_summary', '')}")
            detailed = data['summary'].get('detailed_summary')
            if detailed:
                with st.expander("📖 Detailed Summary", expanded=False):
                    st.markdown(detailed)
        else:
            st.markdown(f'<div class="summary-card">{data["summary"]}</div>', unsafe_allow_html=True)

        # Generate summary audio on demand (only if enabled)
        if st.session_state.get('generate_summary_audio', False):
            if st.button("🔊 Generate audio for this summary", key=f"gen_summary_single_{session.get('session_id')}"):
                with st.spinner("Generating summary audio..."):
                    try:
                        session_info = {
                            'id': session.get('session_id'),
                            'topic': data.get('topic') or data.get('filename') or session.get('session_id'),
                            'input_method': session.get('type', 'unknown')
                        }
                        summary_data = {'short_summary': data['summary']}
                        resp = voice_system.get_summary_audio(summary_data, session_info)
                        if resp.get('success'):
                            audio_path = resp.get('audio_path')
                            cached = resp.get('cached', False)
                            st.success("✅ Summary audio generated")
                            try:
                                st.audio(audio_path, format='audio/mp3')
                            except Exception:
                                pass
                            if os.path.exists(audio_path):
                                with open(audio_path, 'rb') as f:
                                    audio_bytes = f.read()
                                download_label = "📥 Download summary (MP3)" + (" — cached" if cached else "")
                                st.download_button(download_label, data=audio_bytes, file_name=f"summary_{session.get('session_id')}.mp3", mime='audio/mpeg')
                        else:
                            st.error(f"Failed to generate summary audio: {resp.get('error')}")
                    except Exception as e:
                        st.error(f"Error generating summary audio: {e}")
        else:
            st.info("Audio generation disabled. Enable 'Generate audio for summaries' in Voice Settings to show audio buttons.")
    else:
        st.warning("No summary available for this session.")

    if session['type'] == 'youtube_link':
        st.info(f"📺 Source: {data.get('url', 'YouTube Video')}")
    elif session['type'] == 'audio_video_upload':
        st.info(f"🎵 Source: {data.get('filename', 'Uploaded File')}")
    elif session['type'] == 'script_upload':
        st.info(f"📝 Source: {data.get('filename', 'Uploaded Script')}")

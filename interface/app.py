"""
Minimalistic Streamlit Interface for YouTube QA Chatbot.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qa_model import MultimodalQAAgent
from models.speech_input import listen_to_question
from models.speech_output import speak_answer

# Page configuration
st.set_page_config(
    page_title="Data Analytics Q&A",
    page_icon="🎓",
    layout="wide"
)

# CSS for two-column layout
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .source-info {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 1rem;
    }
    /* Style the Ask Question button */
    .stFormSubmitButton > button {
        background-color: #0066cc;
        color: white;
    }
    .stFormSubmitButton > button:hover {
        background-color: #0052a3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'voice_question' not in st.session_state:
    st.session_state.voice_question = ""
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'is_voice_question' not in st.session_state:
    st.session_state.is_voice_question = False
    
if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = "All"

# Initialize QA agent
@st.cache_resource
def get_agent():
    return MultimodalQAAgent()

agent = get_agent()

# Header
st.markdown('<div class="main-title">Ask a question about data analytics</div>', unsafe_allow_html=True)

# Two-column layout
left_col, right_col = st.columns([1, 1.5])

# LEFT COLUMN - Input
with left_col:
    # Use form to enable Enter key submission
    with st.form(key="question_form"):
        # Text input
        question = st.text_input(
            "",
            value=st.session_state.voice_question,
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        # Mic button 
        if st.form_submit_button("🎙️", help="Click to speak your question"):
            st.info("🎙️ Listening... Speak now!")
            voice_q = listen_to_question(timeout=3, phrase_time_limit=15)

            if voice_q:
                st.session_state.voice_question = voice_q
                st.rerun()
            else:
                st.warning("No speech detected.")
                st.rerun()
                
        # Topic selector
        st.markdown("**Topic**")
        topic = st.selectbox(
            "",
            ["All", "Python", "SQL", "Excel", "Tableau", "Power BI"],
            key="selected_topic",
            label_visibility="collapsed"
        )
        
        st.markdown("")
        
        # Ask button
        ask_button = st.form_submit_button("Ask Question", use_container_width=True)
    
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q: {chat['question'][:40]}..."):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer'][:150]}...")

# RIGHT COLUMN - Answer
with right_col:
    # Process question
    if ask_button:
        # Use voice question if available, otherwise use text input
        current_question = st.session_state.voice_question if st.session_state.voice_question else question
        st.session_state.is_voice_question = bool(st.session_state.voice_question)

        if current_question:
            # Prepare topic filter (None if "All" selected)
            topic_filter = None if st.session_state.selected_topic == "All" else st.session_state.selected_topic

            with st.spinner("🤔 Thinking..."):
                result = agent.ask(current_question, topic=topic_filter)

            # Store result in session state
            st.session_state.current_result = result

            # Add to history
            st.session_state.chat_history.append({
                "question": current_question,
                "answer": result['answer']
            })

            # Clear voice question for next question
            st.session_state.voice_question = ""
    
    # Display answer if available
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Display the question
        st.markdown(f"**Question:** {result['question']}")
        st.markdown("")
        
        # Display answer heading and content
        st.markdown("**Answer:**")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
        
        # Source info - show all sources
        if result['has_sources'] and result['sources']:
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            
            for i, source_item in enumerate(result['sources'], 1):
                source = source_item.get('metadata', {})
                score = source_item.get('score', 0)
                
                with st.expander(f"Source {i}: {source.get('title', 'Unknown')} (Relevance: {score:.0%})"):
                    st.markdown(f"**Topic:** {source.get('topic', 'General')}")
                    st.markdown(f"**Timestamp:** {source.get('start_time', 0)}s - {source.get('end_time', 0)}s")
                    
                    if source.get('url'):
                        timestamp_url = f"{source.get('url')}&t={int(source.get('start_time', 0))}s"
                        st.markdown(f"🔗 [Watch video at this timestamp]({timestamp_url})")
        
        # Voice output
        st.markdown("")
        
        # Auto-play for voice questions, manual button for text questions
        if st.session_state.is_voice_question:
            # Reset flag and auto-play once
            st.session_state.is_voice_question = False
            audio_file = speak_answer(result['answer'], voice='nova')
            if audio_file:
                st.audio(audio_file, format='audio/mp3', autoplay=True)
        else:
            if st.button("🔊 Play Answer", key="speak_button"):
                audio_file = speak_answer(result['answer'], voice='nova')
                if audio_file:
                    st.audio(audio_file, format='audio/mp3', autoplay=True)
    else:
        # Placeholder
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 8px; 
                    border: 1px solid #e0e0e0; color: #6c757d; text-align: center;">
            Your answer will appear here...
        </div>
        """, unsafe_allow_html=True)

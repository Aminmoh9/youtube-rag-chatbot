"""
Performance Dashboard - Real-time metrics for app optimization.
Run as: streamlit run src/evaluation/performance_dashboard.py
"""
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Performance Dashboard", page_icon="📊", layout="wide")

st.title("📊 YouTube AI - Performance Dashboard")
st.markdown("Real-time performance metrics and optimization insights")

# Load session data
def load_session_data():
    """Load all session data from content_sessions folder."""
    sessions_dir = Path("data/content_sessions")
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for file in sessions_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['file'] = file.name
                sessions.append(data)
        except Exception as e:
            continue
    return sessions

sessions = load_session_data()

if not sessions:
    st.warning("⚠️ No session data found. Use the app to generate some data first!")
    st.stop()

# Calculate metrics
total_sessions = len(sessions)
input_methods = defaultdict(int)
topics = defaultdict(int)
timestamps = []

for session in sessions:
    input_methods[session.get('input_method', 'unknown')] += 1
    topics[session.get('topic', 'unknown')] += 1
    # Check for both 'timestamp' and 'created_at' fields
    timestamp_str = session.get('timestamp') or session.get('created_at')
    if timestamp_str:
        try:
            timestamps.append(datetime.fromisoformat(timestamp_str))
        except:
            pass

# === TOP METRICS ===
st.markdown("### 🎯 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sessions", total_sessions)

with col2:
    if timestamps:
        last_24h = sum(1 for ts in timestamps if ts > datetime.now() - timedelta(days=1))
        st.metric("Last 24h Sessions", last_24h)
    else:
        st.metric("Last 24h Sessions", "N/A")

with col3:
    avg_chunks = sum(session.get('chunk_count', 0) for session in sessions) / max(len(sessions), 1)
    st.metric("Avg Chunks/Session", f"{avg_chunks:.1f}")

with col4:
    unique_topics = len(topics)
    st.metric("Unique Topics", unique_topics)

st.markdown("---")

# === USAGE ANALYSIS ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Input Method Distribution")
    if input_methods:
        method_df = pd.DataFrame(list(input_methods.items()), columns=['Method', 'Count'])
        fig = px.pie(method_df, values='Count', names='Method', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No input method data")

with col2:
    st.markdown("### 🔥 Top Topics")
    if topics:
        # Let the user choose how many top topics to display
        top_n = st.sidebar.slider("Top N topics to show", min_value=5, max_value=50, value=10, step=1)
        topic_df = pd.DataFrame(list(topics.items()), columns=['Topic', 'Count'])
        topic_df = topic_df.sort_values('Count', ascending=False).head(top_n)
        fig = px.bar(topic_df, x='Topic', y='Count', 
                     color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No topic data")

st.markdown("---")

# === CONTENT PROCESSING METRICS ===
st.markdown("### 📊 Content Processing Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Chunk Distribution")
    chunk_counts = [session.get('chunk_count', 0) for session in sessions if session.get('chunk_count', 0) > 0]
    if chunk_counts:
        fig = go.Figure(data=[go.Histogram(x=chunk_counts, nbinsx=20)])
        fig.update_layout(
            xaxis_title="Chunks per Session",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
        
        st.info(f"""
        - **Min chunks:** {min(chunk_counts)}
        - **Max chunks:** {max(chunk_counts)}
        - **Median chunks:** {sorted(chunk_counts)[len(chunk_counts)//2]}
        """)
    else:
        st.info("No chunk data available")

with col2:
    st.markdown("#### Session Timeline")
    if timestamps:
        timeline_df = pd.DataFrame({'timestamp': timestamps})
        timeline_df['date'] = timeline_df['timestamp'].dt.date
        daily_counts = timeline_df['date'].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2)
        )])
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sessions",
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No timestamp data")

st.markdown("---")

# === QA PERFORMANCE METRICS ===
st.markdown("### 🎯 Q&A Performance Metrics")

# Load QA metrics
metrics_file = Path("data/metrics/qa_metrics.jsonl")
qa_metrics = []
if metrics_file.exists():
    with open(metrics_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qa_metrics.append(json.loads(line))
            except:
                continue

if qa_metrics:
    # Calculate aggregate stats
    latencies = [m['latency_ms'] for m in qa_metrics if 'latency_ms' in m]
    relevance_scores = [m['avg_relevance_score'] for m in qa_metrics if m.get('avg_relevance_score')]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        # Latency categories: <1s excellent, 1-3s good, 3-5s acceptable, >5s slow
        if avg_latency < 1000:
            latency_label = "Excellent"
            latency_color = "normal"
        elif avg_latency < 3000:
            latency_label = "Good"
            latency_color = "normal"
        elif avg_latency < 5000:
            latency_label = "Acceptable"
            latency_color = "normal"
        else:
            latency_label = "Slow"
            latency_color = "inverse"
        
        st.metric("Avg Response Time", f"{avg_latency:.0f}ms", 
                 delta=latency_label,
                 delta_color=latency_color)
    
    with col2:
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        st.metric("95th Percentile", f"{p95_latency:.0f}ms")
    
    with col3:
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        st.metric("Avg Relevance", f"{avg_relevance:.2f}",
                 delta="Excellent" if avg_relevance > 0.85 else "Good" if avg_relevance > 0.75 else "Fair",
                 delta_color="normal" if avg_relevance > 0.75 else "inverse")
    
    with col4:
        total_queries = len(qa_metrics)
        st.metric("Total Q&A", total_queries)
    
    st.info("""
    **Metrics Explained:**
    - **Response Time**: <1s excellent, 1-3s good, 3-5s acceptable, >5s slow
    - **95th Percentile**: 95% of responses are faster than this (tracks worst-case performance)
    - **Relevance**: Pinecone similarity score (>0.85 excellent, >0.75 good, >0.65 fair)
    """)
    
    # Detailed metrics table
    with st.expander("📋 View Detailed Q&A Metrics", expanded=False):
        metrics_detail_df = pd.DataFrame([{
            'Timestamp': m['timestamp'][:19] if 'timestamp' in m else 'N/A',
            'Question': m['question'][:50] + '...' if len(m.get('question', '')) > 50 else m.get('question', 'N/A'),
            'Latency (ms)': f"{m['latency_ms']:.0f}" if 'latency_ms' in m else 'N/A',
            'Latency Quality': m.get('latency_category', 'N/A'),
            'Relevance Score': f"{m['avg_relevance_score']:.3f}" if m.get('avg_relevance_score') else 'N/A',
            'Relevance Quality': m.get('relevance_quality', 'N/A'),
            'Sources Used': m.get('num_sources', 0),
            'Answer Length': m.get('answer_length', 0)
        } for m in qa_metrics])
        
        st.dataframe(metrics_detail_df, width='stretch', height=400)
        
        # Export option
        if st.button("📥 Export Q&A Metrics as CSV"):
            csv = metrics_detail_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"qa_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
else:
    st.info("⚠️ No Q&A metrics yet. Ask questions in the main app to generate performance data.")

st.markdown("---")

# === OPTIMIZATION RECOMMENDATIONS ===
st.markdown("### 🎯 Optimization Recommendations")

recommendations = []

# Check chunk sizes
if chunk_counts:
    avg_chunks = sum(chunk_counts) / len(chunk_counts)
    if avg_chunks > 50:
        recommendations.append({
            "category": "⚠️ Chunking",
            "issue": "High average chunk count",
            "impact": "May slow down retrieval and increase costs",
            "recommendation": "Increase chunk_size in character_chunker.py from 1000 to 1500",
            "priority": "Medium"
        })
    elif avg_chunks < 5:
        recommendations.append({
            "category": "⚠️ Chunking",
            "issue": "Low average chunk count",
            "impact": "May lose context in answers",
            "recommendation": "Decrease chunk_size or use time-based chunking",
            "priority": "Low"
        })

# Check input method balance
if input_methods:
    total = sum(input_methods.values())
    for method, count in input_methods.items():
        if count / total > 0.8:
            recommendations.append({
                "category": "📥 Usage Pattern",
                "issue": f"{method} dominates usage ({count/total*100:.1f}%)",
                "impact": "Other features may need promotion or improvement",
                "recommendation": f"Improve UX/promotion for other input methods",
                "priority": "Low"
            })

# Check topic diversity
if unique_topics < 3 and total_sessions > 10:
    recommendations.append({
        "category": "🎯 Content Diversity",
        "issue": "Low topic diversity",
        "impact": "App may not be meeting varied user needs",
        "recommendation": "Promote diverse use cases or improve topic search",
        "priority": "Medium"
    })

# Check session frequency
if timestamps and len(timestamps) > 1:
    time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 for i in range(1, len(timestamps))]
    avg_gap = sum(time_diffs) / len(time_diffs)
    if avg_gap > 168:  # 1 week
        recommendations.append({
            "category": "👥 User Engagement",
            "issue": "Low usage frequency",
            "impact": "Users not returning regularly",
            "recommendation": "Add features to encourage regular use (saved searches, notifications)",
            "priority": "High"
        })

if recommendations:
    for rec in recommendations:
        with st.expander(f"{rec['category']}: {rec['issue']} - Priority: {rec['priority']}"):
            st.markdown(f"**Impact:** {rec['impact']}")
            st.markdown(f"**Recommendation:** {rec['recommendation']}")
else:
    st.success("✅ No critical optimization issues detected! App is performing well.")

st.markdown("---")

# === MODEL CONFIGURATION ===
st.markdown("### ⚙️ Current Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### QA Model")
    st.code(f"""
LLM: {os.getenv('LLM_MODEL', 'gpt-3.5-turbo')}
Temperature: 0.3 (Factual)
Top K Results: 3
Streaming: Enabled
    """)
    st.info("**Temperature 0.3** = Focused, factual answers. **Streaming** = Real-time word-by-word response display.")

with col2:
    st.markdown("#### Summarization")
    st.code(f"""
Model: gpt-3.5-turbo
Temperature: 0.3 (Factual)
Style: Standard/Brief
Formats: Key points, Timeline
    """)
    st.info("**Temperature 0.3** = Focused, factual summaries. Matches QA model for consistent tone.")

with col3:
    st.markdown("#### Chunking")
    st.code(f"""
Strategies:
- Chapter-based (timestamps)
- Character-based (1000 chars, 200 overlap)
- Time-based (segments)
    """)
    st.info("**Smart chunking** adapts to content type. Chapter > Character > Time-based fallback.")

st.markdown("---")

# === ADVANCED METRICS ===
with st.expander("📈 Advanced Metrics & Raw Data"):
    st.markdown("### Session Details")
    
    def _get_items_count(s):
        # Prefer explicit counts commonly used across input methods
        for key in ('video_count', 'file_count', 'num_files', 'videos', 'files'):
            val = s.get(key)
            if val is not None:
                return val
        # Fallback: if topic_search, try video_count default 1, else 1
        return s.get('video_count', 1) if s.get('input_method') == 'topic_search' else 1

    session_df = pd.DataFrame([{
        'Timestamp': s.get('created_at') or s.get('timestamp', 'N/A'),
        'Input Method': s.get('input_method', 'N/A'),
        'Topic': s.get('topic', 'N/A')[:30] + '...' if len(s.get('topic', '')) > 30 else s.get('topic', 'N/A'),
        'Videos / Files': _get_items_count(s),
        'Chunks': s.get('chunk_count', 0),
        'Session ID': s.get('session_id', 'N/A')[:20] + '...'
    } for s in sessions])
    
    st.dataframe(session_df, width='stretch')
    
    # Export option
    if st.button("📥 Export Session Data as CSV"):
        csv = session_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sessions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Last updated: {}</p>
    <p>💡 Tip: Run this dashboard regularly to monitor app health and optimize performance</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

"""
CSS styles for the YouTube AI Research Assistant.
"""

# Main application styles
APP_CSS = """
<style>
/* Summary Cards */
.summary-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #3B82F6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Video Cards */
.video-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: box-shadow 0.2s;
}

.video-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Answer Box */
.answer-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* Question Box */
.question-box {
    background: #ffffff;
    border: 2px solid #3B82F6;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Source Card */
.source-card {
    background: #f9fafb;
    border-left: 3px solid #10b981;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
}

/* Info Box */
.info-box {
    background: #e0f2fe;
    border-left: 4px solid #0284c7;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}

/* Warning Box */
.warning-box {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}

/* Error Box */
.error-box {
    background: #fee2e2;
    border-left: 4px solid #ef4444;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}

/* Success Box */
.success-box {
    background: #d1fae5;
    border-left: 4px solid #10b981;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}

/* Metric Cards */
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #3B82F6;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin-top: 0.5rem;
}

/* Chapter List */
.chapter-list {
    background: #fafafa;
    border-radius: 6px;
    padding: 0.5rem;
}

.chapter-item {
    padding: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
    transition: background 0.2s;
}

.chapter-item:hover {
    background: #f3f4f6;
}

.chapter-item:last-child {
    border-bottom: none;
}

/* Timestamp Badge */
.timestamp-badge {
    background: #3B82F6;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 500;
}

/* Progress Bar */
.progress-bar {
    background: #e5e7eb;
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    background: linear-gradient(90deg, #3B82F6 0%, #8b5cf6 100%);
    height: 100%;
    transition: width 0.3s ease;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
}

/* Button Styling */
.stButton>button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: #f9fafb;
    border-radius: 6px;
    font-weight: 500;
}

/* Sidebar Styling */
.css-1d391kg {
    background: #f8f9fa;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.3s ease-in;
}

/* Voice Recording Indicator */
.recording-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background: #ef4444;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Loading Spinner */
.custom-spinner {
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3B82F6;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .summary-card {
        padding: 1rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
    }
}

/* Sidebar session item */
.session-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0.35rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.session-name {
    font-weight: 600;
    color: #111827;
    overflow: hidden;
    text-overflow: ellipsis;
    /* Single-line clamp for tighter sidebar rows */
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.2em;
    max-height: 1.2em; /* 1 line */
}

.session-delete {
    display: flex;
    align-items: center;
    justify-content: center;
}

.session-delete .stButton>button {
    padding: 4px 6px;
    font-size: 0.85rem;
    background: transparent;
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.15);
    border-radius: 6px;
}

.session-delete .stButton>button:hover {
    background: rgba(239,68,68,0.06);
}

/* Sidebar buttons styling to make load buttons uniform height and left-aligned text */
.css-1d391kg .stButton>button {
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-left: 12px;
    padding-right: 12px;
    text-align: left;
    white-space: normal;
}

/* Ensure session-box visual spacing when using Streamlit buttons */
.css-1d391kg .session-box {
    min-height: 48px;
}
</style>
"""


def load_css():
    """Load CSS styles into Streamlit app."""
    import streamlit as st
    st.markdown(APP_CSS, unsafe_allow_html=True)

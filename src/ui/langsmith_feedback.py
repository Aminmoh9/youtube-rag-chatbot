"""
LangSmith Feedback UI Component for Streamlit.
"""
import streamlit as st
from src.integrations.langsmith_integration import LangSmithManager


class LangSmithFeedbackUI:
    """UI component for collecting user feedback on QA responses."""
    
    def __init__(self):
        """Initialize feedback UI."""
        self.langsmith = LangSmithManager()
        self.enabled = self.langsmith.enabled
    
    def show_feedback_widget(self, trace_id: str = None, answer_text: str = ""):
        """
        Display feedback collection widget.
        
        Args:
            trace_id: LangSmith trace ID for the interaction
            answer_text: The answer text being rated
        """
        if not self.enabled or not trace_id:
            return
        
        st.markdown("---")
        st.markdown("#### 💬 Rate this answer")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            rating = st.select_slider(
                "Quality:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x,
                key=f"rating_{trace_id}"
            )
        
        with col2:
            feedback_text = st.text_area(
                "Additional feedback (optional):",
                placeholder="Tell us what worked or what could be improved...",
                height=80,
                key=f"feedback_text_{trace_id}"
            )
        
        if st.button("Submit Feedback", key=f"submit_{trace_id}", type="primary"):
            success = self.langsmith.collect_user_feedback(
                run_id=trace_id,
                rating=rating,
                feedback=feedback_text
            )
            
            if success:
                st.success("✅ Thank you for your feedback!")
            else:
                st.warning("⚠️ Could not submit feedback (LangSmith may be disabled)")
    
    def show_inline_rating(self, trace_id: str = None):
        """
        Show inline rating buttons (thumbs up/down).
        
        Args:
            trace_id: LangSmith trace ID
        """
        if not self.enabled or not trace_id:
            return
        
        col1, col2, col3 = st.columns([1, 1, 8])
        
        with col1:
            if st.button("👍", key=f"thumbs_up_{trace_id}"):
                self.langsmith.collect_user_feedback(
                    run_id=trace_id,
                    rating=5,
                    feedback="Helpful answer"
                )
                st.success("Thanks!")
        
        with col2:
            if st.button("👎", key=f"thumbs_down_{trace_id}"):
                self.langsmith.collect_user_feedback(
                    run_id=trace_id,
                    rating=1,
                    feedback="Not helpful"
                )
                st.info("Feedback recorded")
    
    def show_performance_metrics(self, days: int = 7):
        """
        Display performance metrics dashboard.
        
        Args:
            days: Number of days to show metrics for
        """
        if not self.enabled:
            st.warning("⚠️ LangSmith is not enabled")
            return
        
        st.markdown("### 📊 Performance Metrics")
        
        metrics = self.langsmith.get_performance_metrics(days=days)
        
        if not metrics:
            st.info("No metrics available yet")
            return
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries",
                metrics.get("total_runs", 0)
            )
        
        with col2:
            success_rate = metrics.get("success_rate", 0) * 100
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%"
            )
        
        with col3:
            st.metric(
                "Avg Response Time",
                f"{metrics.get('avg_latency_ms', 0):.0f}ms"
            )
        
        with col4:
            st.metric(
                "Errors",
                metrics.get("error_count", 0)
            )
        
        # Additional details
        with st.expander("📈 View Detailed Metrics"):
            st.json(metrics)


# Global instance
feedback_ui = LangSmithFeedbackUI()

"""
Complete LangSmith integration for tracing, evaluation, and monitoring.
"""
import os
from typing import Dict, Any, Optional
from langsmith import Client, RunTree, traceable
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.callbacks.manager import CallbackManager
import streamlit as st

class LangSmithManager:
    """Centralized LangSmith management."""
    
    def __init__(self, project_name: str = "youtube-ai-research"):
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = project_name
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            self.client = Client(api_key=self.api_key)
            self.tracer = LangChainTracer(
                project_name=project_name,
                client=self.client
            )
            print(f"✅ LangSmith initialized for project: {project_name}")
        else:
            print("⚠️ LangSmith disabled (no API key)")
            self.client = None
            self.tracer = None
    
    @traceable(run_type="chain", name="qa_pipeline")
    def trace_qa_pipeline(self, question: str, context: str, topic: str) -> Dict:
        """Trace the complete QA pipeline."""
        # This will automatically create a trace in LangSmith
        return {
            "question": question,
            "context_length": len(context),
            "topic": topic,
            "timestamp": st.session_state.get('timestamp')
        }
    
    def create_evaluation_dataset(self, dataset_name: str, examples: list):
        """Create evaluation dataset in LangSmith."""
        if not self.enabled:
            return None
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="YouTube QA evaluation dataset"
            )
            
            for example in examples:
                self.client.create_example(
                    inputs={
                        "question": example["question"],
                        "context": example["context"]
                    },
                    outputs={
                        "expected_answer": example["answer"],
                        "metadata": example.get("metadata", {})
                    },
                    dataset_id=dataset.id
                )
            
            print(f"✅ Created dataset: {dataset_name}")
            return dataset.id
        except Exception as e:
            print(f"⚠️ Failed to create dataset: {e}")
            return None
    
    def collect_user_feedback(self, run_id: str, rating: int, feedback: str = ""):
        """Collect user feedback for a trace."""
        if not self.enabled or not run_id:
            return
        
        try:
            self.client.create_feedback(
                run_id=run_id,
                key="user_rating",
                score=rating,  # 1-5 scale
                comment=feedback,
                source_info={
                    "via": "streamlit_app",
                    "session_id": st.session_state.get('session_id', 'unknown')
                }
            )
            print(f"✅ Feedback recorded for run: {run_id}")
        except Exception as e:
            print(f"⚠️ Failed to record feedback: {e}")
    
    def get_performance_metrics(self, days: int = 7) -> Dict:
        """Get performance metrics from LangSmith."""
        if not self.enabled:
            return {}
        
        try:
            # Get runs from last N days
            runs = self.client.list_runs(
                project_name=self.project_name,
                start_time=f"{days}d",
                limit=100
            )
            
            # Calculate metrics
            total_runs = len(runs)
            successful = sum(1 for r in runs if not r.error)
            avg_latency = sum(r.latency for r in runs if r.latency) / total_runs if total_runs else 0
            
            return {
                "total_runs": total_runs,
                "success_rate": successful / total_runs if total_runs else 0,
                "avg_latency_ms": avg_latency * 1000 if avg_latency else 0,
                "error_count": total_runs - successful
            }
        except Exception as e:
            print(f"⚠️ Failed to get metrics: {e}")
            return {}
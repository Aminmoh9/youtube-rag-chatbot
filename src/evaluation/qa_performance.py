"""
QA Performance Monitoring and Metrics.
"""
import os
from typing import Dict, List
from langsmith import Client
from datetime import datetime, timedelta
from collections import defaultdict


class QAPerformanceMonitor:
    """Monitor and track QA performance metrics."""
    
    def __init__(self, project_name: str = "youtube-ai-research"):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.project_name = project_name
    
    def monitor_qa_performance(self, days: int = 7) -> Dict:
        """Track QA performance metrics over time."""
        try:
            # Get all runs from last N days
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=f"{days}d",
                limit=500
            ))
            
            if not runs:
                return {
                    "total_queries": 0,
                    "avg_response_time_ms": 0,
                    "success_rate": 0,
                    "error_rate": 0
                }
            
            # Calculate metrics
            total = len(runs)
            successful = sum(1 for r in runs if not r.error)
            total_latency = sum(r.latency for r in runs if r.latency)
            
            metrics = {
                "total_queries": total,
                "successful_queries": successful,
                "failed_queries": total - successful,
                "success_rate": successful / total if total > 0 else 0,
                "error_rate": (total - successful) / total if total > 0 else 0,
                "avg_response_time_ms": (total_latency / total * 1000) if total > 0 else 0,
                "period_days": days
            }
            
            # Get user feedback stats
            feedback = self._get_user_feedback_stats(days)
            metrics.update(feedback)
            
            return metrics
            
        except Exception as e:
            print(f"Error monitoring performance: {e}")
            return {"error": str(e)}
    
    def _get_user_feedback_stats(self, days: int) -> Dict:
        """Get user feedback statistics."""
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=f"{days}d",
                has_feedback=True,
                limit=100
            ))
            
            if not runs:
                return {}
            
            ratings = []
            for run in runs:
                # Get feedback for this run
                feedbacks = list(self.client.list_feedback(run_ids=[run.id]))
                for feedback in feedbacks:
                    if feedback.score:
                        ratings.append(feedback.score)
            
            if ratings:
                return {
                    "user_satisfaction": sum(ratings) / len(ratings),
                    "total_feedback_count": len(ratings),
                    "feedback_rate": len(ratings) / len(runs) if runs else 0
                }
            
            return {}
            
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            return {}
    
    def get_performance_trends(self, days: int = 30) -> Dict:
        """Get performance trends over time."""
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=f"{days}d",
                limit=1000
            ))
            
            # Group by day
            daily_metrics = defaultdict(lambda: {"total": 0, "successful": 0, "latency_sum": 0})
            
            for run in runs:
                if run.start_time:
                    day = run.start_time.date().isoformat()
                    daily_metrics[day]["total"] += 1
                    if not run.error:
                        daily_metrics[day]["successful"] += 1
                    if run.latency:
                        daily_metrics[day]["latency_sum"] += run.latency
            
            trends = []
            for day, metrics in sorted(daily_metrics.items()):
                trends.append({
                    "date": day,
                    "total_queries": metrics["total"],
                    "success_rate": metrics["successful"] / metrics["total"] if metrics["total"] > 0 else 0,
                    "avg_response_time_ms": (metrics["latency_sum"] / metrics["total"] * 1000) if metrics["total"] > 0 else 0
                })
            
            return {
                "trends": trends,
                "period_days": days
            }
            
        except Exception as e:
            print(f"Error getting trends: {e}")
            return {}
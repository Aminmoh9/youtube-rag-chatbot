"""
Error Analysis for LangSmith traces and system failures.
"""
import os
from typing import Dict, List
from langsmith import Client
from collections import defaultdict
from datetime import datetime, timedelta


class ErrorAnalyzer:
    """Analyze errors and failures in the system."""
    
    def __init__(self, project_name: str = "youtube-ai-research"):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.project_name = project_name
    
    def analyze_errors(self, days: int = 7, limit: int = 100) -> Dict:
        """Analyze errors in LangSmith traces."""
        try:
            # Get failed runs
            failed_runs = list(self.client.list_runs(
                project_name=self.project_name,
                error=True,
                start_time=f"{days}d",
                limit=limit
            ))
            
            error_patterns = defaultdict(int)
            error_details = []
            
            for run in failed_runs:
                error_type = run.error.split(":")[0] if run.error else "Unknown"
                error_patterns[error_type] += 1
                
                error_details.append({
                    "run_id": str(run.id),
                    "error_type": error_type,
                    "error_message": run.error,
                    "timestamp": run.start_time.isoformat() if run.start_time else None,
                    "inputs": run.inputs
                })
            
            print("📊 Error Analysis:")
            print(f"Total failed runs: {len(failed_runs)}")
            print("\nError patterns:")
            for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences ({count/len(failed_runs)*100:.1f}%)")
            
            return {
                "total_errors": len(failed_runs),
                "error_patterns": dict(error_patterns),
                "error_details": error_details,
                "analysis_period_days": days
            }
            
        except Exception as e:
            print(f"Error analyzing errors: {e}")
            return {
                "total_errors": 0,
                "error": str(e)
            }
    
    def get_error_trends(self, days: int = 30) -> Dict:
        """Get error trends over time."""
        try:
            failed_runs = list(self.client.list_runs(
                project_name=self.project_name,
                error=True,
                start_time=f"{days}d",
                limit=500
            ))
            
            # Group by day
            daily_errors = defaultdict(int)
            for run in failed_runs:
                if run.start_time:
                    day = run.start_time.date().isoformat()
                    daily_errors[day] += 1
            
            return {
                "daily_errors": dict(daily_errors),
                "total_days": days,
                "avg_errors_per_day": len(failed_runs) / days
            }
            
        except Exception as e:
            print(f"Error getting trends: {e}")
            return {}
    
    def identify_common_failures(self, limit: int = 50) -> List[Dict]:
        """Identify most common failure points."""
        try:
            failed_runs = list(self.client.list_runs(
                project_name=self.project_name,
                error=True,
                limit=limit
            ))
            
            failure_points = defaultdict(int)
            
            for run in failed_runs:
                # Analyze where in the pipeline it failed
                if "embedding" in str(run.error).lower():
                    failure_points["Embedding Generation"] += 1
                elif "pinecone" in str(run.error).lower():
                    failure_points["Vector Store"] += 1
                elif "openai" in str(run.error).lower() or "llm" in str(run.error).lower():
                    failure_points["LLM Generation"] += 1
                elif "youtube" in str(run.error).lower():
                    failure_points["YouTube Integration"] += 1
                else:
                    failure_points["Other"] += 1
            
            return [
                {"component": component, "count": count}
                for component, count in sorted(failure_points.items(), key=lambda x: x[1], reverse=True)
            ]
            
        except Exception as e:
            print(f"Error identifying failures: {e}")
            return []
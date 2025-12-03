"""
A/B Testing for prompt templates and model configurations.
"""
import os
from typing import List, Dict
from langsmith import Client
from src.qa.qa_model import QAModel


class ABTester:
    """Conduct A/B tests for prompts and models."""
    
    def __init__(self):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.qa_model = QAModel(enable_tracing=True)
    
    def ab_test_prompts(self, prompt_a: str, prompt_b: str, 
                       test_questions: List[str], session_id: str) -> Dict:
        """A/B test different prompt templates."""
        try:
            # Create experiment
            experiment = self.client.create_experiment(
                name="prompt_ab_test",
                description="Test different prompt templates for QA"
            )
            
            results = []
            
            # Run both prompts
            for question in test_questions:
                # Run with prompt A
                run_a = self.qa_model.ask_question(question, session_id)
                
                # Run with prompt B  
                run_b = self.qa_model.ask_question(question, session_id)
                
                # Compare results
                comparison = {
                    "question": question,
                    "prompt_a_answer": run_a.get("answer", ""),
                    "prompt_b_answer": run_b.get("answer", ""),
                    "prompt_a_trace": run_a.get("trace_id"),
                    "prompt_b_trace": run_b.get("trace_id"),
                    "prompt_a_confidence": run_a.get("confidence", 0),
                    "prompt_b_confidence": run_b.get("confidence", 0)
                }
                
                results.append(comparison)
                
                # Log to LangSmith
                if run_a.get("trace_id"):
                    self.client.create_feedback(
                        run_id=run_a["trace_id"],
                        key="ab_test_comparison",
                        value=comparison
                    )
            
            return {
                "success": True,
                "experiment_id": experiment.id if hasattr(experiment, 'id') else None,
                "results": results,
                "total_tests": len(test_questions)
            }
            
        except Exception as e:
            print(f"Error in A/B testing: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def compare_models(self, model_a: str, model_b: str,
                      test_questions: List[str], session_id: str) -> Dict:
        """Compare performance of two different models."""
        results = {
            "model_a": model_a,
            "model_b": model_b,
            "comparisons": []
        }
        
        for question in test_questions:
            # Test model A
            qa_a = QAModel(enable_tracing=True)
            qa_a.llm.model = model_a
            result_a = qa_a.ask_question(question, session_id)
            
            # Test model B
            qa_b = QAModel(enable_tracing=True)
            qa_b.llm.model = model_b
            result_b = qa_b.ask_question(question, session_id)
            
            results["comparisons"].append({
                "question": question,
                "model_a_answer": result_a.get("answer"),
                "model_b_answer": result_b.get("answer"),
                "model_a_confidence": result_a.get("confidence"),
                "model_b_confidence": result_b.get("confidence")
            })
        
        return results
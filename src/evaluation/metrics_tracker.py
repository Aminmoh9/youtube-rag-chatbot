"""
Real-time Performance Metrics Tracker.
Measures relevance, accuracy, and latency for each QA interaction.
"""
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class PerformanceMetrics:
    """Track and analyze QA performance metrics."""
    
    def __init__(self, metrics_file: str = "data/metrics/qa_metrics.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_qa_interaction(self, 
                          question: str,
                          answer: str,
                          sources: List[Dict],
                          latency_ms: float,
                          session_id: str,
                          relevance_scores: Optional[List[float]] = None) -> Dict:
        """
        Log a single QA interaction with performance metrics.
        
        Args:
            question: User's question
            answer: Generated answer
            sources: Retrieved source chunks with scores
            latency_ms: Response time in milliseconds
            session_id: Session identifier
            relevance_scores: Pinecone similarity scores for retrieved chunks
            
        Returns:
            Computed metrics dictionary
        """
        # Calculate metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'question': question,
            'answer_length': len(answer),
            'num_sources': len(sources),
            'latency_ms': latency_ms,
            'latency_category': self._categorize_latency(latency_ms),
            
            # Relevance metrics (from Pinecone similarity scores)
            'avg_relevance_score': np.mean(relevance_scores) if relevance_scores else None,
            'min_relevance_score': min(relevance_scores) if relevance_scores else None,
            'max_relevance_score': max(relevance_scores) if relevance_scores else None,
            'relevance_quality': self._categorize_relevance(relevance_scores) if relevance_scores else 'unknown',
            
            # Context quality
            'context_coverage': self._calculate_context_coverage(sources),
            'source_diversity': self._calculate_source_diversity(sources),
        }
        
        # Save to log file
        self._append_to_log(metrics)
        
        return metrics
    
    def _categorize_latency(self, latency_ms: float) -> str:
        """Categorize latency into performance tiers."""
        if latency_ms < 1000:
            return 'excellent'  # <1s
        elif latency_ms < 3000:
            return 'good'  # 1-3s
        elif latency_ms < 5000:
            return 'acceptable'  # 3-5s
        else:
            return 'slow'  # >5s
    
    def _categorize_relevance(self, scores: List[float]) -> str:
        """Categorize relevance quality based on similarity scores."""
        if not scores:
            return 'unknown'
        
        avg_score = np.mean(scores)
        
        # Pinecone cosine similarity: 1.0 = identical, 0.0 = unrelated
        if avg_score > 0.85:
            return 'excellent'
        elif avg_score > 0.75:
            return 'good'
        elif avg_score > 0.65:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_context_coverage(self, sources: List[Dict]) -> float:
        """Calculate how much context was provided."""
        total_chars = sum(len(src.get('text', '')) for src in sources)
        # Normalize to 0-1 scale (assume 3000 chars is "full coverage")
        return min(total_chars / 3000.0, 1.0)
    
    def _calculate_source_diversity(self, sources: List[Dict]) -> float:
        """Calculate diversity of sources (different videos/documents)."""
        if not sources:
            return 0.0
        
        unique_sources = len(set(src.get('metadata', {}).get('video_id', 'unknown') for src in sources))
        return unique_sources / len(sources)
    
    def _append_to_log(self, metrics: Dict):
        """Append metrics to JSONL log file."""
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Get recent metrics from log."""
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except:
                    continue
        
        return metrics[-limit:]
    
    def calculate_aggregate_metrics(self, metrics: List[Dict] = None) -> Dict:
        """Calculate aggregate performance metrics."""
        if metrics is None:
            metrics = self.get_recent_metrics()
        
        if not metrics:
            return {
                'error': 'No metrics data available',
                'total_interactions': 0
            }
        
        latencies = [m['latency_ms'] for m in metrics if 'latency_ms' in m]
        relevance_scores = [m['avg_relevance_score'] for m in metrics if m.get('avg_relevance_score')]
        
        return {
            'total_interactions': len(metrics),
            
            # Latency stats
            'latency': {
                'avg_ms': np.mean(latencies) if latencies else 0,
                'median_ms': np.median(latencies) if latencies else 0,
                'p95_ms': np.percentile(latencies, 95) if latencies else 0,
                'max_ms': max(latencies) if latencies else 0,
                'excellent_pct': sum(1 for m in metrics if m.get('latency_category') == 'excellent') / len(metrics) * 100,
                'good_pct': sum(1 for m in metrics if m.get('latency_category') == 'good') / len(metrics) * 100,
            },
            
            # Relevance stats
            'relevance': {
                'avg_score': np.mean(relevance_scores) if relevance_scores else 0,
                'min_score': min(relevance_scores) if relevance_scores else 0,
                'max_score': max(relevance_scores) if relevance_scores else 0,
                'excellent_pct': sum(1 for m in metrics if m.get('relevance_quality') == 'excellent') / len(metrics) * 100,
                'good_pct': sum(1 for m in metrics if m.get('relevance_quality') == 'good') / len(metrics) * 100,
                'fair_pct': sum(1 for m in metrics if m.get('relevance_quality') == 'fair') / len(metrics) * 100,
                'poor_pct': sum(1 for m in metrics if m.get('relevance_quality') == 'poor') / len(metrics) * 100,
            },
            
            # Context quality
            'context': {
                'avg_coverage': np.mean([m.get('context_coverage', 0) for m in metrics]),
                'avg_source_diversity': np.mean([m.get('source_diversity', 0) for m in metrics]),
                'avg_sources_per_query': np.mean([m.get('num_sources', 0) for m in metrics]),
            }
        }


class QAMetricsWrapper:
    """Wrapper to add metrics tracking to QA operations."""
    
    def __init__(self, qa_model):
        self.qa_model = qa_model
        self.metrics = PerformanceMetrics()
    
    def ask_question_with_metrics(self, question: str, session_id: str, 
                                   namespace: str = None, top_k: int = 3) -> Dict:
        """
        Ask question and track performance metrics.
        
        Args:
            question: User's question
            session_id: Session identifier
            namespace: Pinecone namespace
            top_k: Number of sources to retrieve
            
        Returns:
            QA result with added metrics
        """
        # Start timing
        start_time = time.time()
        
        # Get answer
        result = self.qa_model.ask_question(question, session_id, namespace, top_k)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract relevance scores from sources
        relevance_scores = []
        if result.get('sources'):
            relevance_scores = [src.get('score', 0) for src in result['sources']]
        
        # Log metrics
        metrics = self.metrics.log_qa_interaction(
            question=question,
            answer=result.get('answer', ''),
            sources=result.get('sources', []),
            latency_ms=latency_ms,
            session_id=session_id,
            relevance_scores=relevance_scores
        )
        
        # Add metrics to result
        result['metrics'] = metrics
        
        return result
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        return self.metrics.calculate_aggregate_metrics()


# Global metrics instance
_metrics_tracker = PerformanceMetrics()

def get_metrics_tracker() -> PerformanceMetrics:
    """Get global metrics tracker instance."""
    return _metrics_tracker

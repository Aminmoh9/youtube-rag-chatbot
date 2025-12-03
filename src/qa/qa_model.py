"""
Main QA Model - Question answering with LangSmith tracing integration.
"""
import os
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from src.embeddings.pinecone_topic_isolation import StrictTopicIsolation


class QAModel:
    """Main question answering model with context retrieval."""
    
    def __init__(self, enable_tracing: bool = True):
        """
        Initialize QA Model.
        
        Args:
            enable_tracing: Enable LangSmith tracing
        """
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=0.3,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.isolation_manager = StrictTopicIsolation()
        self.enable_tracing = enable_tracing and bool(os.getenv("LANGSMITH_API_KEY"))
        
        if self.enable_tracing:
            from src.integrations.langsmith_integration import LangSmithManager
            self.langsmith = LangSmithManager()
        else:
            self.langsmith = None
    
    def ask_question(self, question: str, session_id: str, 
                    topic: str = None, top_k: int = 3, namespace: str = None) -> Dict:
        """
        Answer a question using retrieved context.
        
        Args:
            question: User's question
            session_id: Session identifier
            topic: Topic for isolation (optional)
            top_k: Number of context chunks to retrieve
            namespace: Pre-computed namespace (optional, overrides topic-based computation)
            (filters support removed) Optional Pinecone metadata filter dict
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant context
        context_chunks = self._retrieve_context(
            question=question,
            session_id=session_id,
            topic=topic,
            top_k=top_k,
            namespace=namespace
        )
        
        if not context_chunks:
            return {
                'success': False,
                'answer': "I couldn't find relevant information to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Format context
        context_text = self._format_context(context_chunks)
        
        # Generate answer
        if self.enable_tracing and self.langsmith:
            answer_result = self._generate_answer_with_tracing(
                question=question,
                context=context_text,
                topic=topic
            )
        else:
            answer_result = self._generate_answer(
                question=question,
                context=context_text
            )
        
        return {
            'success': True,
            'answer': answer_result['answer'],
            'sources': self._extract_sources(context_chunks),
            'context_chunks': len(context_chunks),
            'confidence': answer_result.get('confidence', 0.8),
            'trace_id': answer_result.get('trace_id')
        }
    
    def _retrieve_context(self, question: str, session_id: str, 
                         topic: str, top_k: int, namespace: str = None) -> List[Dict]:
        """Retrieve relevant context chunks from vector store."""
        try:
            # Query Pinecone with topic isolation using the question text directly
            results = self.isolation_manager.query_with_isolation(
                query_text=question,
                topic=topic,
                top_k=top_k,
                namespace=namespace
            )
            
            return results
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format context chunks into a single string."""
        formatted = []
        
        # Check if this is YouTube content with timestamps
        has_youtube_timestamps = any(
            chunk.get('metadata', {}).get('video_url') and 
            chunk.get('metadata', {}).get('timestamp') is not None 
            for chunk in chunks
        )
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', '')
            
            formatted.append(f"\n--- Source {i} ---")
            
            # Add YouTube-specific info
            if metadata.get('video_url'):
                video_url = metadata['video_url']
                formatted.append(f"Video: {video_url}")
                
                if metadata.get('timestamp') is not None:
                    timestamp = int(metadata['timestamp'])
                    time_str = self._format_timestamp(timestamp)
                    timestamp_url = f"{video_url}&t={timestamp}s"
                    formatted.append(f"Timestamp: {time_str} - {timestamp_url}")
            
            if metadata.get('chunk_index') is not None:
                formatted.append(f"Chunk {metadata['chunk_index']} of {metadata.get('total_chunks', '?')}")
            
            formatted.append(f"\nContent:\n{text}\n")
        
        return "\n".join(formatted)
    
    def _generate_answer(self, question: str, context: str) -> Dict:
        """Generate answer using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant that answers questions based on provided context.
            
Instructions:
- Answer the question using ONLY the information from the provided context
- If the context doesn't contain relevant information, say so
- Be concise but comprehensive
- Avoid including citations like [Source 1] or [Source 2] in your answer
- Avoid mentioning source numbers at all
- Just provide a natural, flowing answer without any bracketed references
- If you're uncertain, express that in your answer
- If the question asks for numbered points, bullet points, or multiple items, format each point on a NEW LINE with a blank line between items
- Use proper markdown formatting: 
  * For main points: Use numbers (1., 2., 3.) followed by a blank line
  * For sub-points: Use bullet points (- or •) indented under the main point
  * Maintain consistent structure across all items"""),
            HumanMessage(content=f"""Context:
{context}

Question: {question}

Answer (without any [Source N] citations):""")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            
            return {
                'answer': response.content,
                'confidence': 0.8
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': "I encountered an error generating the answer.",
                'confidence': 0.0
            }
    
    def _generate_answer_streaming(self, question: str, context: str):
        """Generate answer using LLM with streaming (yields tokens)."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant that answers questions based on provided context.
            
Instructions:
- Answer the question using ONLY the information from the provided context
- If the context doesn't contain relevant information, say so
- Be concise but comprehensive
- Avoid including citations like [Source 1] or [Source 2] in your answer
- Avoid mentioning source numbers at all
- Just provide a natural, flowing answer without any bracketed references
- If you're uncertain, express that in your answer
- If the question asks for numbered points, bullet points, or multiple items, format each point on a NEW LINE with a blank line between items
- Use proper markdown formatting: 
  * For main points: Use numbers (1., 2., 3.) followed by a blank line
  * For sub-points: Use bullet points (- or •) indented under the main point
  * Maintain consistent structure across all items"""),
            HumanMessage(content=f"""Context:
{context}

Question: {question}

Answer (without any [Source N] citations):""")
        ])
        
        try:
            for chunk in self.llm.stream(prompt.format_messages()):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"Error generating streaming answer: {e}")
            yield "I encountered an error generating the answer."
    
    def _format_timestamp(self, seconds: int) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _generate_answer_with_tracing(self, question: str, 
                                     context: str, topic: str) -> Dict:
        """Generate answer with LangSmith tracing."""
        try:
            # Trace the QA pipeline
            trace_data = self.langsmith.trace_qa_pipeline(
                question=question,
                context=context,
                topic=topic or "unknown"
            )
            
            # Generate answer
            answer_result = self._generate_answer(question, context)
            answer_result['trace_id'] = trace_data.get('trace_id')
            
            return answer_result
            
        except Exception as e:
            print(f"Error in traced answer generation: {e}")
            # Fallback to non-traced
            return self._generate_answer(question, context)
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks."""
        sources = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            source = {
                'source_id': metadata.get('source_id', 'Unknown'),
                'title': metadata.get('title', 'Untitled'),
                'url': metadata.get('url', ''),
                'chunk_preview': chunk.get('text', '')[:200] + '...',
                'score': chunk.get('score', 0.0),
                'metadata': metadata  # Include full metadata for timestamps
            }
            
            sources.append(source)
        
        return sources
    
    def ask_question_stream(self, question: str, session_id: str,
                           topic: str = None, top_k: int = 3, namespace: str = None):
        """
        Answer a question with streaming response (yields tokens).
        
        Args:
            question: User's question
            session_id: Session identifier
            topic: Topic for isolation (optional)
            top_k: Number of context chunks to retrieve
            namespace: Pre-computed namespace (optional)
            
        Yields:
            Tokens as they are generated, then final dict with complete metadata
        """
        # Retrieve relevant context
        context_chunks = self._retrieve_context(
            question=question,
            session_id=session_id,
            topic=topic,
            top_k=top_k,
            namespace=namespace
        )
        
        if not context_chunks:
            yield {
                'type': 'error',
                'message': "I couldn't find relevant information to answer this question."
            }
            return
        
        # Format context
        context_text = self._format_context(context_chunks)
        
        # Stream the answer
        full_answer = ""
        for token in self._generate_answer_streaming(question, context_text):
            full_answer += token
            yield {'type': 'token', 'content': token}
        
        # Yield final metadata
        yield {
            'type': 'complete',
            'success': True,
            'answer': full_answer,
            'sources': self._extract_sources(context_chunks),
            'context_chunks': len(context_chunks),
            'confidence': 0.8
        }
    
    def ask_with_feedback(self, question: str, session_id: str,
                         topic: str = None, namespace: str = None) -> Dict:
        """
        Ask question and prepare for user feedback collection.
        
        Args:
            question: User's question
            session_id: Session identifier
            topic: Topic for isolation
            namespace: Pre-computed namespace (optional)
            
        Returns:
            Answer with trace_id for feedback
        """
        result = self.ask_question(question, session_id, topic, namespace=namespace)
        
        # Return with trace_id for feedback collection
        if self.enable_tracing and result.get('trace_id'):
            result['can_provide_feedback'] = True
        else:
            result['can_provide_feedback'] = False
        
        return result
    
    def submit_feedback(self, trace_id: str, rating: int, 
                       feedback: str = "") -> bool:
        """
        Submit user feedback for a QA interaction.
        
        Args:
            trace_id: LangSmith trace ID
            rating: User rating (1-5)
            feedback: Optional feedback text
            
        Returns:
            Success status
        """
        if not self.enable_tracing or not self.langsmith:
            return False
        
        try:
            self.langsmith.collect_user_feedback(
                run_id=trace_id,
                rating=rating,
                feedback=feedback
            )
            return True
            
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            return False


class MultiSourceQA(QAModel):
    """Extended QA model for handling multiple content sources."""
    
    def ask_across_sources(self, question: str, session_ids: List[str],
                          merge_strategy: str = "ranked") -> Dict:
        """
        Answer question using multiple content sources.
        
        Args:
            question: User's question
            session_ids: List of session IDs to query
            merge_strategy: How to merge results ("ranked" or "concat")
            
        Returns:
            Combined answer from multiple sources
        """
        all_chunks = []
        
        # Retrieve from all sessions
        for session_id in session_ids:
            chunks = self._retrieve_context(
                question=question,
                session_id=session_id,
                topic=None,
                top_k=3,
                namespace=None
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {
                'success': False,
                'answer': "No relevant information found across sources.",
                'sources': []
            }
        
        # Sort by score and take top K
        all_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_chunks = all_chunks[:5]
        
        # Format and generate answer
        context_text = self._format_context(top_chunks)
        answer_result = self._generate_answer(question, context_text)
        
        return {
            'success': True,
            'answer': answer_result['answer'],
            'sources': self._extract_sources(top_chunks),
            'source_sessions': session_ids,
            'total_chunks_considered': len(all_chunks)
        }

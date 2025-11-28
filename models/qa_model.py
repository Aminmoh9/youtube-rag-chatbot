"""
QA Model for Multimodal Chatbot
Orchestrates the QA system using OpenAI GPT and Pinecone vector search.
Searches Pinecone for relevant chunks and generates answers with GPT-3.5-turbo.
"""

import os
from typing import Dict, List
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Pinecone search tool
from embeddings.pinecone_utils import query_pinecone

def pinecone_search_tool(query: str, topic: str = None) -> str:
    results = query_pinecone(query, top_k=5, topic=topic)
    if not results:
        return "No relevant sources found."
    context = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        context.append(f"[{i}] {meta.get('text', '')}")
    return "\n---\n".join(context)

pinecone_tool = Tool(
    name="PineconeSearch",
    func=pinecone_search_tool,
    description="Searches Pinecone for relevant video transcript chunks."
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
tools = [pinecone_tool]

class MultimodalQAAgent:
    """
    QA system for multimodal chatbot.
    Handles video transcript search and answer generation.
    """

    def __init__(self, temperature: float = 0.3, top_p: float = 1.0, max_tokens: int = 256, similarity_threshold: float = 0.7):
        """
        Initialize the QA agent.

        Args:
            temperature (float): GPT temperature (0-1, lower = more focused)
            top_p (float): Nucleus sampling parameter
            max_tokens (int): Maximum tokens for response
            similarity_threshold (float): Minimum similarity score for retrieval
        """
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.chat_history = []
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
    
    def _search_videos(self, query: str, topic: str = None) -> List[Dict]:
        """
        Search video transcripts in Pinecone.
        
        Args:
            query (str): User's question
            topic (str): Optional topic filter
            
        Returns:
            list: Retrieved chunks with metadata
        """
        try:
            results = query_pinecone(query, top_k=5, topic=topic)
            
            if not results or len(results) == 0:
                return []
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results 
                if r.get('score', 0) >= self.similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            print(f"Error searching videos: {str(e)}")
            return []
    
    def _format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            results (list): Retrieved chunks
            
        Returns:
            str: Formatted context
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            context_parts.append(
                f"[Source {i}] (Similarity: {score:.2f})\n"
                f"Video: {metadata.get('title', 'Unknown')}\n"
                f"Topic: {metadata.get('topic', 'General')}\n"
                f"Timestamp: {metadata.get('start_time', 0)}-{metadata.get('end_time', 0)}s\n"
                f"URL: {metadata.get('url', '')}\n"
                f"Content: {metadata.get('text', '')}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def ask(self, question: str, topic: str = None) -> Dict[str, any]:
        """
        Main method to ask a question.
        
        Args:
            question (str): User's question
            topic (str): Optional topic filter (SQL, Python, Excel, Tableau, Power BI)
            
        Returns:
            dict: Answer and metadata
        """
        try:
            # Search for relevant chunks
            results = self._search_videos(question, topic=topic)
            
            # Check if we have relevant information
            if not results:
                topic_msg = f" about {topic}" if topic else ""
                return {
                    "question": question,
                    "answer": (
                        f"I don't have enough information in my knowledge base to answer that question{topic_msg}. "
                        "I can only answer questions about SQL, Python, Excel, Tableau, and Power BI "
                        "based on the tutorial videos in my database."
                    ),
                    "has_sources": False,
                    "sources": []
                }
            
            # Format context
            context = self._format_context(results)
            
            # Create prompt with context
            topic_note = f"\n\nNote: The user specifically asked about {topic} topics. If this question is actually about a different tool (like Power BI, Tableau, Excel when filtered for Python, or vice versa), politely say you can only provide information about {topic} from your knowledge base." if topic else ""
            
            prompt = f"""You are a helpful Data Analytics tutor assistant. Answer the question based ONLY on the provided context from video transcripts.

Context from videos:
{context}

Question: {question}{topic_note}

Instructions:
- Answer clearly and concisely using only the provided context.
- Teach the concept directly, without referencing video titles, timestamps, or source numbers.
- Avoid saying things like "According to Source 1" or "The video explains..."
- If the question asks for a definition of a tool or concept (e.g., 'What is Power BI?') and the context does not contain a definition, say: 'The context does not contain a definition of Power BI. Would you like a general overview?'
- If the question is about a different topic than the filter, politely decline.
- Focus on data analytics topics (SQL, Python, Excel, Tableau, Power BI).
- Base your answer strictly on the context; avoid adding information not present.
- Use a conversational and easy-to-understand tone.

Answer:"""
            
            # Generate answer with GPT
            response = self.llm.invoke([
                {"role": "system", "content": "You are a data analytics tutor assistant. You must ONLY answer based on the provided context. Never use your general knowledge. If the context doesn't contain the answer, say you don't know."},
                {"role": "user", "content": prompt}
            ])

            answer = response.content

            # Store in chat history
            self.chat_history.append({
                "question": question,
                "answer": answer
            })

            return {
                "question": question,
                "answer": answer,
                "has_sources": True,
                "sources": results
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "has_sources": False,
                "sources": []
            }



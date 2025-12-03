"""
QA agent specifically for user-uploaded content.
Handles temporary storage and processing of user files.
"""
from typing import Dict
from datetime import datetime
from pathlib import Path
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS  # Local vector store for temporary content
from langchain_text_splitters import RecursiveCharacterTextSplitter


class UploadedContentAgent:
    """Agent for handling user-uploaded content."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Store for active sessions
        self.active_sessions = {}
    
    def process_uploaded_content(self, content_text: str, session_id: str) -> Dict:
        """
        Process uploaded content, create vector store, and generate summary.
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(content_text)

            # Create vector store in memory (temporary)
            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )

            # Create conversation chain
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                verbose=False
            )

            # Generate summary and key concepts
            summary_result = self.summarize_content(session_id)
            summary = summary_result.get("summary", "") if summary_result.get("success") else ""
            key_concepts = summary_result.get("key_concepts", "") if summary_result.get("success") else ""

            # Store in active sessions
            self.active_sessions[session_id] = {
                "vector_store": vector_store,
                "qa_chain": qa_chain,
                "chunks": chunks,
                "created_at": datetime.now(),
                "content_preview": content_text[:500],
                "summary": summary,
                "key_concepts": key_concepts
            }

            return {
                "success": True,
                "session_id": session_id,
                "chunks_count": len(chunks),
                "content_length": len(content_text),
                "summary": summary,
                "key_concepts": key_concepts
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def ask_question(self, session_id: str, question: str) -> Dict:
        """
        Ask a question about uploaded content.
        
        Args:
            session_id: Session ID
            question: User's question
            
        Returns:
            Dict with answer and sources
        """
        if session_id not in self.active_sessions:
            return {
                "success": False,
                "error": "Session not found or expired"
            }
        
        try:
            qa_chain = self.active_sessions[session_id]["qa_chain"]
            
            # Get answer
            result = qa_chain({"question": question})
            answer = result["answer"]
            
            # Get relevant sources
            vector_store = self.active_sessions[session_id]["vector_store"]
            docs = vector_store.similarity_search(question, k=2)
            
            sources = []
            for i, doc in enumerate(docs):
                sources.append({
                    "text": doc.page_content[:200] + "...",
                    "relevance_score": "High" if i == 0 else "Medium"
                })
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def summarize_content(self, session_id: str) -> Dict:
        """
        Generate summary of uploaded content.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict with summary and key points
        """
        if session_id not in self.active_sessions:
            return {
                "success": False,
                "error": "Session not found or expired"
            }
        
        try:
            content_text = self.active_sessions[session_id]["chunks"]
            full_text = " ".join(content_text)
            
            # Generate summary using LLM
            summary_prompt = f"""
            Please provide a comprehensive summary of the following content:
            
            {full_text[:3000]}  # Limit text length
            
            Provide:
            1. A concise 2-3 sentence overview
            2. 3-5 key points or takeaways
            3. Main topics covered
            
            Format as a structured response.
            """
            
            response = self.llm.invoke(summary_prompt)
            
            # Extract key concepts
            concepts_prompt = f"""
            Extract 5-7 key concepts, terms, or topics from this content:
            
            {full_text[:2000]}
            
            List each with a brief explanation.
            """
            
            concepts_response = self.llm.invoke(concepts_prompt)
            
            return {
                "success": True,
                "summary": response.content,
                "key_concepts": concepts_response.content,
                "content_length": len(full_text)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def compare_with_video(self, session_id: str, video_id: str) -> Dict:
        """
        Compare uploaded content with a video in the database.
        
        Args:
            session_id: Session ID
            video_id: YouTube video ID
            
        Returns:
            Dict with comparison results
        """
        # Use the provided session_id and video_id to produce a simple comparison.
        # If we have uploaded content in `active_sessions` and a transcript for the
        # specified video in `data/transcripts/{video_id}_transcript.txt`, compute a
        # lightweight Jaccard similarity over token sets and return shared words.
        
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        # Build full text for uploaded content
        chunks = self.active_sessions[session_id].get("chunks", [])
        uploaded_text = " ".join(chunks)

        # Attempt to load video transcript (if present)
        transcripts_dir = Path("data/transcripts")
        transcript_path = transcripts_dir / f"{video_id}_transcript.txt"
        transcript_text = None
        if transcript_path.exists():
            try:
                transcript_text = transcript_path.read_text(encoding="utf-8")
            except Exception:
                transcript_text = None

        def _tokens(s: str):
            s = (s or "").lower()
            # simple tokenization: words of length >= 3
            toks = re.findall(r"[a-z0-9]{3,}", s)
            return toks

        uploaded_tokens = _tokens(uploaded_text)
        uploaded_set = set(uploaded_tokens)

        result = {
            "success": True,
            "session_id": session_id,
            "video_id": video_id,
            "uploaded_preview": uploaded_text[:400],
            "transcript_found": bool(transcript_text)
        }

        if transcript_text:
            video_tokens = _tokens(transcript_text)
            video_set = set(video_tokens)

            # Jaccard similarity
            union = uploaded_set.union(video_set)
            inter = uploaded_set.intersection(video_set)
            jaccard = (len(inter) / len(union)) if union else 0.0

            # Top shared words by frequency in uploaded text
            from collections import Counter
            shared_counts = Counter([w for w in uploaded_tokens if w in video_set])
            top_shared = [w for w, _ in shared_counts.most_common(10)]

            result.update({
                "jaccard_similarity": round(jaccard, 3),
                "shared_words": top_shared,
                "uploaded_tokens_count": len(uploaded_tokens),
                "video_tokens_count": len(video_tokens)
            })
        else:
            result.update({
                "message": "No transcript available for the given video_id; only uploaded content preview returned."
            })

        return result
    
    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def cleanup_old_sessions(self, hours_old: int = 24):
        """Clean up sessions older than specified hours."""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            age = current_time - session_data["created_at"]
            if age.total_seconds() > hours_old * 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.cleanup_session(session_id)
        
        return len(sessions_to_remove)


# Global instance
uploaded_content_agent = UploadedContentAgent()


def get_session_id():
    """Generate a unique session ID."""
    import uuid
    return str(uuid.uuid4())[:8]
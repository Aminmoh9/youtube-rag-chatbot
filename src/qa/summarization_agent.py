"""
Summarization Agent for generating summaries of video content.
"""
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os


class SummarizationAgent:
    """Agent for generating various types of summaries from video transcripts."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Initialize the summarization agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Model temperature for generation
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Summary prompt templates
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at summarizing video content. 
            Create a clear, concise summary that captures the main points and key insights.
            Format your response with:
            - Main Topic
            - Key Points (bullet points)
            - Main Takeaways"""),
            ("user", "Summarize this video transcript:\n\n{transcript}")
        ])
        
        self.detailed_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating detailed summaries. 
            Provide a comprehensive summary with:
            - Introduction
            - Main sections with detailed explanations
            - Key concepts and terminology
            - Conclusions and implications
            - Practical applications (if relevant)"""),
            ("user", "Create a detailed summary of this video transcript:\n\n{transcript}")
        ])
        
        self.bullet_points_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the main points from the content as clear, concise bullet points.
            Each bullet should be self-contained and capture a key idea or fact.
            Organize by importance or chronological order as appropriate."""),
            ("user", "Extract key points from this transcript:\n\n{transcript}")
        ])
        
        self.executive_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create an executive summary suitable for busy professionals.
            Include:
            - 2-3 sentence overview
            - Top 3-5 key findings
            - Main recommendation or conclusion
            Keep it under 200 words."""),
            ("user", "Create an executive summary of this transcript:\n\n{transcript}")
        ])
    
    def summarize(
        self, 
        transcript: str, 
        summary_type: str = "standard",
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary of the provided transcript.
        
        Args:
            transcript: Video transcript text
            summary_type: Type of summary ('standard', 'detailed', 'bullets', 'executive')
            max_length: Maximum length for the summary (optional)
            
        Returns:
            Dictionary containing summary and metadata
        """
        # Select appropriate prompt based on type
        if summary_type == "detailed":
            prompt = self.detailed_summary_prompt
        elif summary_type == "bullets":
            prompt = self.bullet_points_prompt
        elif summary_type == "executive":
            prompt = self.executive_summary_prompt
        else:
            prompt = self.summary_prompt
        
        # Create chain
        chain = prompt | self.llm
        
        # Truncate transcript if needed
        if max_length and len(transcript) > max_length:
            transcript = transcript[:max_length] + "..."
        
        # Generate summary
        try:
            response = chain.invoke({"transcript": transcript})
            summary = response.content
            
            return {
                "summary": summary,
                "type": summary_type,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "summary": "",
                "type": summary_type,
                "success": False,
                "error": str(e)
            }
    
    def multi_document_summary(
        self, 
        transcripts: List[str],
        video_titles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a unified summary from multiple video transcripts.
        
        Args:
            transcripts: List of video transcripts
            video_titles: Optional list of video titles
            
        Returns:
            Dictionary containing consolidated summary
        """
        # Combine transcripts with titles if provided
        combined_content = []
        for i, transcript in enumerate(transcripts):
            if video_titles and i < len(video_titles):
                combined_content.append(f"Video {i+1}: {video_titles[i]}\n{transcript}")
            else:
                combined_content.append(f"Video {i+1}:\n{transcript}")
        
        full_transcript = "\n\n---\n\n".join(combined_content)
        
        # Use detailed summary for multi-document
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing multiple video transcripts. 
            Create a comprehensive summary that:
            - Identifies common themes across videos
            - Highlights unique insights from each video
            - Shows how the videos relate to each other
            - Synthesizes the information into a coherent narrative"""),
            ("user", "Analyze and summarize these video transcripts:\n\n{transcript}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"transcript": full_transcript})
            summary = response.content
            
            return {
                "summary": summary,
                "type": "multi_document",
                "video_count": len(transcripts),
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "summary": "",
                "type": "multi_document",
                "video_count": len(transcripts),
                "success": False,
                "error": str(e)
            }
    
    def extract_topics(self, transcript: str) -> Dict[str, Any]:
        """
        Extract main topics and themes from a transcript.
        
        Args:
            transcript: Video transcript text
            
        Returns:
            Dictionary containing topics and their descriptions
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the main topics and themes from the content.
            For each topic, provide:
            - Topic name
            - Brief description
            - Key points related to that topic
            Format as a clear, organized list."""),
            ("user", "Extract topics from this transcript:\n\n{transcript}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"transcript": transcript})
            topics = response.content
            
            return {
                "topics": topics,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "topics": "",
                "success": False,
                "error": str(e)
            }
    
    def summarize_video(self, video_id: str, transcript_path: str) -> Dict[str, Any]:
        """
        Generate a summary for a single video from its transcript file.
        
        Args:
            video_id: YouTube video ID
            transcript_path: Path to transcript file
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Read transcript
            from pathlib import Path
            transcript_path = Path(transcript_path)

            if not transcript_path.exists():
                return {
                    "success": False,
                    "error": f"Transcript file not found: {transcript_path}",
                    "short_summary": "",
                    "detailed_summary": ""
                }

            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()

            # Try normal summarization first
            short_result = self.summarize(transcript, summary_type="standard")
            detailed_result = self.summarize(transcript, summary_type="detailed")

            # If both succeeded, return them
            if short_result['success'] and detailed_result['success']:
                return {
                    "success": True,
                    "video_id": video_id,
                    "short_summary": short_result['summary'],
                    "detailed_summary": detailed_result['summary'],
                    "error": None
                }

            # Detect context-length errors and attempt hierarchical fallback
            error_msg = short_result.get('error') or detailed_result.get('error') or ''
            if 'context' in (error_msg or '').lower() or 'context_length_exceeded' in (error_msg or ''):
                # Determine chunk size based on model context limits (approximate using chars/token)
                def _get_model_token_limit(name: str) -> int:
                    lname = (name or '').lower()
                    # Heuristic mapping — conservative defaults
                    if '3.5' in lname:
                        return 4096
                    if 'gpt-4' in lname or 'gpt4' in lname:
                        return 16385
                    if 'gpt-4o' in lname or 'gpt4o' in lname:
                        return 32768
                    # Default to large limit to avoid over-splitting
                    return 16385

                chars_per_token = 4.0
                reserved_tokens_for_prompt = 512
                model_limit = _get_model_token_limit(getattr(self.llm, 'model', None) or '')
                usable_tokens = max(512, model_limit - reserved_tokens_for_prompt)
                chunk_size = int(usable_tokens * chars_per_token)
                # Keep a reasonable minimum/maximum
                chunk_size = max(3000, min(chunk_size, 16000))
                overlap = int(chunk_size * 0.08)  # ~8% overlap

                chunks = []
                start = 0
                transcript_len = len(transcript)
                while start < transcript_len:
                    end = min(start + chunk_size, transcript_len)
                    chunks.append(transcript[start:end])
                    # Next start should overlap slightly; ensure progress
                    next_start = end - overlap
                    if next_start <= start:
                        next_start = end
                    start = next_start

                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    try:
                        r = self.summarize(chunk, summary_type="standard")
                        if r.get('success') and r.get('summary'):
                            chunk_summaries.append(r.get('summary'))
                        else:
                            # As a last resort, truncate chunk and try again
                            r_trunc = self.summarize(chunk[:int(chunk_size/2)], summary_type="standard")
                            if r_trunc.get('success') and r_trunc.get('summary'):
                                chunk_summaries.append(r_trunc.get('summary'))
                    except Exception:
                        continue

                if not chunk_summaries:
                    return {
                        "success": False,
                        "error": error_msg or 'Failed to summarize transcript (no chunk summaries)',
                        "short_summary": "",
                        "detailed_summary": ""
                    }

                # Consolidate chunk summaries into final short summary
                combined = "\n\n---\n\n".join(chunk_summaries)
                combined_short = self.summarize(combined, summary_type="standard")
                combined_detailed = self.summarize(combined, summary_type="detailed")

                return {
                    "success": True,
                    "video_id": video_id,
                    "short_summary": combined_short.get('summary') if combined_short.get('success') else ('\n'.join(chunk_summaries)[:2000]),
                    "detailed_summary": combined_detailed.get('summary') if combined_detailed.get('success') else (combined_short.get('summary') if combined_short.get('success') else ''),
                    "error": None,
                    "fallback_used": True,
                    "chunk_count": len(chunks),
                    "chunk_size_chars": chunk_size
                }

            # If failure was not clearly context-length, return original error
            return {
                "success": False,
                "error": error_msg or 'Unknown summarization error',
                "short_summary": "",
                "detailed_summary": ""
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "short_summary": "",
                "detailed_summary": ""
            }
    
    def generate_questions(self, transcript: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate discussion questions based on the transcript.
        
        Args:
            transcript: Video transcript text
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary containing questions
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Based on the content, generate {num_questions} thoughtful discussion questions.
            Questions should:
            - Encourage critical thinking
            - Explore key concepts
            - Connect to broader implications
            - Be open-ended
            Format as a numbered list."""),
            ("user", "Generate discussion questions for this transcript:\n\n{transcript}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"transcript": transcript})
            questions = response.content
            
            return {
                "questions": questions,
                "count": num_questions,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "questions": "",
                "count": num_questions,
                "success": False,
                "error": str(e)
            }

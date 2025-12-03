"""
Helper for generating summaries for multiple content types.
"""
from typing import Dict, List
from langchain_openai import ChatOpenAI


class SummarizationHelper:
    """Generates topic-level summaries from multiple video summaries."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Initialize summarization helper.
        
        Args:
            model: LLM model to use
            temperature: Temperature for generation
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
    
    def generate_topic_summary(self, video_summaries: List[Dict], topic: str) -> Dict:
        """
        Generate overall summary for a topic with multiple videos.
        
        Args:
            video_summaries: List of video summary dictionaries
            topic: Main topic being summarized
        
        Returns:
            Dictionary with overall summary and metadata
        """
        if not video_summaries:
            return {
                'overall_summary': f"No video summaries available for topic: {topic}",
                'video_count': 0,
                'topic': topic
            }
        
        # Format video summaries
        video_texts = []
        for vs in video_summaries:
            video_texts.append(f"Video: {vs.get('title', 'Unknown')}")
            if isinstance(vs.get('summary'), dict):
                video_texts.append(f"Summary: {vs['summary'].get('short_summary', '')}")
            else:
                video_texts.append(f"Summary: {vs.get('summary', '')}")
            video_texts.append("---")
        
        video_content = "\n".join(video_texts)
        
        prompt = f"""
        Analyze these {len(video_summaries)} videos about "{topic}" and provide:
        
        1. OVERALL SUMMARY: 2-3 sentences about what these videos collectively cover
        2. KEY THEMES: 3-5 main themes that appear across multiple videos
        3. LEARNING PATH: Suggested order to watch these videos for beginners
        4. GAPS: Any important aspects of "{topic}" that seem missing
        
        Videos:
        {video_content}
        
        Format your response with clear headings.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return {
                'overall_summary': response.content,
                'video_count': len(video_summaries),
                'topic': topic
            }
        except Exception as e:
            print(f"Error generating topic summary: {str(e)}")
            return {
                'overall_summary': f"Collected {len(video_summaries)} videos about {topic}.",
                'video_count': len(video_summaries),
                'topic': topic
            }
    
    def combine_summaries(self, summaries: List[str], max_length: int = 500) -> str:
        """
        Combine multiple summaries into one coherent summary.
        
        Args:
            summaries: List of summary texts
            max_length: Maximum length for combined summary
        
        Returns:
            Combined summary text
        """
        if not summaries:
            return "No content to summarize."
        
        if len(summaries) == 1:
            return summaries[0][:max_length]
        
        combined_text = "\n\n".join(summaries)
        
        prompt = f"""
        Combine these summaries into one coherent summary (max {max_length} characters):
        
        {combined_text}
        
        Create a unified summary that captures all key points without repetition.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content[:max_length]
        except:
            # Fallback: just truncate combined text
            return combined_text[:max_length]

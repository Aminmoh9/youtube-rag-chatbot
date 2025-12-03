"""
Helps users get free API keys and stay within limits.
"""
import streamlit as st
import requests
from typing import Dict

class FreeTierAssistant:
    """Helps users access and manage free API tiers."""
    
    def show_free_tier_guide(self):
        """Show guide for getting free API keys."""
        st.markdown("## 🆓 Free Tier Guide")
        
        tabs = st.tabs(["OpenAI", "YouTube", "Pinecone", "Usage Tips"])
        
        with tabs[0]:
            st.markdown("""
            ### OpenAI Free Credits
            
            **🎁 $5 Free Credit** for new users (enough for ~250 research sessions)
            
            **How to get it:**
            1. Go to [platform.openai.com/signup](https://platform.openai.com/signup)
            2. Create account with email/Google/Microsoft
            3. Verify email and phone number
            4. Get **$5 free credit** automatically added
            
            **Create API Key:**
            1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
            2. Click "Create new secret key"
            3. Copy the key (shown only once!)
            4. Paste it in the setup page
            
            **Cost estimates:**
            - Embeddings: ~$0.0001 per 1K tokens
            - Whisper: ~$0.006 per minute
            - GPT-3.5: ~$0.002 per 1K tokens
            """)
        
        with tabs[1]:
            st.markdown("""
            ### YouTube Data API
            
            **🎁 10,000 units/day free quota**
            
            **How to get it:**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create new project or select existing
            3. Enable "YouTube Data API v3"
            4. Go to "Credentials" → "Create Credentials" → "API Key"
            5. Copy the key
            
            **Quota usage:**
            - Search: 100 units per call
            - Video details: 1-3 units per call
            - With 10,000 units: ~100 searches/day
            
            **Restrict key (recommended):**
            1. In Google Cloud Console
            2. Go to your API key
            3. Add restriction: "HTTP referrers"
            4. Add: `*.streamlit.app/*` (if deploying on Streamlit Cloud)
            """)
        
        with tabs[2]:
            st.markdown("""
            ### Pinecone Free Tier
            
            **🎁 Free Starter Tier** (no credit card required)
            
            **Includes:**
            - 1 index
            - 100,000 vectors
            - Basic support
            - No expiration
            
            **How to get it:**
            1. Go to [pinecone.io](https://www.pinecone.io/)
            2. Click "Get Started"
            3. Sign up with email/Google/GitHub
            4. Get API key from dashboard
            
            **Create Index:**
            1. In Pinecone console
            2. Click "Create Index"
            3. Name: `youtube-research`
            4. Dimension: `1536` (for OpenAI embeddings)
            5. Metric: `cosine`
            """)
        
        with tabs[3]:
            st.markdown("""
            ### 💡 Usage Optimization Tips
            
            **To stay within free limits:**
            
            **OpenAI ($5 credit):**
            - Use `gpt-3.5-turbo` instead of GPT-4
            - Limit transcripts to 10 minutes
            - Use `whisper-1` (fast) model
            - Cache embeddings when possible
            
            **YouTube (10K units/day):**
            - Limit to 5-7 videos per search
            - Re-use cached video metadata
            - Use "Quick mode" for metadata-only
            - Space out research sessions
            
            **Pinecone (100K vectors):**
            - Delete old sessions regularly
            - Use separate indexes per major topic
            - Compress text before embedding
            - Monitor vector count in console
            
            **Session Management:**
            - Clear browser cache to remove old keys
            - Bookmark with keys entered
            - Use incognito mode for privacy
            - Export data before clearing session
            """)
        
        # Quick links
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🆓 Get OpenAI Key", use_container_width=True):
                st.markdown("[👉 platform.openai.com](https://platform.openai.com/api-keys)")
        
        with col2:
            if st.button("📺 Get YouTube Key", use_container_width=True):
                st.markdown("[👉 console.cloud.google.com](https://console.cloud.google.com/apis/credentials)")
        
        with col3:
            if st.button("🗄️ Get Pinecone Key", use_container_width=True):
                st.markdown("[👉 pinecone.io](https://www.pinecone.io/)")
    
    def estimate_cost(self, video_count: int, audio_minutes: int = 0) -> Dict:
        """Estimate cost for a research session."""
        estimates = {
            'openai_embeddings': video_count * 0.002,  # ~$0.002 per video
            'openai_whisper': audio_minutes * 0.006,   # $0.006/min
            'openai_gpt': video_count * 0.001,         # ~$0.001 per Q&A
            'youtube_api': video_count * 0.0001,       # Minimal cost
        }
        
        total = sum(estimates.values())
        
        return {
            'total': total,
            'breakdown': estimates,
            'free_credit_equivalent': total * 200,  # How many sessions per $5
            'message': f"Estimated cost: ${total:.4f} ({total*100:.2f} cents)"
        }
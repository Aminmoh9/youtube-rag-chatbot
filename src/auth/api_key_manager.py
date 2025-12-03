"""
API Key Manager for secure API key handling across the application.
"""
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st


class APIKeyManager:
    """Manages API keys from environment variables and Streamlit secrets."""
    
    def __init__(self):
        """Initialize the API key manager."""
        # Load .env file if it exists
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        self._keys = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment variables and Streamlit secrets."""
        # Define required keys
        key_names = [
            "OPENAI_API_KEY",
            "YOUTUBE_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT",
            "PINECONE_INDEX_NAME",
            "LANGSMITH_API_KEY"
        ]
        
        for key_name in key_names:
            # Try environment variable first
            value = os.environ.get(key_name)
            
            # Try Streamlit secrets if running in Streamlit
            if not value:
                try:
                    value = st.secrets.get(key_name)
                except (FileNotFoundError, KeyError):
                    pass
            
            self._keys[key_name] = value
    
    def get_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key by name.
        
        Args:
            key_name: Name of the API key
            
        Returns:
            API key value or None if not found
        """
        return self._keys.get(key_name)
    
    def set_key(self, key_name: str, value: str):
        """
        Set an API key value (for runtime configuration).
        
        Args:
            key_name: Name of the API key
            value: API key value
        """
        self._keys[key_name] = value
        os.environ[key_name] = value
    
    def is_configured(self, key_name: str) -> bool:
        """
        Check if an API key is configured.
        
        Args:
            key_name: Name of the API key
            
        Returns:
            True if key is configured, False otherwise
        """
        value = self.get_key(key_name)
        return bool(value and value.strip())
    
    def validate_openai_key(self) -> bool:
        """Validate OpenAI API key."""
        return self.is_configured("OPENAI_API_KEY")
    
    def validate_youtube_key(self) -> bool:
        """Validate YouTube API key."""
        return self.is_configured("YOUTUBE_API_KEY")
    
    def validate_pinecone_config(self) -> bool:
        """Validate Pinecone configuration."""
        return (
            self.is_configured("PINECONE_API_KEY") and
            self.is_configured("PINECONE_INDEX_NAME")
        )
    
    def get_missing_keys(self) -> list[str]:
        """
        Get list of missing required API keys.
        
        Returns:
            List of missing key names
        """
        required_keys = [
            "OPENAI_API_KEY",
            "YOUTUBE_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_NAME"
        ]
        
        return [key for key in required_keys if not self.is_configured(key)]
    
    def show_setup_instructions(self):
        """Display setup instructions in Streamlit."""
        missing = self.get_missing_keys()
        
        if not missing:
            st.success("✅ All required API keys are configured!")
            return
        
        st.error(f"❌ Missing API keys: {', '.join(missing)}")
        
        st.markdown("""
        ### Setup Instructions
        
        1. Create a `.env` file in the project root directory
        2. Add your API keys:
        
        ```bash
        OPENAI_API_KEY=your_openai_key_here
        YOUTUBE_API_KEY=your_youtube_key_here
        PINECONE_API_KEY=your_pinecone_key_here
        PINECONE_INDEX_NAME=your_index_name
        LANGSMITH_API_KEY=your_langsmith_key_here  # Optional
        ```
        
        3. Restart the application
        
        #### Where to get API keys:
        - **OpenAI**: https://platform.openai.com/api-keys
        - **YouTube**: https://console.cloud.google.com/apis/credentials
        - **Pinecone**: https://app.pinecone.io/
        - **LangSmith** (optional): https://smith.langchain.com/
        """)
    
    def setup_streamlit_sidebar(self):
        """Create API key configuration in Streamlit sidebar."""
        with st.sidebar.expander("🔑 API Configuration", expanded=False):
            missing = self.get_missing_keys()
            
            if missing:
                st.warning(f"Missing: {', '.join(missing)}")
            else:
                st.success("All keys configured")
            
            # Allow runtime key input for missing keys
            for key_name in ["OPENAI_API_KEY", "YOUTUBE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]:
                if not self.is_configured(key_name):
                    value = st.text_input(
                        key_name.replace("_", " ").title(),
                        type="password" if "KEY" in key_name else "default",
                        key=f"input_{key_name}"
                    )
                    if value:
                        self.set_key(key_name, value)
                        st.success(f"✓ {key_name} configured")
                        st.rerun()
    
    def render_ui(self):
        """Alias for setup_streamlit_sidebar for consistency."""
        self.setup_streamlit_sidebar()


# Global instance
api_key_manager = APIKeyManager()

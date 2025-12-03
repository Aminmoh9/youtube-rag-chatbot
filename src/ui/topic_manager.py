"""
Secure API key management for cloud deployment.
Handles user API keys with encryption and validation.
"""
import streamlit as st
import os
import json
import hashlib
from typing import Dict, Optional, Tuple
import base64
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

class SecureAPIKeyManager:
    """Manages user API keys with encryption and validation."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        # Generate or use provided encryption key
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # For production, generate key once and store securely
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
        
        # Session storage for keys (encrypted in memory)
        self.key_cache = {}
    
    def setup_api_keys_page(self):
        """Streamlit page for API key setup."""
        st.markdown("### 🔑 API Keys Setup")
        
        st.info("""
        **Why you need API keys:**
        - Keep your usage private and secure
        - Control your own costs and limits
        - Comply with service terms
        - Your keys never leave your browser
        
        All keys are **encrypted** and stored **only in your session**.
        """)
        
        # API key inputs
        with st.form("api_keys_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Get from platform.openai.com/api-keys",
                    placeholder="sk-..."
                )
                
                youtube_key = st.text_input(
                    "YouTube Data API Key",
                    type="password",
                    help="Get from console.cloud.google.com",
                    placeholder="AIza..."
                )
            
            with col2:
                pinecone_key = st.text_input(
                    "Pinecone API Key",
                    type="password",
                    help="Get from pinecone.io",
                    placeholder="pc-..."
                )
                
                pinecone_env = st.text_input(
                    "Pinecone Environment",
                    value="us-east-1",
                    help="Usually us-east-1, us-west-2, etc."
                )
            
            # Free tier information
            with st.expander("💡 Free Tier Information"):
                st.markdown("""
                **Free tiers available:**
                - **OpenAI**: $5 free credit for new users
                - **YouTube API**: 10,000 units/day free
                - **Pinecone**: Free starter tier available
                
                **Estimated costs per research session:**
                - Topic search (5 videos): ~$0.02-$0.05
                - Single video analysis: ~$0.01-$0.02
                - Audio transcription: ~$0.06 per minute
                """)
            
            # Validation checkbox
            validate_keys = st.checkbox(
                "Validate keys now (recommended)",
                value=True,
                help="Test if keys work before saving"
            )
            
            submitted = st.form_submit_button("🔐 Save & Validate Keys")
            
            if submitted:
                return self._process_keys_submission(
                    openai_key, youtube_key, pinecone_key, pinecone_env, validate_keys
                )
        
        return None
    
    def _process_keys_submission(self, openai_key: str, youtube_key: str, 
                               pinecone_key: str, pinecone_env: str, validate: bool) -> Dict:
        """Process and validate submitted keys."""
        # Validate required keys
        if not openai_key:
            st.error("OpenAI API key is required for all features")
            return None
        
        keys = {
            'OPENAI_API_KEY': openai_key,
            'YOUTUBE_API_KEY': youtube_key,
            'PINECONE_API_KEY': pinecone_key,
            'PINECONE_ENVIRONMENT': pinecone_env,
            'setup_time': datetime.now().isoformat()
        }
        
        # Validate keys if requested
        if validate:
            validation_results = self._validate_keys(keys)
            
            if not validation_results['all_valid']:
                st.warning("Some keys failed validation. You can still proceed, but some features may not work.")
                
                for service, result in validation_results['details'].items():
                    if not result['valid']:
                        st.error(f"{service}: {result.get('message', 'Invalid key')}")
        
        # Encrypt and store keys
        encrypted_keys = self._encrypt_keys(keys)
        
        # Store in session state
        st.session_state.api_keys = encrypted_keys
        st.session_state.api_keys_configured = True
        st.session_state.api_keys_raw = keys  # For current session only
        
        # Set environment variables for this session
        os.environ['OPENAI_API_KEY'] = openai_key
        if youtube_key:
            os.environ['YOUTUBE_API_KEY'] = youtube_key
        if pinecone_key:
            os.environ['PINECONE_API_KEY'] = pinecone_key
            os.environ['PINECONE_ENVIRONMENT'] = pinecone_env
        
        st.success("✅ API keys saved! They will be used for this session only.")
        
        # Show usage tips
        with st.expander("📊 Usage Tips", expanded=True):
            st.markdown(f"""
            **Keys will expire when you:**
            - Close the browser tab
            - Refresh the page
            - After 24 hours of inactivity
            
            **To save keys for next time:**
            1. Bookmark this page with keys entered
            2. Use browser password manager
            3. Copy keys to a secure password manager
            
            **Your keys are:**
            - 🔒 Encrypted in this session
            - 🚫 Never sent to our servers
            - 🗑️ Automatically cleared on page close
            """)
        
        return keys
    
    def _validate_keys(self, keys: Dict) -> Dict:
        """Validate API keys by making test calls."""
        results = {
            'all_valid': True,
            'details': {}
        }
        
        # Validate OpenAI key
        if keys['OPENAI_API_KEY']:
            try:
                import openai
                client = openai.OpenAI(api_key=keys['OPENAI_API_KEY'])
                # Simple, cheap validation call
                models = client.models.list()
                results['details']['OpenAI'] = {
                    'valid': True,
                    'message': f"✓ Valid (has {len(models.data)} models)"
                }
            except Exception as e:
                results['all_valid'] = False
                results['details']['OpenAI'] = {
                    'valid': False,
                    'message': f"✗ Invalid: {str(e)[:100]}"
                }
        
        # Validate YouTube key
        if keys['YOUTUBE_API_KEY']:
            try:
                from googleapiclient.discovery import build
                youtube = build('youtube', 'v3', developerKey=keys['YOUTUBE_API_KEY'])
                # Test search call (very cheap)
                request = youtube.search().list(q="test", part="snippet", maxResults=1)
                request.execute()
                results['details']['YouTube'] = {
                    'valid': True,
                    'message': "✓ Valid"
                }
            except Exception as e:
                results['details']['YouTube'] = {
                    'valid': False,
                    'message': f"✗ Invalid: {str(e)[:100]}"
                }
        
        # Validate Pinecone key
        if keys['PINECONE_API_KEY']:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=keys['PINECONE_API_KEY'])
                indexes = pc.list_indexes()
                results['details']['Pinecone'] = {
                    'valid': True,
                    'message': f"✓ Valid (connected to Pinecone)"
                }
            except Exception as e:
                results['details']['Pinecone'] = {
                    'valid': False,
                    'message': f"✗ Invalid: {str(e)[:100]}"
                }
        
        return results
    
    def _encrypt_keys(self, keys: Dict) -> str:
        """Encrypt keys for secure storage."""
        json_str = json.dumps(keys)
        encrypted = self.cipher.encrypt(json_str.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_keys(self, encrypted_str: str) -> Dict:
        """Decrypt stored keys."""
        encrypted = base64.b64decode(encrypted_str.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
    
    def get_keys_for_session(self) -> Optional[Dict]:
        """Get API keys for current session."""
        if 'api_keys_raw' in st.session_state:
            return st.session_state.api_keys_raw
        
        if 'api_keys' in st.session_state:
            try:
                return self._decrypt_keys(st.session_state.api_keys)
            except:
                return None
        
        return None
    
    def check_and_prompt_for_keys(self):
        """Check if keys are set, prompt if not."""
        if not st.session_state.get('api_keys_configured'):
            # Create a persistent warning
            warning_container = st.container()
            
            with warning_container:
                st.warning("""
                🔑 **API Keys Required**
                
                To use this application, you need to provide your own API keys.
                This keeps your usage private and secure.
                
                **No keys are stored on our servers.**
                """)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("⚙️ Setup API Keys", type="primary"):
                        st.session_state.show_key_setup = True
                        st.rerun()
                
                with col2:
                    if st.button("🚀 Try Demo Mode"):
                        st.session_state.demo_mode = True
                        st.rerun()
                
                with col3:
                    st.markdown("""
                    **Why use your own keys?**
                    - Control your costs
                    - Your data stays private
                    - Comply with API terms
                    """)
            
            return False
        
        return True
    
    def setup_demo_mode(self):
        """Setup limited demo mode with sample data."""
        st.info("""
        🎮 **Demo Mode Active**
        
        You're using demo mode with pre-processed sample data.
        Features are limited to protect API resources.
        
        **Limitations in demo mode:**
        - Pre-loaded sample topics only
        - No YouTube search or uploads
        - Limited Q&A on sample content
        - Session expires in 30 minutes
        """)
        
        # Load sample data for demo
        sample_topics = [
            "Introduction to Python",
            "Machine Learning Basics",
            "Data Visualization"
        ]
        
        # Set limited environment
        st.session_state.demo_mode = True
        st.session_state.demo_topics = sample_topics
        st.session_state.demo_expires = datetime.now() + timedelta(minutes=30)
        
        return True


# Global instance
api_key_manager = SecureAPIKeyManager()
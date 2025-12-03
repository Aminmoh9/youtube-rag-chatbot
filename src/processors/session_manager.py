"""
Session management for content processing.
"""
import json
import hashlib
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime


class SessionManager:
    """Manages processing sessions and their persistence."""
    
    def __init__(self, sessions_dir: str = "data/content_sessions"):
        """
        Initialize session manager.
        
        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions = {}
    
    def generate_session_id(self, seed: str) -> str:
        """
        Generate unique session ID.
        
        Args:
            seed: Seed string for ID generation
        
        Returns:
            Unique session ID
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        seed_hash = hashlib.md5(seed.encode()).hexdigest()[:8]
        return f"session-{timestamp}-{seed_hash}"
    
    def save_session(self, session_id: str, session_data: Dict):
        """
        Save session to disk and memory.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data dictionary
        """
        # Add timestamp if not present
        if 'created_at' not in session_data:
            session_data['created_at'] = datetime.now().isoformat()
        
        # Save to disk
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Cache in memory
        self.active_sessions[session_id] = session_data
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data or None if not found
        """
        # Check memory first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from disk
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                # Cache for future access
                self.active_sessions[session_id] = session_data
                return session_data
        
        return None
    
    def list_sessions(self, limit: int = 50) -> list:
        """
        List all sessions sorted by creation date.
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of session metadata
        """
        sessions = []
        
        for session_file in sorted(self.sessions_dir.glob("session-*.json"), reverse=True):
            if len(sessions) >= limit:
                break
            
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    sessions.append({
                        'session_id': session_data.get('session_id'),
                        'input_method': session_data.get('input_method'),
                        'topic': session_data.get('topic'),
                        'created_at': session_data.get('created_at'),
                        'status': session_data.get('status', 'unknown')
                    })
            except:
                continue
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from disk and memory.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted successfully
        """
        # Remove from memory
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove from disk
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def update_session(self, session_id: str, updates: Dict):
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
        """
        session = self.get_session(session_id)
        if session:
            session.update(updates)
            session['updated_at'] = datetime.now().isoformat()
            self.save_session(session_id, session)

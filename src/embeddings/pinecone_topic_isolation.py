"""
STRICT topic isolation for Pinecone.
Uses ONLY namespaces (safe for free tier).
Separate indexes disabled to avoid max-index error.
"""
import os
import hashlib
from pinecone import Pinecone
from typing import List, Dict

class StrictTopicIsolation:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.main_index_name = "youtube-research-isolated"
        self._ensure_main_index()

    def _ensure_main_index(self):
        """Create main index if it doesn't exist."""
        if self.main_index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.main_index_name,
                dimension=1536, 
                metric='cosine',
                spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
            )

    def get_topic_namespace(self, topic: str) -> str:
        clean_topic = topic.lower().strip()
        namespace_hash = hashlib.md5(clean_topic.encode()).hexdigest()[:16]
        return f"topic-{namespace_hash}"

    def upsert_with_isolation(self, vectors: List[Dict], topic: str) -> Dict:
        """Always uses namespaces. No separate index creation."""
        index = self.pc.Index(self.main_index_name)
        namespace = self.get_topic_namespace(topic)

        # Add topic metadata
        for vector in vectors:
            vector.setdefault('metadata', {})
            vector['metadata']['topic'] = topic
            vector['metadata']['topic_hash'] = namespace

        if not vectors:
            return {
                "vectors_upserted": 0,
                "index": self.main_index_name,
                "namespace": namespace,
                "topic": topic,
                "isolation_method": "namespace",
                "error": "No vectors to upsert"
            }

        index.upsert(vectors=vectors, namespace=namespace)

        return {
            "vectors_upserted": len(vectors),
            "index": self.main_index_name,
            "namespace": namespace,
            "topic": topic,
            "isolation_method": "namespace"
        }

    def query_with_isolation(self, query_text: str, topic: str, top_k: int = 5, namespace: str = None, filters: dict = None):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        embedding = client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        ).data[0].embedding

        index = self.pc.Index(self.main_index_name)
        # Use provided namespace or compute from topic
        namespace = namespace if namespace else self.get_topic_namespace(topic)

        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=filters if filters else None
        )

        # Return all results from the namespace
        # No need to filter by topic name since namespace already isolates by topic
        filtered = []
        for m in results.matches:
            md = m.metadata or {}
            # Try 'text' first, fall back to 'text_preview'
            text = md.get("text", "") or md.get("text_preview", "")
            
            filtered.append({
                "id": m.id,
                "score": m.score,
                "metadata": md,
                "text": text
            })

        return filtered

    def delete_topic_data(self, topic: str) -> bool:
        """Delete all vectors for a topic using namespace deletion."""
        try:
            index = self.pc.Index(self.main_index_name)
            namespace = self.get_topic_namespace(topic)
            index.delete(delete_all=True, namespace=namespace)
            return True
        except:
            return False

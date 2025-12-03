"""
Pinecone data management utilities.
View, search, and manage stored data in Pinecone.
"""
import os
from pinecone import Pinecone
from typing import List, Dict, Optional


class PineconeDataManager:
    """Manage and explore Pinecone data."""
    
    def __init__(self):
        # Use PINECONE_API_KEY from environment; allow using env override for index name
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        # Respect environment variable `PINECONE_INDEX_NAME` if provided; fallback to prior default
        self.main_index_name = os.getenv('PINECONE_INDEX_NAME', "youtube-research-isolated")
        # Optionally use PINECONE_ENVIRONMENT (some Pinecone SDKs may require it)
        self.environment = os.getenv('PINECONE_ENVIRONMENT') or os.getenv('PINECONE_ENV')
    
    def list_all_indexes(self) -> List[str]:
        """List all Pinecone indexes in your account."""
        return self.pc.list_indexes().names()
    
    def get_index_stats(self, index_name: str = None) -> Dict:
        """Get statistics for an index."""
        if not index_name:
            index_name = self.main_index_name
        
        index = self.pc.Index(index_name)
        stats = index.describe_index_stats()
        
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'namespaces': stats.namespaces if hasattr(stats, 'namespaces') else {}
        }
    
    def list_all_topics(self) -> List[Dict]:
        """
        List all topics stored in the main index.
        Each namespace represents a different topic.
        Returns actual topic names from metadata.
        """
        from openai import OpenAI
        
        stats = self.get_index_stats()
        topics = []
        
        index = self.pc.Index(self.main_index_name)
        
        for namespace, info in stats.get('namespaces', {}).items():
            if namespace.startswith('topic-'):
                # Query one vector from this namespace to get the actual topic name
                try:
                    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    embedding = client.embeddings.create(
                        input="get topic name",
                        model="text-embedding-ada-002"
                    ).data[0].embedding
                    
                    results = index.query(
                        vector=embedding,
                        top_k=1,
                        include_metadata=True,
                        namespace=namespace
                    )
                    
                    # Extract actual topic name from metadata
                    topic_name = "Unknown"
                    if results.matches and len(results.matches) > 0:
                        metadata = results.matches[0].metadata
                        topic_name = metadata.get('topic', 'Unknown') if metadata else 'Unknown'
                    
                    topics.append({
                        'namespace': namespace,
                        'topic_name': topic_name,  # Actual readable name!
                        'vector_count': info.get('vector_count', 0),
                        'topic_hash': namespace.replace('topic-', '')
                    })
                except Exception as e:
                    # If we can't get the topic name, still show the namespace
                    topics.append({
                        'namespace': namespace,
                        'topic_name': 'Unknown',
                        'vector_count': info.get('vector_count', 0),
                        'topic_hash': namespace.replace('topic-', '')
                    })
        
        return topics
    
    def search_topic_by_hash(self, topic_hash: str) -> Optional[Dict]:
        """Find topic information by its hash."""
        topics = self.list_all_topics()
        for topic in topics:
            if topic['topic_hash'] == topic_hash:
                return topic
        return None
    
    def sample_vectors_from_topic(self, topic: str, limit: int = 5) -> List[Dict]:
        """
        Get sample vectors from a topic to see what's stored.
        
        Args:
            topic: Topic name (will be hashed to find namespace)
            limit: Number of samples to retrieve
        """
        import hashlib
        from openai import OpenAI
        
        # Get the namespace for this topic
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:16]
        namespace = f"topic-{topic_hash}"
        
        # First try to list IDs in the namespace
        index = self.pc.Index(self.main_index_name)
        
        # Try querying with the topic name itself as the query
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            embedding = client.embeddings.create(
                input=topic,  # Use topic name instead of generic query
                model="text-embedding-ada-002"
            ).data[0].embedding
            
            results = index.query(
                vector=embedding,
                top_k=limit,
                include_metadata=True,
                namespace=namespace
            )
            
            samples = []
            for match in results.matches:
                samples.append({
                    'id': match.id,
                    'score': match.score,
                    'text_preview': match.metadata.get('text', '')[:200] if match.metadata else '',
                    'source_id': match.metadata.get('source_id') if match.metadata else None,
                    'title': match.metadata.get('title') if match.metadata else None,
                    'metadata': match.metadata if match.metadata else {}
                })
            
            return samples
        except Exception as e:
            print(f"Error sampling vectors: {e}")
            return []
    
    def sample_vectors_by_namespace(self, namespace: str, limit: int = 5) -> List[Dict]:
        """
        Get sample vectors directly by namespace (for when you know the exact namespace).
        
        Args:
            namespace: Full namespace string (e.g., 'topic-0c6522d2')
            limit: Number of samples to retrieve
        """
        from openai import OpenAI
        
        index = self.pc.Index(self.main_index_name)
        
        try:
            # Use a generic embedding to sample
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            embedding = client.embeddings.create(
                input="show me content",
                model="text-embedding-ada-002"
            ).data[0].embedding
            
            results = index.query(
                vector=embedding,
                top_k=limit,
                include_metadata=True,
                namespace=namespace
            )
            
            samples = []
            for match in results.matches:
                samples.append({
                    'id': match.id,
                    'score': match.score,
                    'text_preview': match.metadata.get('text', '')[:200] if match.metadata else '',
                    'source_id': match.metadata.get('source_id') if match.metadata else None,
                    'title': match.metadata.get('title') if match.metadata else None,
                    'metadata': match.metadata if match.metadata else {}
                })
            
            return samples
        except Exception as e:
            print(f"Error sampling by namespace: {e}")
            return []
    
    def query_existing_data(self, query_text: str, topic: str, top_k: int = 5) -> List[Dict]:
        """
        Query existing data in Pinecone for a specific topic.
        This is how you reuse already stored data.
        
        Args:
            query_text: Your search query
            topic: Topic name to search within
            top_k: Number of results to return
        """
        from openai import OpenAI
        import hashlib
        
        # Generate embedding for query
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        embedding = client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        # Get namespace for topic
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:16]
        namespace = f"topic-{topic_hash}"
        
        # Query the index
        index = self.pc.Index(self.main_index_name)
        
        try:
            results = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'source_id': match.metadata.get('source_id'),
                    'title': match.metadata.get('title'),
                    'url': match.metadata.get('url', ''),
                    'metadata': match.metadata
                })
            
            return matches
        except Exception as e:
            print(f"Error querying data: {e}")
            return []
    
    def delete_topic(self, topic: str) -> bool:
        """
        Delete all data for a specific topic.
        
        Args:
            topic: Topic name to delete
        """
        import hashlib
        
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:16]
        namespace = f"topic-{topic_hash}"
        
        try:
            index = self.pc.Index(self.main_index_name)
            index.delete(delete_all=True, namespace=namespace)
            print(f"✅ Deleted all data for topic: {topic}")
            return True
        except Exception as e:
            print(f"❌ Error deleting topic: {e}")
            return False
    
    def get_topic_metadata(self, topic: str) -> Dict:
        """
        Get metadata about what's stored for a topic.
        
        Args:
            topic: Topic name
        """
        import hashlib
        
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:16]
        namespace = f"topic-{topic_hash}"
        
        stats = self.get_index_stats()
        namespace_stats = stats.get('namespaces', {}).get(namespace, {})
        
        return {
            'topic': topic,
            'namespace': namespace,
            'vector_count': namespace_stats.get('vector_count', 0),
            'exists': namespace in stats.get('namespaces', {})
        }
    
    def export_topic_data(self, topic: str, output_file: str = None) -> List[Dict]:
        """
        Export all data for a topic to view or backup.
        
        Args:
            topic: Topic name
            output_file: Optional JSON file to save data
        """
        samples = self.sample_vectors_from_topic(topic, limit=100)
        
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"✅ Exported {len(samples)} vectors to {output_file}")
        
        return samples


def main():
    """Example usage of PineconeDataManager."""
    manager = PineconeDataManager()
    
    print("=== Pinecone Data Manager ===\n")
    
    # List all indexes
    print("📊 Available Indexes:")
    indexes = manager.list_all_indexes()
    for idx in indexes:
        print(f"  - {idx}")
    
    print("\n📈 Main Index Stats:")
    stats = manager.get_index_stats()
    print(f"  Total Vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    
    # List all topics
    print("\n🏷️ Stored Topics:")
    topics = manager.list_all_topics()
    if topics:
        for topic in topics:
            print(f"  - Namespace: {topic['namespace']}")
            print(f"    Vectors: {topic['vector_count']}")
            print(f"    Hash: {topic['topic_hash']}")
    else:
        print("  No topics found")
    
    # Example: Query existing data
    print("\n💡 Example: To query existing data in your app:")
    print("""
    from src.utils.pinecone_manager import PineconeDataManager
    
    manager = PineconeDataManager()
    
    # Search existing data
    results = manager.query_existing_data(
        query_text="What are the main points?",
        topic="your_topic_name",
        top_k=5
    )
    
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Text: {result['text'][:200]}...")
    """)


if __name__ == "__main__":
    main()

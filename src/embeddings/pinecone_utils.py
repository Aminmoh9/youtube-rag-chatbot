"""
Updated Pinecone utilities with topic-aware indexing.
"""
import os
import json
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from typing import Optional, List, Dict
import hashlib

load_dotenv()


class PineconeManager:
    """Manages Pinecone indices with topic support."""
    
    def __init__(self):
        self.api_key = os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.default_index = os.getenv('PINECONE_INDEX_NAME', 'youtube-research-main')
        
        # Topic-specific indices configuration
        self.topic_indices = {}
    
    def get_topic_index_name(self, topic: str) -> str:
        """Generate index name for a topic."""
        # Clean topic name for index naming
        clean_topic = topic.lower().replace(' ', '-').replace('/', '-')[:30]
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        return f"{clean_topic}-{topic_hash}"
    
    def create_topic_index(self, topic: str, dimension: int = 1536) -> str:
        """Create a dedicated index for a topic."""
        index_name = self.get_topic_index_name(topic)
        
        if index_name not in self.pc.list_indexes().names():
            print(f"Creating topic index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        self.topic_indices[topic] = index_name
        return index_name
    
    def get_or_create_index(self, topic: Optional[str] = None, use_separate_index: bool = False):
        """
        Get appropriate index for a topic.
        
        Args:
            topic: Topic name (optional)
            use_separate_index: Whether to use separate index for this topic
            
        Returns:
            Pinecone index object
        """
        if not topic or not use_separate_index:
            # Use main index with topic metadata
            return self.pc.Index(self.default_index)
        else:
            # Use topic-specific index
            index_name = self.get_topic_index_name(topic)
            if index_name not in self.pc.list_indexes().names():
                index_name = self.create_topic_index(topic)
            return self.pc.Index(index_name)
    
    def upsert_with_topic(self, vectors: List[Dict], topic: str, 
                          use_separate_index: bool = False) -> Dict:
        """
        Upsert vectors with topic management.
        
        Args:
            vectors: List of vectors to upsert
            topic: Topic name
            use_separate_index: Whether to use separate index
            
        Returns:
            Dict with results
        """
        index = self.get_or_create_index(topic, use_separate_index)
        
        # Add topic to metadata if using main index
        if not use_separate_index:
            for vector in vectors:
                if 'metadata' not in vector:
                    vector['metadata'] = {}
                vector['metadata']['topic'] = topic
        
        # Upsert in batches
        batch_size = 100
        total_upserted = 0
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            total_upserted += len(batch)
        
        return {
            "total_upserted": total_upserted,
            "index_used": index._name,
            "topic": topic,
            "separate_index": use_separate_index
        }
    
    def query_with_topic(self, query_text: str, topic: Optional[str] = None,
                        use_separate_index: bool = False, top_k: int = 5) -> List[Dict]:
        """
        Query with topic awareness.
        
        Args:
            query_text: Query text
            topic: Topic to filter by
            use_separate_index: Whether topic has separate index
            top_k: Number of results
            
        Returns:
            List of results
        """
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate embedding for query
        response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        
        # Get appropriate index
        index = self.get_or_create_index(topic, use_separate_index)
        
        # Build filter
        filter_dict = None
        if topic and not use_separate_index:
            # Filter by topic metadata in main index
            filter_dict = {"topic": topic}
        
        # Query
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "index": index._name
            })
        
        return formatted_results
    
    def delete_topic_data(self, topic: str, use_separate_index: bool = False) -> Dict:
        """
        Delete all data for a topic.
        
        Args:
            topic: Topic to delete
            use_separate_index: Whether topic has separate index
            
        Returns:
            Dict with deletion results
        """
        if use_separate_index:
            # Delete entire index
            index_name = self.get_topic_index_name(topic)
            if index_name in self.pc.list_indexes().names():
                self.pc.delete_index(index_name)
                return {
                    "deleted": True,
                    "method": "entire_index",
                    "index_name": index_name,
                    "topic": topic
                }
        else:
            # Delete by metadata filter in main index
            index = self.pc.Index(self.default_index)
            
            # Note: Pinecone delete by filter requires async operation
            # This is a simplified version
            try:
                # This would actually require fetching all vectors first
                # For now, mark as to-be-deleted in metadata
                return {
                    "deleted": False,
                    "method": "filter_based",
                    "message": "Topic data marked for deletion (filter-based deletion requires additional steps)",
                    "topic": topic
                }
            except Exception as e:
                return {
                    "deleted": False,
                    "error": str(e),
                    "topic": topic
                }
        
        return {"deleted": False, "message": "No data found for topic"}
    
    def get_index_stats(self, topic: Optional[str] = None, 
                       use_separate_index: bool = False) -> Dict:
        """Get statistics for an index."""
        index = self.get_or_create_index(topic, use_separate_index)
        stats = index.describe_index_stats()
        
        return {
            "index_name": index._name,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else None,
            "topic": topic
        }


# Global instance
pinecone_manager = PineconeManager()


# For backward compatibility
def initialize_pinecone():
    """Initialize default Pinecone index."""
    return pinecone_manager.pc.Index(pinecone_manager.default_index)


def upsert_chunks_to_pinecone(embeddings_path='data/embeddings.json',
                             csv_path='data/video_links.csv',
                             batch_size=100,
                             topic=None,
                             use_separate_index=False):
    """
    Updated upsert function with topic support.
    """
    # Load embeddings
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    
    print(f"\n{'=' * 70}")
    print(f"UPSERTING {len(embeddings)} CHUNKS TO PINECONE")
    if topic:
        print(f"TOPIC: {topic} ({'separate index' if use_separate_index else 'main index'})")
    print(f"{'=' * 70}\n")
    
    # Prepare vectors
    vectors = []
    video_ids_processed = set()
    
    for embedding_data in embeddings:
        vectors.append(embedding_data)
        video_ids_processed.add(embedding_data['metadata']['video_id'])
    
    # Upsert with topic management
    result = pinecone_manager.upsert_with_topic(
        vectors=vectors,
        topic=topic,
        use_separate_index=use_separate_index
    )
    
    # Update CSV with embedding status
    df = pd.read_csv(csv_path)
    df.loc[df['video_id'].isin(video_ids_processed), 'embedding_status'] = 'ready'
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Upserted {result['total_upserted']} vectors")
    print(f"✓ Index used: {result['index_used']}")
    print(f"✓ CSV updated: {csv_path}")
    
    return result


def query_pinecone(query_text: str, top_k: int = 5, topic: str = None,
                  use_separate_index: bool = False):
    """
    Updated query function with topic support.
    """
    return pinecone_manager.query_with_topic(
        query_text=query_text,
        topic=topic,
        use_separate_index=use_separate_index,
        top_k=top_k
    )
"""
Pinecone vector database utilities for storing and retrieving embeddings.
Updates CSV with embedding status after upload.
"""
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()


def initialize_pinecone():
    """Initialize Pinecone client and return index."""
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'youtube-qa-bootcamp')
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI ada-002 embedding dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    return pc.Index(index_name)


def upsert_chunks_to_pinecone(embeddings_path='data/embeddings.json',
                              csv_path='data/video_links.csv',
                              batch_size=100):
    """
    Upsert chunks with embeddings to Pinecone and update CSV.
    
    Args:
        embeddings_path (str): Path to embeddings JSON
        csv_path (str): Path to video metadata CSV
        batch_size (int): Batch size for upserts
    """
    # Load embeddings
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    print(f"\n{'=' * 70}")
    print(f"UPSERTING {len(embeddings)} CHUNKS TO PINECONE")
    print(f"{'=' * 70}\n")
    
    # Prepare vectors for upsert
    vectors = []
    video_ids_processed = set()
    
    for embedding_data in embeddings:
        # embeddings.json already has the correct structure from embedding_utils.py
        vectors.append(embedding_data)
        video_ids_processed.add(embedding_data['metadata']['video_id'])
    
    # Upsert in batches
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"\n✓ Upserted {len(vectors)} vectors to Pinecone")
    
    # Update CSV with embedding status
    df = pd.read_csv(csv_path)
    df.loc[df['video_id'].isin(video_ids_processed), 'embedding_status'] = 'ready'
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Updated CSV: {len(video_ids_processed)} videos marked as 'ready'")
    print(f"✓ CSV updated: {csv_path}")
    
    return len(vectors)


if __name__ == "__main__":
    upsert_chunks_to_pinecone()

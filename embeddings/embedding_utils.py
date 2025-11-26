"""
Generate embeddings for text chunks using OpenAI's embedding API.
Creates embeddings.json with vectors and metadata for Pinecone upload.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import time

load_dotenv()


def generate_embedding(text, model="text-embedding-ada-002"):
    """
    Generate embedding for a single text using OpenAI API.
    
    Args:
        text (str): Text to embed
        model (str): OpenAI embedding model
    
    Returns:
        list: Embedding vector (1536 dimensions for ada-002)
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Clean text
    text = text.replace("\n", " ").strip()
    
    # Generate embedding
    response = client.embeddings.create(
        input=text,
        model=model
    )
    
    return response.data[0].embedding


def generate_embeddings_for_chunks(chunks_path='data/chunks.json',
                                   output_path='data/embeddings.json',
                                   model="text-embedding-ada-002",
                                   batch_delay=0.1):
    """
    Generate embeddings for all chunks.
    
    Args:
        chunks_path (str): Path to chunks JSON
        output_path (str): Where to save embeddings
        model (str): OpenAI embedding model
        batch_delay (float): Delay between API calls (seconds)
    
    Returns:
        list: Chunks with embeddings added
    """
    # Load chunks
    if not Path(chunks_path).exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    if len(chunks) == 0:
        print("No chunks found to embed!")
        return []
    
    print("=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: 1536\n")
    
    embeddings_data = []
    successful = 0
    failed = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks"), 1):
        try:
            # Generate embedding
            embedding = generate_embedding(chunk['text'], model=model)
            
            # Prepare data for Pinecone
            metadata = {
                'video_id': chunk['video_id'],
                'title': chunk['title'],
                'chunk_type': chunk['chunk_type'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'topic': chunk['topic'],
                'channel': chunk['channel'],
                'url': chunk['url'],
                'text': chunk['text'][:1000]  # Store first 1000 chars in metadata
            }
            
            # Only add chapter_title if it's not None
            if chunk.get('chapter_title'):
                metadata['chapter_title'] = chunk['chapter_title']
            
            embeddings_data.append({
                'id': chunk['chunk_id'],
                'values': embedding,
                'metadata': metadata
            })
            
            successful += 1
            
            # Rate limiting
            time.sleep(batch_delay)
            
        except Exception as e:
            print(f"\n✗ Error embedding chunk {chunk['chunk_id']}: {str(e)}")
            failed += 1
    
    # Save embeddings
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"EMBEDDING GENERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(chunks)}")
    print(f"\n✓ Saved to: {output_path}")
    
    return embeddings_data


def estimate_embedding_cost(num_chunks, avg_tokens_per_chunk=500):
    """
    Estimate cost for embedding generation.
    
    Args:
        num_chunks (int): Number of chunks
        avg_tokens_per_chunk (int): Average tokens per chunk
    
    Returns:
        float: Estimated cost in USD
    """
    # OpenAI text-embedding-ada-002 pricing: $0.0001 per 1K tokens
    total_tokens = num_chunks * avg_tokens_per_chunk
    cost = (total_tokens / 1000) * 0.0001
    
    print(f"Estimated cost for {num_chunks} chunks:")
    print(f"  Total tokens: ~{total_tokens:,}")
    print(f"  Cost: ~${cost:.4f}")
    
    return cost


if __name__ == "__main__":
    # Check if chunks exist
    chunks_path = 'data/chunks.json'
    
    if not Path(chunks_path).exists():
        print("❌ Chunks file not found!")
        print("Please run chunking.py first to create chunks.")
        exit(1)
    
    # Load chunks to estimate cost
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Found {len(chunks)} chunks to embed\n")
    estimate_embedding_cost(len(chunks))
    
    print("\n" + "=" * 70)
    response = input("\nProceed with embedding generation? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        generate_embeddings_for_chunks()
    else:
        print("Embedding generation cancelled.")

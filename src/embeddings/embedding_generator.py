"""
Embedding generation for text chunks.
"""
import os
from typing import List, Dict
from openai import OpenAI


class EmbeddingGenerator:
    """Generates embeddings for text chunks using OpenAI."""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """
        Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model to use
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_embeddings(self, chunks: List[Dict], session_metadata: Dict) -> List[Dict]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'text', 'metadata'
            session_metadata: Additional metadata to add to all chunks
        
        Returns:
            List of vectors ready for Pinecone upsert
        """
        embeddings = []
        
        for chunk in chunks:
            try:
                response = self.client.embeddings.create(
                    input=chunk['text'],
                    model=self.model
                )
                
                embedding_vector = response.data[0].embedding
                
                # Prepare vector
                vector_metadata = chunk['metadata'].copy()
                vector_metadata.update(session_metadata)
                # ADD THE FULL TEXT TO METADATA (for Pinecone storage and Q&A retrieval)
                vector_metadata['text'] = chunk['text']  # Store full text in metadata
                
                vector_data = {
                    'id': chunk['id'],
                    'values': embedding_vector,
                    'metadata': vector_metadata
                }
                
                embeddings.append(vector_data)
                
            except Exception as e:
                print(f"Error embedding chunk {chunk['id']}: {str(e)}")
                continue
        
        return embeddings
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []

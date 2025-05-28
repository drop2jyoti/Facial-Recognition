import redis
import numpy as np
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingStore:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True,
            encoding='utf-8'
        )
        print("Redis connection initialized")
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            print("Successfully connected to Redis")
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
    
    def store_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Store face embedding for a user
        Args:
            user_id: unique identifier for the user
            embedding: face embedding vector
        Returns:
            bool: success status
        """
        try:
            # Convert numpy array to list and store as JSON
            embedding_list = embedding.tolist()
            key = f"face:{user_id}"
            self.redis_client.set(key, json.dumps(embedding_list))
            print(f"Stored embedding for user {user_id}")
            
            # Verify storage
            stored = self.redis_client.get(key)
            if stored:
                print(f"Verified storage for user {user_id}")
            else:
                print(f"Warning: Could not verify storage for user {user_id}")
            
            return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False
    
    def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve face embedding for a user
        Args:
            user_id: unique identifier for the user
        Returns:
            numpy array of embedding or None if not found
        """
        try:
            key = f"face:{user_id}"
            embedding_json = self.redis_client.get(key)
            if embedding_json:
                embedding_list = json.loads(embedding_json)
                print(f"Retrieved embedding for user {user_id}")
                return np.array(embedding_list)
            print(f"No embedding found for user {user_id}")
            return None
        except Exception as e:
            print(f"Error retrieving embedding: {e}")
            return None
    
    def delete_embedding(self, user_id: str) -> bool:
        """
        Delete face embedding for a user
        Args:
            user_id: unique identifier for the user
        Returns:
            bool: success status
        """
        try:
            key = f"face:{user_id}"
            self.redis_client.delete(key)
            print(f"Deleted embedding for user {user_id}")
            return True
        except Exception as e:
            print(f"Error deleting embedding: {e}")
            return False
    
    def find_matches(self, query_embedding: np.ndarray, threshold: float = 0.5) -> List[Dict]:
        """
        Find matching faces in the database
        Args:
            query_embedding: face embedding to match
            threshold: similarity threshold (default: 0.5)
        Returns:
            list of matching user IDs and their similarity scores
        """
        matches = []
        try:
            # Get all face embeddings
            face_keys = list(self.redis_client.scan_iter("face:*"))
            print(f"Found {len(face_keys)} registered faces in database")
            
            for key in face_keys:
                # Handle both string and bytes keys
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                user_id = key.split(':')[1]
                stored_embedding = self.get_embedding(user_id)
                
                if stored_embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                    )
                    print(f"Similarity for {user_id}: {similarity}")
                    
                    # Add to matches if above threshold
                    if similarity >= threshold:
                        matches.append({
                            'user_id': user_id,
                            'similarity': float(similarity)
                        })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            print(f"Found {len(matches)} matches above threshold {threshold}")
            if len(matches) > 0:
                print(f"Best match: {matches[0]}")
            
            return matches
        except Exception as e:
            print(f"Error finding matches: {e}")
            return [] 
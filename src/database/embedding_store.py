import redis
import numpy as np
import logging
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True,
                encoding='utf-8'
            )
            logger.info("Successfully connected to Redis")
            
            # Test Redis connection
            try:
                self.redis_client.ping()
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.error(f"Error connecting to Redis: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
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
            logger.info(f"Stored embedding for user {user_id}")
            
            # Verify storage
            stored = self.redis_client.get(key)
            if stored:
                logger.info(f"Verified storage for user {user_id}")
            else:
                logger.warning(f"Warning: Could not verify storage for user {user_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
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
                logger.info(f"Retrieved embedding for user {user_id}")
                return np.array(embedding_list)
            logger.error(f"No embedding found for user {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
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
            logger.info(f"Deleted embedding for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False
    
    def find_matches(self, query_embedding: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """
        Find matching faces in the database
        Args:
            query_embedding: face embedding to match
            threshold: similarity threshold (default: 0.7)
        Returns:
            list of matching user IDs and their similarity scores
        """
        try:
            matches = []
            # Get all face keys
            face_keys = list(self.redis_client.scan_iter("face:*"))
            logger.info(f"Searching through {len(face_keys)} registered faces")
            
            for key in face_keys:
                # Handle both string and bytes keys
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                user_id = key.split(':')[1]
                
                # Get stored embedding
                stored_embedding = self.get_embedding(user_id)
                if stored_embedding is None:
                    continue
                
                # Calculate similarity
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                logger.info(f"Similarity with {user_id}: {similarity}")
                
                if similarity >= threshold:
                    matches.append({
                        'user_id': user_id,
                        'similarity': float(similarity)
                    })
            
            # Sort matches by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"Found {len(matches)} matches above threshold {threshold}")
            if len(matches) > 0:
                logger.info(f"Best match: {matches[0]}")
            
            return matches
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return [] 
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
import cv2
from io import BytesIO
from models.facenet import FaceNet
from utils.face_detection import FaceDetector
from utils.face_preprocessing import FacePreprocessor
from database.embedding_store import EmbeddingStore
import time
from typing import List, Dict, Optional
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug: Print API key
api_key = os.getenv("API_KEY")
logger.info(f"Loaded API Key: {api_key}")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(title="Face Recognition API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Initialize components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceNet().to(device)
model.eval()
face_detector = FaceDetector()
embedding_store = EmbeddingStore()
preprocessor = FacePreprocessor()

# Verify API key
async def verify_api_key(api_key: str = Depends(api_key_header)):
    stored_key = os.getenv("API_KEY")
    logger.info(f"Received API Key: {api_key}")
    logger.info(f"Stored API Key: {stored_key}")
    if api_key != stored_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

@app.get("/debug/registered-users")
@limiter.limit("30/minute")
async def get_registered_users(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    List all registered users (for debugging)
    """
    try:
        # Get all face keys from Redis
        face_keys = list(embedding_store.redis_client.scan_iter("face:*"))
        users = []
        for key in face_keys:
            # Handle both string and bytes keys
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            # Extract user_id from the key
            user_id = key.split(':')[1]
            users.append(user_id)
        
        return {
            "registered_users": users,
            "total_users": len(users)
        }
    except Exception as e:
        logger.error(f"Error in get_registered_users: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving registered users: {str(e)}"
        )

@app.post("/register")
@limiter.limit("5/minute")
async def register_face(
    request: Request,
    user_id: str,
    file: UploadFile = File(..., description="Image file containing the face"),
    api_key: str = Depends(verify_api_key)
):
    """
    Register a new face embedding
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess face
        preprocessed_face = preprocessor.preprocess_face(img)
        if preprocessed_face is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image or face detection failed. Please ensure the image contains a clear, front-facing face."
            )
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(preprocessed_face).permute(2, 0, 1).unsqueeze(0).to(device)
            embedding = model(face_tensor).cpu().numpy()[0]
        
        # Store embedding
        success = embedding_store.store_embedding(user_id, embedding)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store embedding"
            )
        
        return {
            "message": "Face registered successfully",
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/verify")
@limiter.limit("10/minute")
async def verify_face(
    request: Request,
    user_id: str,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Verify if the face matches the registered face
    """
    try:
        # Get stored embedding
        stored_embedding = embedding_store.get_embedding(user_id)
        if stored_embedding is None:
            raise HTTPException(status_code=404, detail="User not registered")
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess face
        preprocessed_face = preprocessor.preprocess_face(img)
        if preprocessed_face is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(preprocessed_face).permute(2, 0, 1).unsqueeze(0).to(device)
            query_embedding = model(face_tensor).cpu().numpy()[0]
        
        # Calculate similarity
        similarity = np.dot(query_embedding, stored_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
        )
        
        return {
            "verified": bool(similarity >= 0.7),
            "similarity": float(similarity)
        }
    
    except Exception as e:
        logger.error(f"Error in verify_face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify")
@limiter.limit("10/minute")
async def identify_face(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Identify the person in the image
    """
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess face
        preprocessed_face = preprocessor.preprocess_face(img)
        if preprocessed_face is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image or face detection failed. Please ensure the image contains a clear, front-facing face."
            )
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(preprocessed_face).permute(2, 0, 1).unsqueeze(0).to(device)
            query_embedding = model(face_tensor).cpu().numpy()[0]
        
        # Get all registered users
        face_keys = list(embedding_store.redis_client.scan_iter("face:*"))
        registered_users = []
        for key in face_keys:
            # Handle both string and bytes keys
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            user_id = key.split(':')[1]
            registered_users.append(user_id)
        
        # Find matches
        matches = embedding_store.find_matches(query_embedding)
        
        # Calculate similarities for all users for debugging
        all_similarities = []
        for user_id in registered_users:
            stored_embedding = embedding_store.get_embedding(user_id)
            if stored_embedding is not None:
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                all_similarities.append({
                    'user_id': user_id,
                    'similarity': float(similarity)
                })
        
        # Sort all similarities
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "matches": matches,
            "total_registered_faces": len(registered_users),
            "registered_users": registered_users,
            "all_similarities": all_similarities,  # For debugging
            "message": "No matches found" if not matches else f"Found {len(matches)} potential matches"
        }
    
    except Exception as e:
        logger.error(f"Error in identify_face: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/unregister/{user_id}")
async def unregister_face(user_id: str):
    """
    Remove a registered face
    """
    try:
        success = embedding_store.delete_embedding(user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete embedding")
        
        return {"message": "Face unregistered successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
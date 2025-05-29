from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
import cv2
from io import BytesIO
from src.models.facenet import FaceNet
from src.utils.face_detection import FaceDetector
from src.utils.face_preprocessing import FacePreprocessor
from src.database.embedding_store import EmbeddingStore
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

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

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
    
    if not api_key:
        logger.error("No API key provided")
        raise HTTPException(
            status_code=401,
            detail="No API key provided"
        )
    
    if not stored_key:
        logger.error("No API key configured in environment")
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server"
        )
    
    if api_key != stored_key:
        logger.error(f"API key mismatch. Received: {api_key}, Stored: {stored_key}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key

@app.get("/")
async def read_root():
    return FileResponse("src/static/index.html")

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
            logger.error("No face detected in registration image or preprocessing failed")
            raise HTTPException(
                status_code=400,
                detail="Could not process face in image. Please ensure the image contains a clear, front-facing face that is not too small or blurry."
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
        logger.info(f"Starting face verification for user_id: {user_id}")
        
        # Get stored embedding
        stored_embedding = embedding_store.get_embedding(user_id)
        if stored_embedding is None:
            logger.error(f"No stored embedding found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="User not registered")
        
        logger.info(f"Successfully retrieved stored embedding for user_id: {user_id}")
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess face
        preprocessed_face = preprocessor.preprocess_face(img)
        if preprocessed_face is None:
            logger.error("No face detected in verification image or preprocessing failed")
            raise HTTPException(status_code=400, detail="Could not process face in image for verification. Please ensure the image contains a clear, front-facing face that is not too small or blurry.")
        
        logger.info("Successfully preprocessed face from verification image")
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(preprocessed_face).permute(2, 0, 1).unsqueeze(0).to(device)
            query_embedding = model(face_tensor).cpu().numpy()[0]
        
        logger.info("Successfully generated embedding from verification image")
        
        # Calculate similarity
        similarity = np.dot(query_embedding, stored_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
        )
        
        logger.info(f"Calculated similarity score: {similarity}")
        
        # Get threshold from environment variable or use default
        threshold = float(os.getenv("FACE_MATCHING_THRESHOLD", "0.7"))
        logger.info(f"Using similarity threshold: {threshold}")
        
        is_verified = bool(similarity >= threshold)
        logger.info(f"Verification result: {'Verified' if is_verified else 'Not verified'}")
        
        return {
            "verified": is_verified,
            "similarity": float(similarity),
            "threshold": threshold
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
        logger.info("Starting face identification")
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess face
        preprocessed_face = preprocessor.preprocess_face(img)
        if preprocessed_face is None:
            logger.error("No face detected in identification image or preprocessing failed")
            raise HTTPException(
                status_code=400,
                detail="Could not process face in image for identification. Please ensure the image contains a clear, front-facing face that is not too small or blurry."
            )
        
        logger.info("Successfully preprocessed face from identification image")
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(preprocessed_face).permute(2, 0, 1).unsqueeze(0).to(device)
            query_embedding = model(face_tensor).cpu().numpy()[0]
        
        logger.info("Successfully generated embedding from identification image")
        
        # Get all registered users
        face_keys = list(embedding_store.redis_client.scan_iter("face:*"))
        registered_users = []
        for key in face_keys:
            # Handle both string and bytes keys
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            user_id = key.split(':')[1]
            registered_users.append(user_id)
        
        logger.info(f"Found {len(registered_users)} registered users: {registered_users}")
        
        if not registered_users:
            logger.error("No registered users found in database")
            return {
                "matches": [],
                "total_registered_faces": 0,
                "registered_users": [],
                "all_similarities": [],
                "message": "No users registered in the database"
            }
        
        # Find matches
        matches = embedding_store.find_matches(query_embedding)
        logger.info(f"Found {len(matches)} potential matches")
        
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
                logger.info(f"Similarity with {user_id}: {similarity}")
        
        # Sort all similarities
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get threshold from environment variable or use default
        threshold = float(os.getenv("FACE_MATCHING_THRESHOLD", "0.7"))
        logger.info(f"Using similarity threshold: {threshold}")
        
        return {
            "matches": matches,
            "total_registered_faces": len(registered_users),
            "registered_users": registered_users,
            "all_similarities": all_similarities,
            "threshold": threshold,
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
async def health_check(api_key: str = Depends(verify_api_key)):
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
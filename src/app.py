from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv

from models.facenet import FaceNet
from utils.face_detection import FaceDetector
from database.embedding_store import EmbeddingStore

load_dotenv()

app = FastAPI(title="Face Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceNet().to(device)
model.eval()
face_detector = FaceDetector()
embedding_store = EmbeddingStore()

@app.get("/debug/registered-users")
async def list_registered_users():
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
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving registered users: {str(e)}"
        )

@app.post("/register")
async def register_face(
    user_id: str = Query(..., description="Unique identifier for the user"),
    file: UploadFile = File(..., description="Image file containing the face")
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
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Detect and align face
        face = face_detector.preprocess_face(image)
        if face is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image or face detection failed. Please ensure the image contains a clear, front-facing face."
            )
        
        # Generate embedding
        try:
            with torch.no_grad():
                face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device)
                embedding = model(face_tensor).cpu().numpy()[0]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating face embedding: {str(e)}"
            )
        
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
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/verify")
async def verify_face(user_id: str, file: UploadFile = File(...)):
    """
    Verify if the face matches the registered face
    """
    try:
        # Get stored embedding
        stored_embedding = embedding_store.get_embedding(user_id)
        if stored_embedding is None:
            raise HTTPException(status_code=404, detail="User not registered")
        
        # Process new image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        face = face_detector.preprocess_face(image)
        if face is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device)
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """
    Identify the person in the image
    """
    try:
        # Process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        face = face_detector.preprocess_face(image)
        if face is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image or face detection failed. Please ensure the image contains a clear, front-facing face."
            )
        
        # Generate embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device)
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
        print(f"Error in identify endpoint: {str(e)}")  # Add debug logging
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
import cv2
import numpy as np
from typing import Tuple, List, Optional
from mtcnn import MTCNN

class FacePreprocessor:
    def __init__(self, 
                 min_face_size: int = 20,
                 detection_confidence: float = 0.9,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize face preprocessor with VGGFace2-like settings
        
        Args:
            min_face_size: Minimum face size to detect
            detection_confidence: Minimum confidence for face detection
            target_size: Target size for face images (default: VGGFace2 size)
        """
        self.detector = MTCNN(min_face_size=min_face_size)
        self.detection_confidence = detection_confidence
        self.target_size = target_size
        
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Detect and align face using MTCNN (similar to VGGFace2 preprocessing)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (aligned_face, bbox) or None if no face detected
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Detect faces
        detections = self.detector.detect_faces(image)
        
        if not detections:
            return None
            
        # Get the face with highest confidence
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        if best_detection['confidence'] < self.detection_confidence:
            return None
            
        # Get face bounding box
        bbox = best_detection['box']
        x, y, w, h = bbox
        
        # Get facial landmarks
        landmarks = best_detection['keypoints']
        
        # Align face using landmarks
        aligned_face = self._align_face(image, landmarks)
        
        return aligned_face, bbox
        
    def _align_face(self, image: np.ndarray, landmarks: dict) -> np.ndarray:
        """
        Align face using facial landmarks (similar to VGGFace2 alignment)
        
        Args:
            image: Input image
            landmarks: Dictionary of facial landmarks
            
        Returns:
            Aligned face image
        """
        # Get eye centers
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        
        # Calculate angle for alignment
        eye_angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        # Get center point
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, eye_angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(image, rotation_matrix, 
                                (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_CUBIC)
        
        # Resize to target size
        aligned = cv2.resize(aligned, self.target_size)
        
        return aligned
        
    def preprocess_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Complete face preprocessing pipeline
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed face image or None if no face detected
        """
        # Detect and align face
        result = self.detect_face(image)
        if result is None:
            return None
            
        aligned_face, _ = result
        
        # Convert to RGB
        if len(aligned_face.shape) == 2:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)
            
        # Normalize pixel values
        aligned_face = aligned_face.astype(np.float32) / 255.0
        
        return aligned_face 
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
    
    def detect_faces(self, image):
        """
        Detect faces in an image using MTCNN
        Args:
            image: numpy array or PIL Image
        Returns:
            list of face boxes (x, y, width, height)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        faces = self.detector.detect_faces(np.array(image))
        return faces
    
    def align_face(self, image, face_box, desired_size=(160, 160)):
        """
        Align face using facial landmarks
        Args:
            image: numpy array or PIL Image
            face_box: face box from MTCNN
            desired_size: output size of aligned face
        Returns:
            aligned face image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Extract face box coordinates
        x, y, width, height = face_box['box']
        
        # Ensure we have the required keypoints
        if 'keypoints' not in face_box or 'left_eye' not in face_box['keypoints'] or 'right_eye' not in face_box['keypoints']:
            # If keypoints are missing, use simple cropping
            face = np.array(image)[y:y+height, x:x+width]
            return cv2.resize(face, desired_size)
        
        left_eye = face_box['keypoints']['left_eye']
        right_eye = face_box['keypoints']['right_eye']
        
        # Calculate angle for alignment
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate desired right eye position
        desired_left_eye = (0.35, 0.35)
        desired_right_eye = (1 - desired_left_eye[0], desired_left_eye[1])
        
        # Calculate scale
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * desired_size[0]
        scale = desired_dist / dist
        
        # Calculate center point
        center = ((left_eye[0] + right_eye[0]) // 2,
                 (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Update translation
        tX = desired_size[0] * 0.5
        tY = desired_size[1] * desired_left_eye[1]
        M[0, 2] += (tX - center[0])
        M[1, 2] += (tY - center[1])
        
        # Apply affine transform
        aligned_face = cv2.warpAffine(np.array(image), M, desired_size,
                                    flags=cv2.INTER_CUBIC)
        
        return aligned_face
    
    def preprocess_face(self, image, desired_size=(160, 160)):
        """
        Detect, align and preprocess face
        Args:
            image: numpy array or PIL Image
            desired_size: output size of aligned face
        Returns:
            preprocessed face image or None if no face detected
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Use the first detected face
        face_box = faces[0]
        
        # Check if we have a valid face box
        if 'box' not in face_box:
            return None
            
        aligned_face = self.align_face(image, face_box, desired_size)
        
        # Convert to RGB
        if len(aligned_face.shape) == 2:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)
        elif aligned_face.shape[2] == 4:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGBA2RGB)
        
        # Normalize pixel values
        aligned_face = aligned_face.astype('float32')
        aligned_face = (aligned_face - 127.5) / 128.0
        
        return aligned_face 
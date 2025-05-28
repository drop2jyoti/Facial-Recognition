# Facial Recognition System Using FaceNet

A real-time facial recognition system built with FastAPI, FaceNet, and Redis. This system can detect, align, and identify faces from images, producing 128-dimensional embeddings for each face and matching them against a database for verification or identification.

## Features

- Face detection and alignment using MTCNN
- Face embedding generation using FaceNet (ResNet50 backbone)
- FastAPI-based REST API
- Redis-based embedding storage
- Docker containerization
- Support for face registration, verification, and identification

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (if running locally)
- Redis (handled by Docker)

## Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/drop2jyoti/Facial-Recognition.git
cd Facial-Recognition
```

2. Build and start the containers:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Register a Face
Register a new face in the system.

```bash
curl -X POST "http://localhost:8000/register?user_id=USER_ID" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/face/image.jpg"
```

Replace:
- `USER_ID`: Unique identifier for the person
- `/path/to/face/image.jpg`: Path to the face image file

### 2. Verify a Face
Verify if a face matches a registered user.

```bash
curl -X POST "http://localhost:8000/verify?user_id=USER_ID" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/face/image.jpg"
```

### 3. Identify a Face
Identify a face from all registered users.

```bash
curl -X POST "http://localhost:8000/identify" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/face/image.jpg"
```

### 4. Debug Endpoints

List all registered users:
```bash
curl http://localhost:8000/debug/registered-users
```

## Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
REDIS_HOST=localhost
REDIS_PORT=6379
MODEL_PATH=models/facenet_weights.pth
FACE_DETECTION_CONFIDENCE=0.9
FACE_MATCHING_THRESHOLD=0.7
```

4. Run the application:
```bash
python src/app.py
```

## Project Structure

```
Facial-Recognition/
├── src/
│   ├── app.py              # FastAPI application
│   │   └── facenet.py      # FaceNet model implementation
│   ├── utils/
│   │   └── face_detection.py  # Face detection and alignment
│   └── database/
│       └── embedding_store.py  # Redis-based embedding storage
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Configuration

The system can be configured through environment variables:

- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `MODEL_PATH`: Path to FaceNet model weights
- `FACE_DETECTION_CONFIDENCE`: Minimum confidence for face detection (default: 0.9)
- `FACE_MATCHING_THRESHOLD`: Similarity threshold for face matching (default: 0.7)

## Troubleshooting

1. If face detection fails:
   - Ensure the image contains a clear, front-facing face
   - Check if the image is properly lit
   - Try adjusting the `FACE_DETECTION_CONFIDENCE` threshold

2. If face matching is too strict/loose:
   - Adjust the `FACE_MATCHING_THRESHOLD` value
   - Higher values (closer to 1.0) make matching stricter
   - Lower values make matching more lenient

3. If Redis connection fails:
   - Check if Redis is running
   - Verify Redis host and port settings
   - Check Redis connection logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


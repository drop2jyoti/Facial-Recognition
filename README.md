# Facial Recognition System

A robust facial recognition system built with FastAPI, PyTorch, and Redis. This system provides face registration, verification, and identification capabilities with a secure API interface.

## Features

- Face registration and verification
- Face identification against registered faces
- Secure API with key-based authentication
- Rate limiting to prevent abuse
- Redis-based embedding storage
- Docker support for easy deployment
- Pre-trained FaceNet model for accurate face recognition
- Face detection and preprocessing pipeline
- Debug endpoints for system monitoring

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Redis (included in Docker setup)
- API Key (generated using the provided script)

## Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-recognition.git
cd facial-recognition
```

2. Generate an API key:
```bash
python scripts/generate_api_key.py
```
This will create a `.env` file with your API key.

3. Start the services:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

## API Endpoints

All endpoints require an API key to be passed in the `X-API-Key` header.

### Register a Face
```bash
curl -X POST "http://localhost:8000/register?user_id=user123" \
  -H "accept: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/face/image.jpg"
```

### Verify a Face
```bash
curl -X POST "http://localhost:8000/verify?user_id=user123" \
  -H "accept: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/face/image.jpg"
```

### Identify a Face
```bash
curl -X POST "http://localhost:8000/identify" \
  -H "accept: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/face/image.jpg"
```

### Debug Endpoints

#### List Registered Users
```bash
curl "http://localhost:8000/debug/registered-users" \
  -H "X-API-Key: YOUR_API_KEY"
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

## Rate Limiting

The API implements rate limiting to prevent abuse:
- Register endpoint: 5 requests per minute
- Verify endpoint: 10 requests per minute
- Identify endpoint: 10 requests per minute
- Debug endpoints: 30 requests per minute

## Environment Variables

Create a `.env` file in the project root with the following variables:
```
API_KEY=your_generated_api_key
REDIS_HOST=redis
REDIS_PORT=6379
MODEL_PATH=models/facenet_weights.pth
FACE_DETECTION_CONFIDENCE=0.9
FACE_MATCHING_THRESHOLD=0.7
```

## Project Structure

```
facial-recognition/
├── src/
│   ├── app.py              # FastAPI application
│   ├── models/
│   │   └── facenet.py      # FaceNet model implementation
│   ├── utils/
│   │   ├── face_detection.py
│   │   └── face_preprocessing.py
│   └── database/
│       └── embedding_store.py
├── scripts/
│   └── generate_api_key.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Security Features

- API key authentication for all endpoints
- Rate limiting to prevent abuse
- Secure Redis connection
- Input validation and sanitization
- Error handling and logging

## Troubleshooting

1. **API Key Issues**
   - Ensure the `.env` file is properly copied into the Docker container
   - Verify the API key is correctly set in the `.env` file
   - Check the API key is being passed in the `X-API-Key` header

2. **Face Detection Issues**
   - Ensure the image contains a clear, front-facing face
   - Check the image format (supported: JPG, PNG)
   - Verify the image size and quality

3. **Redis Connection Issues**
   - Check if Redis container is running: `docker-compose ps`
   - Verify Redis connection settings in `.env`
   - Check Redis logs: `docker-compose logs redis`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


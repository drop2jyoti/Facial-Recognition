# Facial Recognition System

A robust facial recognition system built with FastAPI, PyTorch, and Redis. This system provides face registration, verification, and identification capabilities through a secure API and a user-friendly web interface.

## Features

- Face registration, verification, and identification via API and Web UI
- Secure API with key-based authentication
- Rate limiting to prevent abuse
- Redis-based embedding storage
- Docker support for easy deployment
- Pre-trained FaceNet model for accurate face recognition
- Face detection and preprocessing pipeline
- Debug endpoints for system monitoring
- User-friendly web interface for easy interaction

## System Design

Here is a high-level diagram illustrating the architecture of the system:

![System Architecture](system-design.png)

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
This will build the Docker image, start the FastAPI application service and the Redis service.

4. Access the Web Interface:
Open your browser and go to `http://localhost:8000`.

You will be prompted to enter the API key. Use the key generated in step 2.

## Using the Web Interface

The web interface at `http://localhost:8000` allows you to:

- **Register Face:** Register a new face with a user ID.
- **Verify Face:** Verify if a face matches a previously registered user ID.
- **Identify Face:** Identify a face by comparing it against all registered faces.
- **List Registered Users:** View a list of all registered user IDs, with options to view details and unregister users.

## API Endpoints

All endpoints require an API key to be passed in the `X-API-Key` header. Refer to the web interface's network requests in your browser's developer tools for examples of API calls.

### Register a Face
`POST /register?user_id={user_id}` with `multipart/form-data` (file: image)

### Verify a Face
`POST /verify?user_id={user_id}` with `multipart/form-data` (file: image)

### Identify a Face
`POST /identify` with `multipart/form-data` (file: image)

### Debug Endpoints

#### List Registered Users
`GET /debug/registered-users`

#### Health Check
`GET /health`

## Purging Redis Data

To clear all registered face data from the Redis database, you can use the `FLUSHALL` command within the Redis container:

```bash
docker-compose exec redis redis-cli FLUSHALL
```

Ensure your Docker services are running (`docker-compose up -d`) before running this command.

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
MODEL_PATH=/app/models/facenet_keras.h5
FACE_DETECTION_CONFIDENCE=0.9 # Optional, default 0.9
FACE_MATCHING_THRESHOLD=0.7   # Optional, default 0.7
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
│   └── static/
│       ├── index.html      # Web interface HTML
│       ├── css/
│       │   └── styles.css  # Web interface CSS
│       └── js/
│           └── app.js      # Web interface JavaScript
├── scripts/
│   └── generate_api_key.py # API key generation script
├── Dockerfile              # Docker build instructions
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project README
```

## Security Features

- API key authentication for all endpoints
- Rate limiting to prevent abuse
- Secure Redis connection (within Docker network)
- Input validation and sanitization
- Error handling and logging

## Troubleshooting

1. **API Key Issues**
   - Ensure the `.env` file is present and correct in the project root.
   - Verify the API key is correctly set in the `.env` file.
   - Check the API key is being entered correctly in the web interface or passed in the `X-API-Key` header for API calls.

2. **Face Processing Issues (e.g., "Could not process face...")**
   - Ensure the uploaded image contains a clear, front-facing face.
   - Check the image is not too small, blurry, or has poor lighting.
   - Verify the image format (supported: JPG, PNG).

3. **Redis Connection Issues**
   - Check if Redis container is running: `docker-compose ps`
   - Verify Redis host and port settings in `.env` match `docker-compose.yml`.
   - Check Redis container logs: `docker-compose logs redis`

## Future Enhancements

Here are some potential areas for future improvement and enhancements to the Facial Recognition System:

*   **Improved Face Processing:**
    *   Explore using more modern or specialized face detection models.
    *   Implement checks for face quality (e.g., sharpness, contrast, pose).
    *   Enhance the system to handle multiple faces in an image.

*   **Scalable Embedding Storage and Search:**
    *   Integrate a dedicated vector database (like Milvus, Pinecone, Chroma, or Redis Stack) for efficient storage and search of large numbers of embeddings.
    *   Implement proper database indexing strategies.

*   **Enhanced Security:**
    *   Implement a more sophisticated authentication and authorization system (e.g., OAuth2, JWT).
    *   Ensure sensitive credentials are managed securely in production environments.
    *   Explore techniques to mitigate adversarial attacks.

*   **Improved Web Interface:**
    *   Add visual feedback (e.g., bounding boxes) during face detection/preprocessing.
    *   Implement real-time face processing from the camera feed.
    *   Provide more detailed progress indicators for operations.
    *   Add UI elements for managing registered users (viewing details, unregistering). (Completed)
    *   Allow users to crop or adjust the detected face region.

*   **Robustness and Error Handling:**
    *   Expand logging to capture more detailed information.
    *   Add more rigorous backend input validation.
    *   Implement mechanisms for graceful degradation.

*   **Performance Optimization:**
    *   Explore model quantization or pruning.
    *   Offload computationally intensive tasks to background worker queues.

*   **Testing and Code Quality:**
    *   Add comprehensive unit and integration tests.
    *   Set up a CI/CD pipeline.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy .env file
COPY .env .

# Create necessary directories
RUN mkdir -p models

# Set environment variables
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MODEL_PATH=models/facenet_weights.pth
ENV FACE_DETECTION_CONFIDENCE=0.9
ENV FACE_MATCHING_THRESHOLD=0.7

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "src/app.py"] 
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configure TensorFlow to use memory efficiently
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# Copy the model file first to leverage Docker caching
COPY Model /app/Model/

# Copy application code
COPY api_only.py ml_api.py README.md /app/

# Set environment variables
ENV PORT=8089

# Expose the port
EXPOSE 8089

# Set health check to ensure app is ready before receiving traffic
HEALTHCHECK --interval=5s --timeout=30s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8089/health || exit 1

# Command to run the API-only version by default
CMD ["python", "api_only.py"] 
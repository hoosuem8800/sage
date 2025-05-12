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

# Copy the model file first to leverage Docker caching
COPY Model /app/Model/

# Copy application code
COPY api_only.py ml_api.py app.py README.md /app/

# Set environment variables
ENV PORT=8089

# Expose the port
EXPOSE 8089

# Command to run the API-only version by default
# (This can be overridden by setting a different CMD in the Space settings)
CMD ["python", "api_only.py"] 
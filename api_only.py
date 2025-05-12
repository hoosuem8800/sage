from fastapi import FastAPI, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from typing import Dict, Union, Any, List
import gc
import uvicorn
import asyncio
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set TensorFlow to use memory growth to avoid OOM issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except Exception as e:
        logger.error(f"Error configuring GPUs: {str(e)}")
else:
    logger.info("No GPUs found, using CPU mode")

app = FastAPI(title="Chest X-ray Classification API",
             description="API for classifying chest X-rays (Normal, Lung Opacity, Pneumonia)",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global model variable
model = None
model_loading = False
model_load_started = False
model_ready = False
start_time = time.time()

# Class names
CLASS_NAMES = ["Lung_Opacity", "Normal", "Pneumonia"]

# Define a function to load the model in a separate thread
def load_model_in_thread():
    global model, model_loading, model_ready
    try:
        logger.info(f"Starting model loading in background thread at {time.time() - start_time:.2f}s after startup")
        model_path = os.path.join(os.path.dirname(__file__), "Model", "classification_cnn.h5")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return

        logger.info(f"Loading model from {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Run a warmup prediction
        logger.info("Running warmup prediction")
        dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
        loaded_model.predict(dummy_input, verbose=0)
        
        # Only set the global model after successful loading
        model = loaded_model
        model_ready = True
        logger.info(f"Model loaded successfully and ready for predictions at {time.time() - start_time:.2f}s after startup")
    except Exception as e:
        logger.error(f"Error loading model in background: {str(e)}")
    finally:
        model_loading = False

def load_model_if_needed():
    global model, model_loading, model_load_started, model_ready
    
    if model is not None and model_ready:
        return model
    
    if not model_load_started:
        model_loading = True
        model_load_started = True
        threading.Thread(target=load_model_in_thread, daemon=True).start()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model loading has started, please try again in a few seconds"
        )
    
    if model_loading:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading, please try again in a few seconds"
        )
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model failed to load, please check logs"
        )
    
    return model

@app.on_event("startup")
async def startup_event():
    global model_loading, model_load_started, start_time
    logger.info("Application startup: Initializing model loading")
    # Start model loading in a separate thread to not block the API
    model_loading = True
    model_load_started = True
    start_time = time.time()
    threading.Thread(target=load_model_in_thread, daemon=True).start()
    logger.info("Application startup complete, model loading in background")

@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Union[str, List[str]]]:
    """Root endpoint for API health check."""
    return {
        "status": "online", 
        "message": "Chest X-ray Classification API is running", 
        "model_status": "loaded" if model_ready else "loading",
        "uptime": f"{time.time() - start_time:.2f}s",
        "classes": CLASS_NAMES
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Simple health check endpoint that doesn't load the model.
    Railway uses this to verify the application is running.
    """
    # Always return healthy to allow the application to fully start
    return {
        "status": "healthy",
        "message": "API is running",
        "model_status": "loaded" if model_ready else "loading",
        "uptime": f"{time.time() - start_time:.2f}s"
    }

@app.get("/model-status", status_code=status.HTTP_200_OK)
async def model_status() -> Dict[str, Any]:
    """Get the current status of the model loading process."""
    global model, model_loading, model_ready, start_time
    
    return {
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "model_ready": model_ready,
        "uptime": f"{time.time() - start_time:.2f}s"
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Process chest X-ray image and classify as Normal, Lung Opacity, or Pneumonia.
    
    Args:
        file (UploadFile): The uploaded X-ray image file
        
    Returns:
        Dict with classification results
    
    Raises:
        HTTPException: If file upload or processing fails
    """
    # Validate file type
    valid_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type not supported. Must be one of: {', '.join(valid_types)}"
        )
    
    try:
        # Check if model is available
        if not model_ready or model is None:
            if not model_load_started:
                # Start loading the model if it hasn't started yet
                background_tasks.add_task(load_model_if_needed)
                
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready yet, please try again in a few seconds"
            )
        
        # Read and process image
        contents = await file.read()
        
        try:
            # Decode image in grayscale mode
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image file. Could not decode the image."
                )
            
            # Get image dimensions
            orig_height, orig_width = img.shape
            logger.info(f"Processing image: {orig_width}x{orig_height}px")
                
            # Resize to target size
            img = cv2.resize(img, (128, 128))
            
            # Reshape and normalize
            img = img.reshape(1, 128, 128, 1).astype('float32') / 255.0
            
            # Free up memory
            del contents
            gc.collect()
            
            # Make prediction
            prediction = model.predict(img, verbose=0)
            
            # Get the class with highest probability
            class_idx = np.argmax(prediction[0])
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
            probability = float(prediction[0][class_idx])
            
            # Create class probabilities dict
            class_probabilities = {}
            for i, name in enumerate(CLASS_NAMES):
                if i < len(prediction[0]):
                    class_probabilities[name] = float(prediction[0][i])
            
            # Free up memory
            del img
            gc.collect()
            
            # Return result
            return {
                "diagnosis": class_name,
                "confidence": probability * 100,  # Convert to percentage
                "class_probabilities": class_probabilities,
                "image_size": f"{orig_width}x{orig_height}"
            }
            
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing the image: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing the request: {str(e)}"
        )

@app.get("/unload-model", status_code=status.HTTP_200_OK)
async def unload_model() -> Dict[str, str]:
    """Unload the model from memory"""
    global model, model_ready
    if model is not None:
        del model
        model = None
        model_ready = False
        # Force garbage collection
        gc.collect()
        logger.info("Model unloaded from memory")
        return {"status": "success", "message": "Model unloaded from memory"}
    else:
        return {"status": "success", "message": "Model was not loaded"}

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8089))
    logger.info(f"Starting server on port {port}")
    # Use uvicorn with additional restart and timeout settings
    try:
        logger.info(f"Starting FastAPI server on http://0.0.0.0:{port}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            timeout_keep_alive=120
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise 
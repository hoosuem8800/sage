from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from typing import Dict, Union, Any, List
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Class names
CLASS_NAMES = ["Lung_Opacity", "Normal", "Pneumonia"]

def load_model_if_needed():
    """Load the model if it's not already loaded"""
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "Model", "classification_cnn.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Run a warmup prediction
        dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        logger.info("Model warmup complete")
    
    return model

@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Union[str, List[str]]]:
    """Root endpoint for API health check."""
    return {
        "status": "online", 
        "message": "Chest X-ray Classification API is running", 
        "classes": CLASS_NAMES
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check if model file exists
        model_path = os.path.join(os.path.dirname(__file__), "Model", "classification_cnn.h5")
        model_file_exists = os.path.exists(model_path)
        
        # Check if model is loaded
        model_loaded = model is not None
        
        return {
            "status": "healthy" if model_file_exists else "unhealthy",
            "model_file_exists": model_file_exists,
            "model_loaded": model_loaded,
            "classes": CLASS_NAMES
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile) -> Dict[str, Any]:
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
        # Load model if not already loaded
        try:
            model = load_model_if_needed()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {str(e)}"
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
    global model
    if model is not None:
        del model
        model = None
        # Force garbage collection
        gc.collect()
        logger.info("Model unloaded from memory")
        return {"status": "success", "message": "Model unloaded from memory"}
    else:
        return {"status": "success", "message": "Model was not loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
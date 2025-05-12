from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
import threading
import time
import uvicorn
import sys
import importlib.util

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
start_time = time.time()
main_app_ready = False
model_ready = False

# Import the API endpoints directly
from api_only import root as api_root
from api_only import health_check as api_health
from api_only import model_status as api_model_status
from api_only import predict as api_predict
from api_only import unload_model as api_unload_model
from api_only import load_model_in_thread, model, model_ready

# Include the endpoints from api_only.py
app.get("/")(api_root)
app.get("/model-status")(api_model_status)
app.post("/predict")(api_predict)
app.get("/unload-model")(api_unload_model)

# Add a GET handler for /predict to provide better error messages
@app.get("/predict")
async def predict_get_handler():
    """Helper endpoint to explain that /predict requires POST method"""
    return JSONResponse(
        status_code=405,
        content={
            "detail": "Method Not Allowed. The /predict endpoint requires a POST request with an X-ray image file.",
            "usage": "Send a POST request with Content-Type: multipart/form-data and include a 'file' field containing your chest X-ray image.",
            "example": "curl -X POST -F 'file=@chest_xray.jpg' https://this-api-url.com/predict"
        }
    )

# Override health check to ensure we always return 200 status
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint that always returns healthy.
    Railway uses this to verify the application is running.
    """
    return {
        "status": "healthy",
        "message": "API is running",
        "model_status": "loaded" if model_ready else "loading",
        "uptime": f"{time.time() - start_time:.2f}s"
    }

# Error handler for 503 errors to return 200 to health checks
@app.middleware("http")
async def handle_503_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        # Return a JSON response with error details
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
        )

@app.on_event("startup")
async def startup_event():
    global main_app_ready
    logger.info("Application startup: Initializing model loading")
    # Start model loading in a separate thread to not block the API
    threading.Thread(target=load_model_in_thread, daemon=True).start()
    main_app_ready = True
    logger.info("Application startup complete, model loading in background")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8089))
    logger.info(f"Starting server on port {port}")
    
    try:
        logger.info(f"Starting FastAPI server on http://0.0.0.0:{port}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise 
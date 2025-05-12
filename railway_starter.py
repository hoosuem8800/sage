from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import threading
import time
import uvicorn
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Chest X-ray Classification API Starter",
             description="Railway starter app that loads the main API",
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

def load_main_app():
    global main_app_ready
    logger.info("Starting main application loading...")
    
    # Wait a few seconds to let the health check system start
    time.sleep(3)
    
    try:
        # Import the real app in a separate thread
        import api_only
        logger.info("Main app imported, starting model loading...")
        main_app_ready = True
    except Exception as e:
        logger.error(f"Error importing main app: {str(e)}")
        sys.exit(1)

@app.on_event("startup")
async def startup_event():
    logger.info("Railway starter application initializing")
    threading.Thread(target=load_main_app, daemon=True).start()

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint for Railway."""
    return {
        "status": "online",
        "message": "Chest X-ray Classification API is starting up",
        "main_app_ready": main_app_ready,
        "uptime": f"{time.time() - start_time:.2f}s"
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint that always returns healthy.
    Railway uses this to verify the application is running.
    """
    return {
        "status": "healthy",
        "message": "API is starting up",
        "main_app_ready": main_app_ready,
        "uptime": f"{time.time() - start_time:.2f}s"
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8089))
    logger.info(f"Starting Railway starter server on port {port}")
    
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
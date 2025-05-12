import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from typing import Dict, Tuple, Union, List
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def predict_xray(image):
    """Process an X-ray image and classify it"""
    if image is None:
        return {
            "error": "No image provided"
        }
    
    try:
        # Load model if needed
        model = load_model_if_needed()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Get original dimensions
        orig_height, orig_width = image.shape
        logger.info(f"Processing image: {orig_width}x{orig_height}px")
        
        # Resize to target size
        image_resized = cv2.resize(image, (128, 128))
        
        # Reshape and normalize
        image_processed = image_resized.reshape(1, 128, 128, 1).astype('float32') / 255.0
        
        # Make prediction
        prediction = model.predict(image_processed, verbose=0)
        
        # Get the class with highest probability
        class_idx = np.argmax(prediction[0])
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
        probability = float(prediction[0][class_idx])
        
        # Create class probabilities
        class_probabilities = {}
        for i, name in enumerate(CLASS_NAMES):
            if i < len(prediction[0]):
                class_probabilities[name] = float(prediction[0][i])
        
        # Generate the confidence label
        confidence_label = f"{class_name} ({probability*100:.1f}%)"
        
        # Create a dictionary of results to display
        results = {
            "diagnosis": class_name,
            "confidence": f"{probability*100:.1f}%",
            "image_size": f"{orig_width}x{orig_height}"
        }
        
        # Create detailed results for each class
        detailed_results = ""
        for name, prob in class_probabilities.items():
            detailed_results += f"{name}: {prob*100:.1f}%\n"
        
        return confidence_label, detailed_results, results
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}", "", {"error": str(e)}

# Create Gradio interface
with gr.Blocks(title="Chest X-ray Analysis") as demo:
    gr.Markdown("# Chest X-ray Analysis")
    gr.Markdown("Upload a chest X-ray image to classify it as Normal, Lung Opacity, or Pneumonia.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload X-ray Image")
            submit_btn = gr.Button("Analyze X-ray", variant="primary")
        
        with gr.Column():
            diagnosis = gr.Label(label="Diagnosis")
            details = gr.Textbox(label="Detailed Results", lines=5)
            json_output = gr.JSON(label="Full Results")
    
    submit_btn.click(
        fn=predict_xray,
        inputs=[input_image],
        outputs=[diagnosis, details, json_output]
    )
    
    gr.Markdown("## About this Model")
    gr.Markdown("""
    This model analyzes chest X-ray images to detect:
    - Normal (healthy lungs)
    - Lung Opacity (general abnormalities in the lung fields)
    - Pneumonia (specific findings consistent with pneumonia)
    
    *Note: This model is for demonstration purposes and should not be used for clinical diagnosis.*
    """)

# Launch the app
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port) 
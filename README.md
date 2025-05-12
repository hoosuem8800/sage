---
title: Chopper Chest X-ray Analysis
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: latest
app_file: api_only.py
pinned: false
license: mit
---

# Chopper Chest X-ray Analysis

This model analyzes chest X-ray images to detect conditions including:
- Normal (healthy lungs)
- Lung Opacity (general abnormalities in the lung fields)
- Pneumonia (specific findings consistent with pneumonia)

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input: Grayscale chest X-ray images
- Output: Classification into three categories with confidence scores
- Image preprocessing: Resizing to 128x128px, normalization

## Deployment Instructions

### Using Docker (Recommended)

This model is configured to use Docker for deployment on Hugging Face Spaces:

1. In your Space settings, ensure:
   - SDK is set to "Docker"
   - SDK version is set to "latest"
   - App file is set to "api_only.py"

2. The Dockerfile will:
   - Use Python 3.9 (more stable with TensorFlow)
   - Install all necessary dependencies
   - Run the FastAPI server on port 7860

### Alternative Deployment Methods

You can also use one of these options:

1. **FastAPI only**: Set `app_file` to `api_only.py` and SDK to "FastAPI"
2. **Gradio interface**: Set `app_file` to `app.py` and SDK to "Gradio"

## Usage

### API Integration

You can integrate this model with your own applications using the REST API:

```python
import requests

# Replace with your actual Space URL
url = "https://hoosuem-CHEST_model.hf.space/predict"

# Prepare the image file
files = {'file': open('chest_xray.jpg', 'rb')}

# Send request
response = requests.post(url, files=files)
print(response.json())
```

Example response:
```json
{
  "diagnosis": "Normal",
  "confidence": 98.5,
  "class_probabilities": {
    "Lung_Opacity": 0.01,
    "Normal": 0.985,
    "Pneumonia": 0.005
  }
}
```

## Integration with Chopper Healthcare Platform

This model is used by the Chopper Healthcare platform to provide automated analysis of patient X-rays. The backend sends X-ray images to this model and receives diagnostic suggestions that are then reviewed by healthcare professionals.

## Troubleshooting

If you encounter TensorFlow compatibility issues:

1. Check the logs in the "Factory" tab of your Space
2. Try using Docker deployment instead of direct SDK deployment
3. Modify the Dockerfile to use a different Python or TensorFlow version if needed

## Model Limitations

This model is intended to assist healthcare professionals and should not be used as the sole basis for diagnosis. All predictions should be verified by qualified medical personnel. 
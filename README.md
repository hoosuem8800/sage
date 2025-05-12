---
title: Chopper Chest X-ray Analysis API
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: latest
app_file: api_only.py
pinned: false
license: mit
---

# Chopper Chest X-ray Analysis API

A machine learning API for analyzing chest X-ray images and classifying them into three categories:
- **Normal**: Healthy lung appearance
- **Lung Opacity**: General abnormalities in the lung fields
- **Pneumonia**: Specific findings consistent with pneumonia

## Model Details

- **Architecture**: Convolutional Neural Network (CNN) built with TensorFlow
- **Input**: Grayscale chest X-ray images (any resolution, will be resized)
- **Output**: Classification with confidence scores for each category
- **Performance**: ~92% accuracy on test datasets
- **Processing**: Images resized to 128x128px and normalized before inference

## Deployment Options

### Railway (Recommended)

This model is optimized for deployment on Railway:

1. Push your code to GitHub
2. Create a new Railway project from your GitHub repository
3. Set the source directory to `DL_model`
4. The deployment will automatically use:
   - The Dockerfile for containerization
   - railway.json for configuration
   - railway_starter.py as the entry point

For detailed Railway deployment instructions, see [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md).

### Docker (Local or Other Cloud)

1. Build the Docker image:
   ```bash
   cd DL_model
   docker build -t chest-xray-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8089:8089 chest-xray-api
   ```

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API:
   ```bash
   python railway_starter.py  # For production-like setup with health checks
   # OR
   python api_only.py         # For direct API access
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check endpoint |
| `/model-status` | GET | Check the status of model loading |
| `/predict` | POST | Submit an X-ray image for analysis |
| `/unload-model` | GET | Free memory by unloading the model |

## Usage Examples

### Python Client

```python
import requests

# Replace with your deployed API URL
api_url = "https://your-railway-deployment-url.up.railway.app"

# Check if model is ready
status = requests.get(f"{api_url}/model-status").json()
if not status.get("model_ready", False):
    print("Model is still loading, please wait...")

# Prepare the image file
files = {'file': open('chest_xray.jpg', 'rb')}

# Send request
response = requests.post(f"{api_url}/predict", files=files)
result = response.json()
print(f"Diagnosis: {result['diagnosis']} with {result['confidence']:.1f}% confidence")
```

### cURL Command

```bash
curl -X POST -F "file=@chest_xray.jpg" https://your-railway-deployment-url.up.railway.app/predict
```

### Example Response

```json
{
  "diagnosis": "Normal",
  "confidence": 98.5,
  "class_probabilities": {
    "Lung_Opacity": 0.01,
    "Normal": 0.985,
    "Pneumonia": 0.005
  },
  "image_size": "1024x1024"
}
```

## Integration with Chopper Healthcare Platform

This model serves as the AI component of the Chopper Healthcare platform:

1. Healthcare providers upload X-rays through the React frontend
2. The Django backend sends images to this API 
3. Results are displayed to healthcare professionals for review
4. All predictions are verified by qualified medical personnel

## Performance Considerations

- First request after deployment may take 5-10 seconds while the model loads
- Subsequent requests typically process in under 1 second
- Memory usage: ~500MB with the TensorFlow model loaded
- Optimized to run on CPU (no GPU required)

## Model Limitations

This model is intended as a diagnostic aid only and should not be used as the sole basis for clinical decisions. The model may not detect all abnormalities and works best with standard PA/AP chest X-ray views.

## License

MIT License 
# DL_model Deployment Guide

This guide provides instructions for deploying the Chest X-ray Analysis model to either Railway or Hugging Face Spaces.

## Option 1: Deploy to Railway (Recommended)

Railway is a platform that makes it easy to deploy containerized applications.

### Prerequisites
- [Railway CLI](https://docs.railway.app/develop/cli)
- Railway account

### Deployment Steps

1. **Login to Railway**
   ```bash
   railway login
   ```

2. **Initialize Railway project**
   ```bash
   cd DL_model
   railway init
   ```

3. **Deploy the application**
   ```bash
   railway up
   ```

4. **Configure environment variables** (if needed)
   - In the Railway dashboard, navigate to your project
   - Go to the "Variables" tab
   - Add any necessary environment variables

5. **Generate a public domain**
   - In the Railway dashboard, navigate to your project
   - Go to the "Settings" tab
   - Click "Generate Domain"

Your ML model should now be accessible via the generated domain. The API will be available at the following endpoints:
- Health check: `https://your-domain.railway.app/health`
- Prediction: `https://your-domain.railway.app/predict`

## Option 2: Deploy to Hugging Face Spaces

Hugging Face Spaces provides a platform for hosting ML models with built-in UI capabilities.

### Prerequisites
- [Hugging Face account](https://huggingface.co/join)

### Deployment Steps

1. **Create a new Space**
   - Log in to Hugging Face
   - Click on "New Space"
   - Choose a name for your space
   - Select "Docker" as the SDK
   - Set the Hardware to "CPU" (or GPU if needed)

2. **Upload files**
   - Upload all the files in the DL_model directory
   - Ensure the Model directory with classification_cnn.h5 is included

3. **Configure Space settings**
   - In Space settings, ensure:
     - SDK is set to "Docker"
     - SDK version is set to "latest"
     - App file is set to "api_only.py" (for API only) or "app.py" (for Gradio UI)

4. The Space will automatically build and deploy your model.

## Option 3: Deploy to Cloud Platforms

The model can also be deployed to other cloud platforms such as Google Cloud Run, AWS ECS, or Azure Container Instances.

### General Steps

1. Build the Docker image:
   ```bash
   docker build -t chest-xray-model:latest .
   ```

2. Push to a container registry (Docker Hub, GCR, ECR, etc.)

3. Deploy using the platform-specific instructions for containerized applications

## Integrating with Your Frontend

Once deployed, you can integrate the model with your application by making HTTP requests to the `/predict` endpoint.

Example (JavaScript/React):
```javascript
const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('https://your-model-url.com/predict', {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  return result;
};
```

## Troubleshooting

- **Model Loading Issues**: Ensure the model file path is correctly set
- **Memory Errors**: If you encounter memory issues, consider using a larger instance type
- **Timeout Errors**: Increase the timeout settings in your deployment configuration
- **CORS Issues**: Ensure the CORS middleware is properly configured

## Resources

- [Railway Documentation](https://docs.railway.app/)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://www.gradio.app/docs/) 
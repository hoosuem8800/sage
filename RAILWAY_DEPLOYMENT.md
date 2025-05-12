# Railway Deployment Instructions

This document provides the steps to deploy the Chest X-ray Classification API to Railway.

## Deployment Steps

1. **Push your code to GitHub**
   Ensure your GitHub repository is up to date with the latest code.

2. **Create a new Railway Project**
   - Go to [Railway](https://railway.app/)
   - Click "New Project" 
   - Select "Deploy from GitHub repo"
   - Connect and select your GitHub repository

3. **Configure the Deployment**
   - Set the source directory to `DL_model`
   - The railway.json file should already be configured with the correct settings
   - Ensure the working directory is properly set to the DL_model folder

4. **Add Environment Variables (if needed)**
   - `PORT`: 8089 (should already be set in Dockerfile)
   - `TF_CPP_MIN_LOG_LEVEL`: 2 (should already be set in Dockerfile)

5. **Deploy the Application**
   - Railway will automatically build and deploy your application
   - The deployment might take a few minutes as it needs to build the Docker container and load the TensorFlow model

6. **Monitor the Deployment**
   - Check the logs in Railway to ensure the application starts correctly
   - The application will show "healthy" in Railway once it's fully up and running
   - Note that the TensorFlow model might take some time to load initially

## Important Architecture Changes

We've implemented a two-stage startup process to help with Railway deployment:

1. **railway_starter.py**: A lightweight FastAPI application that starts quickly and always responds to health checks
2. **api_only.py**: The main application that loads the ML model in the background

This approach ensures that Railway's health checks pass while the model is still loading, preventing premature application restarts.

## Troubleshooting

1. **"Application failed to respond" Error**
   - Check that the deployment is using the correct directory (DL_model)
   - Verify that railway_starter.py is being used as the start command
   - The railway.json file should specify `"startCommand": "python railway_starter.py"`

2. **Memory Issues**
   - We're using tensorflow-cpu for lower memory usage
   - If memory issues persist, consider upgrading your Railway plan

3. **Model Loading Time**
   - The API endpoints will return a 503 error with a message saying the model is still loading
   - The /health and / endpoints will still respond with a 200 OK status
   - Check the /model-status endpoint to monitor loading progress

## Testing Your Deployment

Once deployed, you can test the API using:

1. Health Check:
   ```
   curl https://your-railway-url.railway.app/health
   ```

2. Model Status:
   ```
   curl https://your-railway-url.railway.app/model-status
   ```

3. Classification (after model is loaded):
   ```
   curl -X POST -F "file=@chest_xray.jpg" https://your-railway-url.railway.app/predict
   ```

## Local Testing

Before deploying, you can test the API locally:

1. Start the API:
   ```
   python railway_starter.py
   ```

2. In another terminal, run the test script:
   ```
   python test_api.py
   ```

## Integration with Frontend

After successful deployment, update your React frontend to call the API endpoint:

```javascript
const API_URL = "https://your-railway-url.railway.app";

// Example function to call the API
async function classifyImage(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });
    
    if (response.status === 503) {
      // Model is still loading
      return { error: "Model is still loading. Please try again in a few moments." };
    }
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error classifying image:", error);
    return { error: error.message };
  }
}
``` 
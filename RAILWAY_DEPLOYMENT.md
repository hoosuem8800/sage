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
   - Make sure Railway uses the Dockerfile in the DL_model directory
   - Set the working directory to DL_model if needed
   - The railway.json file should already be configured with the correct settings

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

## Troubleshooting

1. **"Application failed to respond" Error**
   - This could happen if the health check times out before the model is loaded
   - The updated code should handle this by returning "healthy" immediately while loading the model in the background
   - Check the logs to see if there are any specific errors

2. **Memory Issues**
   - TensorFlow can be memory-intensive
   - We've optimized by using tensorflow-cpu and configuring memory growth
   - If memory issues persist, consider upgrading your Railway plan

3. **Model Loading Time**
   - The model might take a while to load the first time
   - The API endpoints will return a 503 error with a message saying the model is still loading
   - The /health and / endpoints will still respond with a 200 OK status

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
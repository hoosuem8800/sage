#!/usr/bin/env python3
import requests
import sys
import time
import argparse
import os
from pprint import pprint

def test_endpoint(base_url, endpoint, method="GET", data=None, files=None):
    """Test a specific API endpoint and return the result"""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    print(f"\n🔍 Testing {method} {url}")
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, data=data, files=files, timeout=30)
        else:
            print(f"Unsupported method: {method}")
            return None
            
        print(f"📊 Status code: {response.status_code}")
        try:
            data = response.json()
            print(f"📋 Response data:")
            pprint(data)
            return data
        except:
            print(f"📋 Raw response: {response.text[:200]}")
            return response.text
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def file_exists(filepath):
    return os.path.isfile(filepath) and os.access(filepath, os.R_OK)

def main():
    parser = argparse.ArgumentParser(description="Check a deployed FastAPI application")
    parser.add_argument("base_url", help="Base URL of the API (e.g., https://your-app.railway.app)")
    parser.add_argument("--image", help="Path to an X-ray image to test prediction", default=None)
    args = parser.parse_args()
    
    base_url = args.base_url
    
    print(f"🚀 Checking API at {base_url}")
    
    # Test the health endpoint
    health_data = test_endpoint(base_url, "health")
    
    if not health_data:
        print("❌ Health check failed. API might be down.")
        sys.exit(1)
    
    # Test the root endpoint
    root_data = test_endpoint(base_url, "/")
    
    # Test the model status endpoint
    model_status = test_endpoint(base_url, "model-status")
    
    if model_status and not model_status.get("model_ready", False):
        print("\n⏳ Model is still loading. Waiting for it to be ready...")
        
        for i in range(12):  # Wait up to 2 minutes
            print(f"Checking again in 10 seconds... ({i+1}/12)")
            time.sleep(10)
            model_status = test_endpoint(base_url, "model-status")
            if model_status and model_status.get("model_ready", False):
                print("✅ Model is now ready!")
                break
    
    # First test the /predict endpoint with GET to check the API documentation
    test_endpoint(base_url, "predict", method="GET")
    
    # Test the predict endpoint with an image if provided
    prediction_tested = False
    if args.image:
        if not file_exists(args.image):
            print(f"\n❌ Image file not found or not readable: {args.image}")
            print("Please provide a valid image file path.")
        else:
            try:
                print(f"\n📝 Testing prediction with image: {args.image}")
                with open(args.image, 'rb') as img:
                    files = {'file': (os.path.basename(args.image), img, 'image/jpeg')}
                    prediction = test_endpoint(base_url, "predict", method="POST", files=files)
                    prediction_tested = True
                    
                    if prediction:
                        if isinstance(prediction, dict) and "diagnosis" in prediction:
                            print(f"\n🎉 Successfully tested prediction endpoint!")
                            print(f"📊 Diagnosis: {prediction['diagnosis']} with {prediction.get('confidence', 0):.1f}% confidence")
                        else:
                            print(f"\n⚠️ Got response from prediction endpoint, but no diagnosis found.")
                    else:
                        print(f"\n❌ Failed to get prediction.")
            except Exception as e:
                print(f"\n❌ Error testing prediction endpoint: {str(e)}")
    else:
        print("\n⚠️ No image provided for testing prediction.")
        print("Use --image parameter to test the prediction endpoint.")
        print("Example: python check_deployment.py https://your-api.railway.app --image ./chest_xray.jpg")
    
    # Overall summary
    print("\n📋 Summary:")
    print(f"- Health endpoint: {'✅ OK' if health_data else '❌ Failed'}")
    print(f"- Root endpoint: {'✅ OK' if root_data else '❌ Failed'}")
    print(f"- Model status: {'✅ Ready' if model_status and model_status.get('model_ready', False) else '⏳ Loading' if model_status else '❌ Failed'}")
    print(f"- Prediction endpoint: {'✅ Tested' if prediction_tested else '⚠️ Not tested'}")
    
    if not prediction_tested:
        print("\n📝 How to use the API:")
        print("1. The predict endpoint requires a POST request with an X-ray image")
        print("2. Example with curl:")
        print(f"   curl -X POST -F 'file=@chest_xray.jpg' {base_url}/predict")
        print("3. Example with Python:")
        print("   ```python")
        print(f"   import requests")
        print(f"   response = requests.post('{base_url}/predict', files={{'file': open('chest_xray.jpg', 'rb')}})")
        print("   print(response.json())")
        print("   ```")

if __name__ == "__main__":
    main() 
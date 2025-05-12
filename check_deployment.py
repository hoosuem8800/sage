#!/usr/bin/env python3
import requests
import sys
import time
import argparse

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
            print(f"📋 Response data: {data}")
            return data
        except:
            print(f"📋 Raw response: {response.text[:200]}")
            return response.text
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

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
    
    # Test the predict endpoint with an image if provided
    if args.image:
        try:
            with open(args.image, 'rb') as img:
                files = {'file': img}
                prediction = test_endpoint(base_url, "predict", method="POST", files=files)
                
                if prediction:
                    if "diagnosis" in prediction:
                        print(f"\n🎉 Successfully tested prediction endpoint!")
                        print(f"📊 Diagnosis: {prediction['diagnosis']} with {prediction.get('confidence', 0):.1f}% confidence")
                    else:
                        print(f"\n⚠️ Got response from prediction endpoint, but no diagnosis found.")
                else:
                    print(f"\n❌ Failed to get prediction.")
        except Exception as e:
            print(f"\n❌ Error testing prediction endpoint: {str(e)}")
    else:
        print("\n⚠️ No image provided for testing prediction. Use --image parameter to test the prediction endpoint.")
    
    # Overall summary
    print("\n📋 Summary:")
    print(f"- Health endpoint: {'✅ OK' if health_data else '❌ Failed'}")
    print(f"- Root endpoint: {'✅ OK' if root_data else '❌ Failed'}")
    print(f"- Model status: {'✅ Ready' if model_status and model_status.get('model_ready', False) else '⏳ Loading' if model_status else '❌ Failed'}")
    print(f"- Prediction endpoint: {'✅ Tested' if args.image else '⚠️ Not tested'}")

if __name__ == "__main__":
    main() 
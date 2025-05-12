import requests
import time
import sys
import os

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        print(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking health: {str(e)}")
        return False

def test_model_status(base_url):
    """Test the model status endpoint"""
    try:
        response = requests.get(f"{base_url}/model-status")
        print(f"Model status: {response.status_code}")
        if response.status_code == 200:
            print(response.json())
            return True
        return False
    except Exception as e:
        print(f"Error checking model status: {str(e)}")
        return False

def wait_for_api_to_start(base_url, max_retries=30, wait_time=2):
    """Wait for the API to start responding"""
    for i in range(max_retries):
        print(f"Attempt {i+1}/{max_retries} to connect to API...")
        if test_health_endpoint(base_url):
            print("API is responding!")
            return True
        time.sleep(wait_time)
    
    print("API failed to respond within the timeout period")
    return False

def main():
    # Default to localhost:8089 if not specified
    base_url = "http://localhost:8089"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Testing API at: {base_url}")
    
    if wait_for_api_to_start(base_url):
        # Try accessing model status
        test_model_status(base_url)
    else:
        print("Failed to connect to API")
        sys.exit(1)

if __name__ == "__main__":
    main() 
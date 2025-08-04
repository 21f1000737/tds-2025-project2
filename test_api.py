#!/usr/bin/env python3
import requests

def test_api_upload():
    url = "http://0.0.0.0:8000/api/"
    
    try:
        with open("question.txt", "rb") as file:
            files = {"file": file}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
    except FileNotFoundError:
        print("Error: question.txt file not found")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_upload()
#!/usr/bin/env python3
"""
Test script for both multipart and base64 file upload to the Model Finetuning API
"""

import requests
import base64
import json
import os

API_BASE_URL = "http://localhost:8000"

def encode_file_to_base64(file_path):
    """Encode a file to base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_multipart_upload():
    """Test multipart file upload"""
    print("\n🔄 Testing Multipart File Upload")
    print("-" * 40)
    
    csv_file_path = "sample_training_data.json"  # Using JSON file for testing
    
    if not os.path.exists(csv_file_path):
        print(f"❌ File '{csv_file_path}' not found")
        return None
    
    # Prepare form data
    files = {
        'data_file': ('training_data.json', open(csv_file_path, 'rb'), 'application/json')
    }
    
    data = {
        'model_name': 'unsloth/llama-3-8b-bnb-4bit',
        'max_seq_length': 512,
        'num_train_epochs': 1,
        'per_device_train_batch_size': 1,
        'learning_rate': 0.0001,
        'max_steps': 10,
        'warmup_steps': 2,
        'save_steps': 5,
        'output_dir': './multipart_results'
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/finetune", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multipart upload successful!")
            print(f"   Job ID: {result['job_id']}")
            print(f"   Status: {result['status']}")
            print(f"   Message: {result['message']}")
            print(f"   Dataset Info: {result['dataset_info']['total_rows']} rows")
            return result['job_id']
        else:
            print(f"❌ Multipart upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during multipart upload: {e}")
        return None
    finally:
        files['data_file'][1].close()

def test_base64_upload():
    """Test base64 JSON upload"""
    print("\n🔄 Testing Base64 JSON Upload")
    print("-" * 40)
    
    json_file_path = "sample_training_data.json"
    
    if not os.path.exists(json_file_path):
        print(f"❌ File '{json_file_path}' not found")
        return None
    
    # Encode file to base64
    try:
        base64_content = encode_file_to_base64(json_file_path)
        print(f"   File encoded to base64 ({len(base64_content)} characters)")
    except Exception as e:
        print(f"❌ Error encoding file: {e}")
        return None
    
    # Prepare JSON payload
    payload = {
        "file_content": base64_content,
        "file_type": "json",
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "max_seq_length": 512,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": 0.0001,
        "max_steps": 10,
        "warmup_steps": 2,
        "save_steps": 5,
        "output_dir": "./base64_results"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/finetune",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Base64 upload successful!")
            print(f"   Job ID: {result['job_id']}")
            print(f"   Status: {result['status']}")
            print(f"   Message: {result['message']}")
            print(f"   Dataset Info: {result['dataset_info']['total_rows']} rows")
            return result['job_id']
        else:
            print(f"❌ Base64 upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during base64 upload: {e}")
        return None

def monitor_job(job_id):
    """Monitor a training job"""
    if not job_id:
        return
    
    print(f"\n📊 Monitoring Job: {job_id}")
    print("-" * 40)
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
        if response.status_code == 200:
            job_data = response.json()
            print(f"   Status: {job_data['status']}")
            if job_data.get('error'):
                print(f"   Error: {job_data['error']}")
            if job_data.get('logs'):
                print(f"   Recent logs: {len(job_data['logs'])} entries")
        else:
            print(f"❌ Failed to get job status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error monitoring job: {e}")

def main():
    """Main test function"""
    print("🚀 Testing Hybrid Finetuning API (Multipart + Base64)")
    print("=" * 60)
    
    # Test API health
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
        else:
            print("❌ API health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on port 8000")
        return
    
    # Test both upload methods
    multipart_job_id = test_multipart_upload()
    base64_job_id = test_base64_upload()
    
    # Monitor jobs
    if multipart_job_id:
        monitor_job(multipart_job_id)
    
    if base64_job_id:
        monitor_job(base64_job_id)
    
    print("\n" + "=" * 60)
    print("🎉 Hybrid API testing completed!")
    print("\n💡 Usage Examples:")
    print("   Multipart: curl -X POST -F 'data_file=@file.csv' -F 'learning_rate=0.001' http://localhost:8000/finetune")
    print("   Base64:    curl -X POST -H 'Content-Type: application/json' -d '{\"file_content\":\"...\", \"file_type\":\"csv\"}' http://localhost:8000/finetune")
    print(f"   Dashboard: http://localhost:5000")

if __name__ == "__main__":
    main()

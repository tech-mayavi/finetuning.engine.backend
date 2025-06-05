#!/usr/bin/env python3
"""
Test script for the Model Finetuning API
"""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"

def test_api():
    """Test the finetuning API"""
    
    print("🚀 Testing Model Finetuning API")
    print("=" * 50)
    
    # Test 1: Check if API is running
    print("\n1. Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"   Response: {response.json()}")
        else:
            print("❌ API health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on port 8000")
        return
    
    # Test 2: Start a finetuning job
    print("\n2. Starting a finetuning job...")
    finetune_request = {
        "dataset_name": "test_dataset",
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "max_seq_length": 512,  # Smaller for testing
        "num_train_epochs": 1,
        "max_steps": 10,  # Very small for quick testing
        "per_device_train_batch_size": 1,
        "learning_rate": 2e-4
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/finetune",
            json=finetune_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["job_id"]
            print("✅ Finetuning job started!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {job_data['status']}")
            print(f"   Dashboard: {job_data['dashboard_url']}")
        else:
            print(f"❌ Failed to start job: {response.status_code}")
            print(f"   Error: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error starting job: {e}")
        return
    
    # Test 3: Monitor job status
    print("\n3. Monitoring job status...")
    for i in range(5):  # Check status 5 times
        try:
            response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
            if response.status_code == 200:
                status_data = response.json()
                print(f"   Check {i+1}: Status = {status_data['status']}")
                
                if status_data['status'] in ['completed', 'failed']:
                    break
                    
            time.sleep(2)  # Wait 2 seconds between checks
        except Exception as e:
            print(f"   Error checking status: {e}")
    
    # Test 4: List all jobs
    print("\n4. Listing all jobs...")
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        if response.status_code == 200:
            jobs_data = response.json()
            print(f"✅ Found {jobs_data['total']} job(s)")
            for job in jobs_data['jobs']:
                print(f"   - Job {job['id']}: {job['status']}")
        else:
            print(f"❌ Failed to list jobs: {response.status_code}")
    except Exception as e:
        print(f"❌ Error listing jobs: {e}")
    
    # Test 5: Get job logs
    print("\n5. Getting job logs...")
    try:
        response = requests.get(f"{API_BASE_URL}/logs/{job_id}")
        if response.status_code == 200:
            logs_data = response.json()
            print(f"✅ Retrieved {len(logs_data['logs'])} log entries")
            if logs_data['logs']:
                print("   Recent logs:")
                for log in logs_data['logs'][-3:]:  # Show last 3 logs
                    print(f"   - [{log['timestamp']}] {log['level']}: {log['message']}")
        else:
            print(f"❌ Failed to get logs: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting logs: {e}")
    
    # Test 6: Dashboard info
    print("\n6. Getting dashboard info...")
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard")
        if response.status_code == 200:
            dashboard_data = response.json()
            print("✅ Dashboard info retrieved")
            print(f"   URL: {dashboard_data['dashboard_url']}")
            print(f"   Status: {dashboard_data['status']}")
        else:
            print(f"❌ Failed to get dashboard info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting dashboard info: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print(f"\n💡 Tips:")
    print(f"   - Visit {API_BASE_URL}/docs for interactive API documentation")
    print(f"   - Visit http://localhost:5000 for the real-time training dashboard")
    print(f"   - Check training_logs.jsonl for detailed logs")

if __name__ == "__main__":
    test_api()

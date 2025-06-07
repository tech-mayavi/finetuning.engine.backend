#!/usr/bin/env python3
"""
Test script for the Chat API functionality
"""

import requests
import time
import json
import sys

API_BASE_URL = "https://finetune_engine.deepcite.in"

def test_chat_api():
    """Test the chat API endpoints"""
    
    print("🤖 Testing Chat API")
    print("=" * 50)
    
    # Test 1: Check if API is running
    print("\n1. Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
            data = response.json()
            print(f"   Message: {data['message']}")
            print(f"   Version: {data['version']}")
        else:
            print("❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on port 8000")
        return False
    
    # Test 2: Get available models
    print("\n2. Getting available models...")
    try:
        response = requests.get(f"{API_BASE_URL}/models/available")
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ Found {models_data['total']} available model(s)")
            
            if models_data['models']:
                for model in models_data['models']:
                    print(f"   - {model['name']} ({model['size_mb']} MB)")
                    print(f"     Path: {model['path']}")
                    if model.get('created_at'):
                        print(f"     Created: {model['created_at']}")
                
                # Use the first available model for testing
                test_model_path = models_data['models'][0]['path']
                print(f"\n   Using model for testing: {test_model_path}")
            else:
                print("⚠️  No trained models found. You need to train a model first.")
                print("   You can still test the API endpoints, but chat will fail.")
                test_model_path = "./results/lora_model"  # Default path
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            print(f"   Error: {response.text}")
            test_model_path = "./results/lora_model"  # Default path
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        test_model_path = "./results/lora_model"  # Default path
    
    # Test 3: Check model status
    print("\n3. Checking model status...")
    try:
        response = requests.get(f"{API_BASE_URL}/models/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Model status retrieved")
            print(f"   Loaded: {status_data['loaded']}")
            if status_data['loaded']:
                print(f"   Current model: {status_data['model_path']}")
            else:
                print("   No model currently loaded")
        else:
            print(f"❌ Failed to get model status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting model status: {e}")
    
    # Test 4: Load a model
    print("\n4. Loading a model...")
    try:
        load_request = {
            "model_path": test_model_path,
            "max_seq_length": 2048
        }
        
        response = requests.post(
            f"{API_BASE_URL}/models/load",
            json=load_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            load_data = response.json()
            print(f"✅ Model loaded successfully!")
            print(f"   Status: {load_data['status']}")
            print(f"   Message: {load_data['message']}")
            print(f"   Model path: {load_data['model_path']}")
            model_loaded = True
        else:
            print(f"❌ Failed to load model: {response.status_code}")
            print(f"   Error: {response.text}")
            model_loaded = False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False
    
    # Test 5: Single chat message
    if model_loaded:
        print("\n5. Testing single chat message...")
        try:
            chat_request = {
                "message": "Hello! Can you explain what machine learning is?",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{API_BASE_URL}/chat/single",
                json=chat_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                chat_data = response.json()
                print("✅ Chat response received!")
                print(f"   Status: {chat_data['status']}")
                print(f"   User: {chat_request['message']}")
                print(f"   Assistant: {chat_data['response']}")
                print(f"   Model: {chat_data['model_path']}")
            else:
                print(f"❌ Failed to get chat response: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Error in single chat: {e}")
    else:
        print("\n5. Skipping single chat test (model not loaded)")
    
    # Test 6: Conversation chat
    if model_loaded:
        print("\n6. Testing conversation chat...")
        try:
            conversation_request = {
                "messages": [
                    {"role": "user", "content": "Hi there!"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                    {"role": "user", "content": "Can you tell me about artificial intelligence?"}
                ],
                "max_tokens": 120,
                "temperature": 0.8
            }
            
            response = requests.post(
                f"{API_BASE_URL}/chat/conversation",
                json=conversation_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                conv_data = response.json()
                print("✅ Conversation response received!")
                print(f"   Status: {conv_data['status']}")
                print("   Conversation:")
                for msg in conversation_request['messages']:
                    print(f"     {msg['role'].title()}: {msg['content']}")
                print(f"     Assistant: {conv_data['response']}")
            else:
                print(f"❌ Failed to get conversation response: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Error in conversation chat: {e}")
    else:
        print("\n6. Skipping conversation chat test (model not loaded)")
    
    # Test 7: Quick chat
    if model_loaded:
        print("\n7. Testing quick chat...")
        try:
            quick_message = "What is Python programming?"
            
            response = requests.post(
                f"{API_BASE_URL}/chat/quick",
                params={"message": quick_message},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                quick_data = response.json()
                print("✅ Quick chat response received!")
                print(f"   User: {quick_data['message']}")
                print(f"   Assistant: {quick_data['response']}")
                print(f"   Model: {quick_data['model_path']}")
            else:
                print(f"❌ Failed to get quick chat response: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Error in quick chat: {e}")
    else:
        print("\n7. Skipping quick chat test (model not loaded)")
    
    # Test 8: Model unloading
    if model_loaded:
        print("\n8. Testing model unloading...")
        try:
            response = requests.post(f"{API_BASE_URL}/models/unload")
            
            if response.status_code == 200:
                unload_data = response.json()
                print("✅ Model unloaded successfully!")
                print(f"   Status: {unload_data['status']}")
                print(f"   Message: {unload_data['message']}")
            else:
                print(f"❌ Failed to unload model: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Error unloading model: {e}")
    else:
        print("\n8. Skipping model unload test (model not loaded)")
    
    # Test 9: Error handling - chat without model
    print("\n9. Testing error handling (chat without loaded model)...")
    try:
        chat_request = {
            "message": "This should fail because no model is loaded",
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat/single",
            json=chat_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            error_data = response.json()
            print("✅ Error handling works correctly!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Error: {error_data['detail']}")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error in error handling test: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Chat API testing completed!")
    
    print(f"\n💡 Usage Examples:")
    print(f"   # List available models")
    print(f"   curl {API_BASE_URL}/models/available")
    print(f"   ")
    print(f"   # Load a model")
    print(f"   curl -X POST {API_BASE_URL}/models/load \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"model_path\": \"{test_model_path}\"}}'")
    print(f"   ")
    print(f"   # Send a chat message")
    print(f"   curl -X POST {API_BASE_URL}/chat/single \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"message\": \"Hello!\", \"max_tokens\": 100}}'")
    print(f"   ")
    print(f"   # Quick chat")
    print(f"   curl -X POST '{API_BASE_URL}/chat/quick?message=Hello%20there!'")
    print(f"   ")
    print(f"   # Check API documentation")
    print(f"   Visit: {API_BASE_URL}/docs")

def interactive_chat():
    """Interactive chat session with the loaded model"""
    print("\n🗨️  Starting interactive chat session...")
    print("Type 'quit' to exit, 'load <path>' to load a model, 'status' to check model status")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'status':
                response = requests.get(f"{API_BASE_URL}/models/status")
                if response.status_code == 200:
                    status = response.json()
                    if status['loaded']:
                        print(f"Model loaded: {status['model_path']}")
                    else:
                        print("No model currently loaded")
                else:
                    print("Error checking model status")
                continue
            elif user_input.lower().startswith('load '):
                model_path = user_input[5:].strip()
                load_request = {"model_path": model_path}
                response = requests.post(f"{API_BASE_URL}/models/load", json=load_request)
                if response.status_code == 200:
                    print(f"✅ Model loaded: {model_path}")
                else:
                    print(f"❌ Failed to load model: {response.text}")
                continue
            elif not user_input:
                continue
            
            # Send chat message
            chat_request = {
                "message": user_input,
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{API_BASE_URL}/chat/single",
                json=chat_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                chat_data = response.json()
                print(f"Assistant: {chat_data['response']}")
            else:
                error_data = response.json()
                print(f"❌ Error: {error_data['detail']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_chat()
    else:
        test_chat_api()
        
        # Ask if user wants to start interactive session
        try:
            choice = input("\nWould you like to start an interactive chat session? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                interactive_chat()
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")

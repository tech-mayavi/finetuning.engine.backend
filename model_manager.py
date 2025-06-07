import os
import json
import torch
from typing import Optional, Dict, Any, List
from datetime import datetime
from unsloth import FastLanguageModel
from transformers import TextStreamer
import gc

class ModelManager:
    """Manages loading, unloading, and inference with fine-tuned models"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_path = None
        self.model_metadata = {}
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Scan for available trained models"""
        models = []
        
        # Check default results directory
        results_dir = "./results"
        if os.path.exists(results_dir):
            for item in os.listdir(results_dir):
                model_path = os.path.join(results_dir, item)
                if os.path.isdir(model_path):
                    # Check if it's a valid model directory
                    if self._is_valid_model_dir(model_path):
                        metadata = self._get_model_metadata(model_path)
                        models.append({
                            "name": item,
                            "path": model_path,
                            "size_mb": self._get_directory_size(model_path),
                            "created_at": metadata.get("created_at"),
                            "training_config": metadata.get("config", {})
                        })
        
        # Check for lora_model directory (common output)
        lora_model_path = "./lora_model"
        if os.path.exists(lora_model_path) and self._is_valid_model_dir(lora_model_path):
            metadata = self._get_model_metadata(lora_model_path)
            models.append({
                "name": "lora_model",
                "path": lora_model_path,
                "size_mb": self._get_directory_size(lora_model_path),
                "created_at": metadata.get("created_at"),
                "training_config": metadata.get("config", {})
            })
            
        return models
    
    def _is_valid_model_dir(self, path: str) -> bool:
        """Check if directory contains a valid model"""
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        return all(os.path.exists(os.path.join(path, f)) for f in required_files)
    
    def _get_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Extract metadata from model directory"""
        metadata = {}
        
        # Try to read adapter config
        config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                    metadata["adapter_config"] = adapter_config
            except:
                pass
        
        # Get creation time
        try:
            stat = os.stat(model_path)
            metadata["created_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except:
            pass
            
        return metadata
    
    def _get_directory_size(self, path: str) -> float:
        """Get directory size in MB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 2)  # Convert to MB
        except:
            return 0.0
    
    def load_model(self, model_path: str, max_seq_length: int = 2048) -> Dict[str, Any]:
        """Load a fine-tuned model"""
        try:
            # Unload current model first
            if self.current_model is not None:
                self.unload_model()
            
            # Validate model path
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
            
            if not self._is_valid_model_dir(model_path):
                raise ValueError(f"Invalid model directory: {model_path}")
            
            # Load the model and tokenizer
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(model)
            
            # Ensure model is on GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_path = model_path
            self.model_metadata = self._get_model_metadata(model_path)
            
            return {
                "status": "success",
                "message": f"Model loaded successfully from {model_path}",
                "model_path": model_path,
                "metadata": self.model_metadata
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}",
                "model_path": model_path
            }
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model to free memory"""
        try:
            if self.current_model is not None:
                del self.current_model
                del self.current_tokenizer
                self.current_model = None
                self.current_tokenizer = None
                self.current_model_path = None
                self.model_metadata = {}
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    "status": "success",
                    "message": "Model unloaded successfully"
                }
            else:
                return {
                    "status": "info",
                    "message": "No model currently loaded"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error unloading model: {str(e)}"
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        if self.current_model is None:
            return {
                "loaded": False,
                "model_path": None,
                "metadata": {}
            }
        else:
            return {
                "loaded": True,
                "model_path": self.current_model_path,
                "metadata": self.model_metadata
            }
    
    def _get_model_device(self) -> torch.device:
        """Get the device where the model is located"""
        if self.current_model is None:
            return torch.device("cpu")
        
        # Get the device of the first parameter
        try:
            return next(self.current_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    
    def generate_response(self, 
                         message: str, 
                         max_tokens: int = 150, 
                         temperature: float = 0.7,
                         do_sample: bool = True) -> Dict[str, Any]:
        """Generate a response using the loaded model"""
        
        if self.current_model is None or self.current_tokenizer is None:
            return {
                "status": "error",
                "message": "No model currently loaded. Please load a model first.",
                "response": ""
            }
        
        try:
            # Format the prompt
            prompt = f"### Instruction:\n{message}\n\n### Response:\n"
            
            # Tokenize
            inputs = self.current_tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Get model device and move inputs to the same device
            model_device = self._get_model_device()
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.current_tokenizer.eos_token_id,
                    eos_token_id=self.current_tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after "### Response:")
            response_start = full_response.find("### Response:")
            if response_start != -1:
                response = full_response[response_start + len("### Response:"):].strip()
            else:
                response = full_response.strip()
            
            return {
                "status": "success",
                "message": "Response generated successfully",
                "response": response,
                "prompt": prompt,
                "full_output": full_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating response: {str(e)}",
                "response": ""
            }
    
    def generate_conversation_response(self, 
                                     messages: List[Dict[str, str]], 
                                     max_tokens: int = 150, 
                                     temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response for a conversation with multiple turns"""
        
        if self.current_model is None or self.current_tokenizer is None:
            return {
                "status": "error",
                "message": "No model currently loaded. Please load a model first.",
                "response": ""
            }
        
        try:
            # Build conversation prompt
            conversation_prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    conversation_prompt += f"### Instruction:\n{content}\n\n"
                elif role == "assistant":
                    conversation_prompt += f"### Response:\n{content}\n\n"
            
            # Add final response prompt
            conversation_prompt += "### Response:\n"
            
            # Generate response using the conversation context
            result = self.generate_response(
                conversation_prompt, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            if result["status"] == "success":
                result["conversation_prompt"] = conversation_prompt
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating conversation response: {str(e)}",
                "response": ""
            }

# Global model manager instance
model_manager = ModelManager()

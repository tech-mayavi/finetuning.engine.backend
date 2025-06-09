import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Import the model manager for loading models
from model_manager import model_manager

# Store evaluation jobs status
evaluation_jobs: Dict[str, Dict[str, Any]] = {}

class EvaluationService:
    def __init__(self):
        self.jobs = evaluation_jobs
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent evaluations
    
    def create_prediction_job(self, model_path: str, test_data: List[Dict], batch_size: int = 50) -> str:
        """Create a new prediction job"""
        job_id = f"eval_{uuid.uuid4().hex[:8]}"
        
        # Initialize job tracking
        self.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "model_path": model_path,
            "total_rows": len(test_data),
            "completed_rows": 0,
            "batch_size": batch_size,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "results": [],
            "progress_percentage": 0
        }
        
        # Start processing in background
        self.executor.submit(self._process_prediction_job, job_id, model_path, test_data, batch_size)
        
        return job_id
    
    def _process_prediction_job(self, job_id: str, model_path: str, test_data: List[Dict], batch_size: int):
        """Process prediction job in background thread"""
        try:
            # Update job status
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["started_at"] = datetime.now().isoformat()
            
            # Load model
            print(f"Loading model for evaluation job {job_id}: {model_path}")
            load_result = model_manager.load_model(model_path)
            
            if load_result["status"] == "error":
                raise Exception(f"Failed to load model: {load_result['message']}")
            
            print(f"Model loaded successfully for job {job_id}")
            
            # Process data in batches
            results = []
            total_rows = len(test_data)
            
            for i in range(0, total_rows, batch_size):
                batch = test_data[i:i + batch_size]
                batch_results = self._process_batch(batch)
                results.extend(batch_results)
                
                # Update progress
                completed = len(results)
                progress_percentage = (completed / total_rows) * 100
                
                self.jobs[job_id]["completed_rows"] = completed
                self.jobs[job_id]["progress_percentage"] = round(progress_percentage, 2)
                
                print(f"Job {job_id}: Processed {completed}/{total_rows} rows ({progress_percentage:.1f}%)")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Save results
            self.jobs[job_id]["results"] = results
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            print(f"Job {job_id} completed successfully with {len(results)} predictions")
            
        except Exception as e:
            print(f"Job {job_id} failed: {str(e)}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of test examples"""
        results = []
        
        for example in batch:
            try:
                # Format the prompt based on instruction and input
                prompt = self._format_prompt(example.get('instruction', ''), example.get('input', ''))
                
                # Generate prediction using the loaded model
                generation_result = model_manager.generate_response(
                    message=prompt,
                    max_tokens=150,
                    temperature=0.7,
                    do_sample=True
                )
                
                if generation_result["status"] == "success":
                    prediction = generation_result["response"].strip()
                else:
                    prediction = f"[ERROR: {generation_result['message']}]"
                
                # Create result with prediction
                result = {
                    **example,  # Keep original instruction, input, output
                    "predict": prediction
                }
                results.append(result)
                
            except Exception as e:
                # Handle individual prediction errors
                result = {
                    **example,
                    "predict": f"[ERROR: {str(e)}]"
                }
                results.append(result)
        
        return results
    
    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """Format instruction and input into a proper prompt"""
        if input_text and input_text.strip():
            return f"{instruction}\n\nInput: {input_text}\n\nOutput:"
        else:
            return f"{instruction}\n\nOutput:"
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a prediction job"""
        return self.jobs.get(job_id)
    
    def get_job_results(self, job_id: str) -> Optional[List[Dict]]:
        """Get results of a completed prediction job"""
        job = self.jobs.get(job_id)
        if job and job["status"] == "completed":
            return job["results"]
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all evaluation jobs"""
        return list(self.jobs.values())
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its results"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False

# Global evaluation service instance
evaluation_service = EvaluationService()

def validate_test_data(data: List[Dict]) -> Dict[str, Any]:
    """Validate test data format"""
    if not data:
        raise ValueError("Test data cannot be empty")
    
    required_fields = ['instruction', 'output']
    optional_fields = ['input']
    
    # Check first few rows for required fields
    sample_size = min(5, len(data))
    for i, row in enumerate(data[:sample_size]):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a valid object")
        
        for field in required_fields:
            if field not in row:
                raise ValueError(f"Row {i} missing required field: {field}")
    
    # Count fields
    field_counts = {}
    for row in data:
        for field in row.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
    
    return {
        "total_rows": len(data),
        "fields": list(field_counts.keys()),
        "field_coverage": field_counts,
        "sample_data": data[:3]  # First 3 rows as sample
    }

def load_test_data_from_file(file_path: str) -> tuple[List[Dict], str]:
    """Load test data from CSV or JSON file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        return data, 'csv'
    
    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data, 'json'
        elif isinstance(data, dict):
            # Convert single object to list
            return [data], 'json'
        else:
            raise ValueError("Invalid JSON format. Expected array of objects or single object.")
    
    elif file_extension == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data, 'jsonl'
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported: .csv, .json, .jsonl")

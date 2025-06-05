from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import threading
import json
import os
from datetime import datetime
import uuid
import pandas as pd
import tempfile
import shutil

# Import the training functionality
from train_with_logging import train_with_config, start_log_monitoring

app = FastAPI(
    title="Model Finetuning API",
    description="API for finetuning language models with real-time logging",
    version="1.0.0"
)

# Store training jobs status
training_jobs: Dict[str, Dict[str, Any]] = {}

class FinetuneRequest(BaseModel):
    dataset_name: Optional[str] = "alpaca"  # Made optional with default
    model_name: Optional[str] = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: Optional[int] = 2048
    num_train_epochs: Optional[int] = 3
    per_device_train_batch_size: Optional[int] = 2
    gradient_accumulation_steps: Optional[int] = 4
    learning_rate: Optional[float] = 2e-4
    max_steps: Optional[int] = 60
    warmup_steps: Optional[int] = 5
    save_steps: Optional[int] = 25
    logging_steps: Optional[int] = 1
    output_dir: Optional[str] = "./results"
    lora_r: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.0
    
    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"

class FinetuneResponse(BaseModel):
    job_id: str
    status: str
    message: str
    dashboard_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    logs: Optional[list] = None
    error: Optional[str] = None

def validate_csv_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate CSV data format for training"""
    required_columns = ['instruction', 'output']
    optional_columns = ['input']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty data
    if len(df) == 0:
        raise ValueError("CSV file is empty")
    
    # Check for null values in required columns
    null_counts = {}
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_counts[col] = null_count
    
    return {
        "total_rows": len(df),
        "columns": list(df.columns),
        "null_counts": null_counts,
        "sample_data": df.head(3).to_dict('records')
    }

def run_training_job_with_csv(job_id: str, csv_path: str, config: Dict[str, Any]):
    """Run training with CSV data in a separate thread"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Start log monitoring server
        log_server_thread = threading.Thread(target=start_log_monitoring, daemon=True)
        log_server_thread.start()
        
        # Call training function with CSV and config
        train_with_config(csv_path, config)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["message"] = "Training completed successfully"
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["failed_at"] = datetime.now().isoformat()
    finally:
        # Clean up temporary CSV file
        if os.path.exists(csv_path):
            os.remove(csv_path)

def run_training_job(job_id: str, config: FinetuneRequest):
    """Run training in a separate thread (legacy method)"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Start log monitoring server
        log_server_thread = threading.Thread(target=start_log_monitoring, daemon=True)
        log_server_thread.start()
        
        # Convert to dict and call new training function
        config_dict = config.dict()
        train_with_config(None, config_dict)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["message"] = "Training completed successfully"
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["failed_at"] = datetime.now().isoformat()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Model Finetuning API",
        "version": "1.0.0",
        "endpoints": {
            "/finetune": "POST - Start a new finetuning job",
            "/jobs/{job_id}": "GET - Get job status",
            "/jobs": "GET - List all jobs",
            "/logs/{job_id}": "GET - Get job logs",
            "/dashboard": "GET - Get dashboard URL"
        }
    }

@app.post("/finetune-simple")
async def start_simple_finetuning(background_tasks: BackgroundTasks):
    """Start a simple finetuning job with default parameters"""
    
    # Create default request
    request = FinetuneRequest()
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    training_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "config": request.dict(),
        "created_at": datetime.now().isoformat(),
        "logs": []
    }
    
    # Start training in background
    background_tasks.add_task(run_training_job, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Simple finetuning job started with default parameters",
        "dashboard_url": "http://localhost:5000"
    }

@app.post("/finetune")
async def start_finetuning_with_csv(
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...),
    model_name: str = Form("unsloth/llama-3-8b-bnb-4bit"),
    max_seq_length: int = Form(2048),
    num_train_epochs: int = Form(3),
    per_device_train_batch_size: int = Form(2),
    gradient_accumulation_steps: int = Form(4),
    learning_rate: float = Form(2e-4),
    max_steps: int = Form(60),
    warmup_steps: int = Form(5),
    save_steps: int = Form(25),
    logging_steps: int = Form(1),
    output_dir: str = Form("./results"),
    lora_r: int = Form(16),
    lora_alpha: int = Form(16),
    lora_dropout: float = Form(0.0)
):
    """Start a new finetuning job with CSV data and configurable parameters"""
    
    try:
        # Validate file type
        if not data_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Validate parameters
        if max_steps <= 0:
            raise HTTPException(status_code=400, detail="max_steps must be greater than 0")
        
        if num_train_epochs <= 0:
            raise HTTPException(status_code=400, detail="num_train_epochs must be greater than 0")
        
        if learning_rate <= 0 or learning_rate > 1:
            raise HTTPException(status_code=400, detail="learning_rate must be between 0 and 1")
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, f"training_data_{uuid.uuid4().hex}.csv")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(data_file.file, buffer)
        
        # Validate CSV data
        try:
            df = pd.read_csv(temp_file_path)
            validation_result = validate_csv_data(df)
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Prepare configuration
        config = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "output_dir": output_dir,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout
        }
        
        # Initialize job tracking
        training_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "config": config,
            "dataset_info": validation_result,
            "created_at": datetime.now().isoformat(),
            "logs": []
        }
        
        # Start training in background
        background_tasks.add_task(run_training_job_with_csv, job_id, temp_file_path, config)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Finetuning job queued with {validation_result['total_rows']} training samples",
            "dashboard_url": "http://localhost:5000",
            "dataset_info": validation_result,
            "config": config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training job: {str(e)}")

@app.post("/finetune-legacy", response_model=FinetuneResponse)
async def start_finetuning_legacy(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """Start a new finetuning job (legacy endpoint without CSV upload)"""
    
    try:
        # Validate request parameters
        if request.max_steps and request.max_steps <= 0:
            raise HTTPException(status_code=400, detail="max_steps must be greater than 0")
        
        if request.num_train_epochs and request.num_train_epochs <= 0:
            raise HTTPException(status_code=400, detail="num_train_epochs must be greater than 0")
        
        if request.learning_rate and (request.learning_rate <= 0 or request.learning_rate > 1):
            raise HTTPException(status_code=400, detail="learning_rate must be between 0 and 1")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        training_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "config": request.dict(),
            "created_at": datetime.now().isoformat(),
            "logs": []
        }
        
        # Start training in background
        background_tasks.add_task(run_training_job, job_id, request)
        
        return FinetuneResponse(
            job_id=job_id,
            status="queued",
            message="Finetuning job has been queued and will start shortly",
            dashboard_url="http://localhost:5000"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training job: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific training job"""
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    # Try to read logs if they exist
    logs = []
    if os.path.exists('training_logs.jsonl'):
        try:
            with open('training_logs.jsonl', 'r') as f:
                logs = [json.loads(line) for line in f.readlines()]
        except Exception:
            pass
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        logs=logs[-10:],  # Return last 10 log entries
        error=job.get("error")
    )

@app.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    return {
        "jobs": list(training_jobs.values()),
        "total": len(training_jobs)
    }

@app.get("/logs/{job_id}")
async def get_job_logs(job_id: str):
    """Get detailed logs for a specific job"""
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logs = []
    if os.path.exists('training_logs.jsonl'):
        try:
            with open('training_logs.jsonl', 'r') as f:
                logs = [json.loads(line) for line in f.readlines()]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")
    
    return {
        "job_id": job_id,
        "logs": logs
    }

@app.get("/dashboard")
async def get_dashboard_info():
    """Get dashboard information"""
    return {
        "dashboard_url": "http://localhost:5000",
        "status": "available",
        "description": "Real-time training monitoring dashboard"
    }

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a training job (if possible)"""
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    # In a real implementation, you'd need to actually stop the training process
    # For now, we'll just mark it as cancelled
    training_jobs[job_id]["status"] = "cancelled"
    training_jobs[job_id]["cancelled_at"] = datetime.now().isoformat()
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job has been cancelled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

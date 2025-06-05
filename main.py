from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import threading
import json
import os
from datetime import datetime
import uuid

# Import the training functionality
from train_with_logging import main as train_main, start_log_monitoring

app = FastAPI(
    title="Model Finetuning API",
    description="API for finetuning language models with real-time logging",
    version="1.0.0"
)

# Store training jobs status
training_jobs: Dict[str, Dict[str, Any]] = {}

class FinetuneRequest(BaseModel):
    dataset_name: str
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

def run_training_job(job_id: str, config: FinetuneRequest):
    """Run training in a separate thread"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Start log monitoring server
        log_server_thread = threading.Thread(target=start_log_monitoring, daemon=True)
        log_server_thread.start()
        
        # Update the training configuration (this would need to be passed to the training function)
        # For now, we'll call the existing main function
        # In a production setup, you'd want to modify train_with_logging.py to accept parameters
        
        train_main()
        
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

@app.post("/finetune", response_model=FinetuneResponse)
async def start_finetuning(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """Start a new finetuning job"""
    
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

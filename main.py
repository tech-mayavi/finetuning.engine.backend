from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
import asyncio
import threading
import json
import os
from datetime import datetime
import uuid
import pandas as pd
import tempfile
import shutil
import base64

# Import the training functionality
from train_with_logging import train_with_config, start_log_monitoring

app = FastAPI(
    title="Model Finetuning API",
    description="API for finetuning language models with real-time logging",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

class FinetuneBase64Request(BaseModel):
    file_content: str  # base64 encoded file content
    file_type: str     # "csv", "json", "jsonl"
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

def validate_data_file(data: pd.DataFrame, file_type: str) -> Dict[str, Any]:
    """Validate data format for training (supports CSV and JSON)"""
    required_columns = ['instruction', 'output']
    optional_columns = ['input']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty data
    if len(data) == 0:
        raise ValueError(f"{file_type.upper()} file is empty")
    
    # Check for null values in required columns
    null_counts = {}
    for col in required_columns:
        null_count = data[col].isnull().sum()
        if null_count > 0:
            null_counts[col] = null_count
    
    return {
        "total_rows": len(data),
        "columns": list(data.columns),
        "null_counts": null_counts,
        "sample_data": data.head(3).to_dict('records'),
        "file_type": file_type.upper()
    }

def load_data_file(file_path: str) -> tuple[pd.DataFrame, str]:
    """Load data from CSV or JSON file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
        return df, 'csv'
    
    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of objects: [{"instruction": "...", "output": "..."}, ...]
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it's object with arrays: {"instruction": [...], "output": [...]}
            if all(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                # Single object, convert to single-row DataFrame
                df = pd.DataFrame([data])
        else:
            raise ValueError("Invalid JSON format. Expected array of objects or object with arrays.")
        
        return df, 'json'
    
    elif file_extension == '.jsonl':
        # Handle JSONL (JSON Lines) format
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        df = pd.DataFrame(data)
        return df, 'jsonl'
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .json, .jsonl")

def run_training_job_with_data_file(job_id: str, data_file_path: str, config: Dict[str, Any]):
    """Run training with data file (CSV/JSON) in a separate thread"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Start log monitoring server
        log_server_thread = threading.Thread(target=start_log_monitoring, daemon=True)
        log_server_thread.start()
        
        # Call training function with data file and config
        train_with_config(data_file_path, config)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["message"] = "Training completed successfully"
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["failed_at"] = datetime.now().isoformat()
    finally:
        # Clean up temporary data file
        if os.path.exists(data_file_path):
            os.remove(data_file_path)

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
async def start_finetuning_hybrid(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Hybrid endpoint: Start finetuning with either multipart file upload or base64 JSON"""
    
    content_type = request.headers.get("content-type", "")
    
    if content_type.startswith("multipart/form-data"):
        # Handle multipart file upload
        return await handle_multipart_request(request, background_tasks)
    elif content_type.startswith("application/json"):
        # Handle base64 JSON request
        return await handle_base64_request(request, background_tasks)
    else:
        raise HTTPException(
            status_code=400,
            detail="Content-Type must be either 'multipart/form-data' or 'application/json'"
        )

async def handle_multipart_request(request: Request, background_tasks: BackgroundTasks):
    """Handle multipart file upload request"""
    try:
        # Parse multipart form data
        form = await request.form()
        
        # Extract file
        data_file = form.get("data_file")
        if not data_file:
            raise HTTPException(status_code=400, detail="data_file is required")
        
        # Check if data_file is actually an UploadFile object
        if not hasattr(data_file, 'filename') or not hasattr(data_file, 'file'):
            raise HTTPException(status_code=400, detail="data_file must be a valid file upload")
        
        # Extract parameters with defaults
        model_name = str(form.get("model_name", "unsloth/llama-3-8b-bnb-4bit"))
        max_seq_length = int(form.get("max_seq_length", 2048))
        num_train_epochs = int(form.get("num_train_epochs", 3))
        per_device_train_batch_size = int(form.get("per_device_train_batch_size", 2))
        gradient_accumulation_steps = int(form.get("gradient_accumulation_steps", 4))
        learning_rate = float(form.get("learning_rate", 2e-4))
        max_steps = int(form.get("max_steps", 60))
        warmup_steps = int(form.get("warmup_steps", 5))
        save_steps = int(form.get("save_steps", 25))
        logging_steps = int(form.get("logging_steps", 1))
        output_dir = str(form.get("output_dir", "./results"))
        lora_r = int(form.get("lora_r", 16))
        lora_alpha = int(form.get("lora_alpha", 16))
        lora_dropout = float(form.get("lora_dropout", 0.0))
        
        # Validate file type
        allowed_extensions = ['.csv', '.json', '.jsonl']
        if not data_file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")
        
        file_extension = os.path.splitext(data_file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File must be one of: {', '.join(allowed_extensions)}. Got: {file_extension}"
            )
        
        # Validate parameters
        if max_steps <= 0:
            raise HTTPException(status_code=400, detail="max_steps must be greater than 0")
        
        if num_train_epochs <= 0:
            raise HTTPException(status_code=400, detail="num_train_epochs must be greater than 0")
        
        if learning_rate <= 0 or learning_rate > 1:
            raise HTTPException(status_code=400, detail="learning_rate must be between 0 and 1")
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, f"training_data_{uuid.uuid4().hex}{file_extension}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(data_file.file, buffer)
        
        # Load and validate data file
        try:
            df, file_type = load_data_file(temp_file_path)
            validation_result = validate_data_file(df, file_type)
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=f"Invalid {file_extension} file: {str(e)}")
        
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
            "logs": [],
            "upload_method": "multipart"
        }
        
        # Start training in background
        background_tasks.add_task(run_training_job_with_data_file, job_id, temp_file_path, config)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Finetuning job queued with {validation_result['total_rows']} training samples from {validation_result['file_type']} file (multipart upload)",
            "dashboard_url": "http://localhost:5000",
            "dataset_info": validation_result,
            "config": config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multipart request: {str(e)}")

async def handle_base64_request(request: Request, background_tasks: BackgroundTasks):
    """Handle base64 JSON request"""
    try:
        # Parse JSON body
        body = await request.json()
        base64_request = FinetuneBase64Request(**body)
        
        # Validate file type
        allowed_file_types = ['csv', 'json', 'jsonl']
        if base64_request.file_type.lower() not in allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"file_type must be one of: {', '.join(allowed_file_types)}. Got: {base64_request.file_type}"
            )
        
        # Validate parameters
        if base64_request.max_steps <= 0:
            raise HTTPException(status_code=400, detail="max_steps must be greater than 0")
        
        if base64_request.num_train_epochs <= 0:
            raise HTTPException(status_code=400, detail="num_train_epochs must be greater than 0")
        
        if base64_request.learning_rate <= 0 or base64_request.learning_rate > 1:
            raise HTTPException(status_code=400, detail="learning_rate must be between 0 and 1")
        
        # Decode base64 content
        try:
            file_content = base64.b64decode(base64_request.file_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        file_extension = f".{base64_request.file_type.lower()}"
        temp_file_path = os.path.join(temp_dir, f"training_data_{uuid.uuid4().hex}{file_extension}")
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Load and validate data file
        try:
            df, file_type = load_data_file(temp_file_path)
            validation_result = validate_data_file(df, file_type)
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=f"Invalid {base64_request.file_type} content: {str(e)}")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Prepare configuration
        config = {
            "model_name": base64_request.model_name,
            "max_seq_length": base64_request.max_seq_length,
            "num_train_epochs": base64_request.num_train_epochs,
            "per_device_train_batch_size": base64_request.per_device_train_batch_size,
            "gradient_accumulation_steps": base64_request.gradient_accumulation_steps,
            "learning_rate": base64_request.learning_rate,
            "max_steps": base64_request.max_steps,
            "warmup_steps": base64_request.warmup_steps,
            "save_steps": base64_request.save_steps,
            "logging_steps": base64_request.logging_steps,
            "output_dir": base64_request.output_dir,
            "lora_r": base64_request.lora_r,
            "lora_alpha": base64_request.lora_alpha,
            "lora_dropout": base64_request.lora_dropout
        }
        
        # Initialize job tracking
        training_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "config": config,
            "dataset_info": validation_result,
            "created_at": datetime.now().isoformat(),
            "logs": [],
            "upload_method": "base64"
        }
        
        # Start training in background
        background_tasks.add_task(run_training_job_with_data_file, job_id, temp_file_path, config)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Finetuning job queued with {validation_result['total_rows']} training samples from {validation_result['file_type']} content (base64 upload)",
            "dashboard_url": "http://localhost:5000",
            "dataset_info": validation_result,
            "config": config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing base64 request: {str(e)}")

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

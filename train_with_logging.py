import os
import json
import sys
import threading
from datetime import datetime
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import pandas as pd

# Import our custom logging callback
from log_monitor import DetailedLoggingCallback

def setup_model_and_tokenizer():
    """Setup Unsloth model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def prepare_dataset_from_file(file_path: str, tokenizer):
    """Prepare dataset from CSV, JSON, or JSONL file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        # Load CSV data
        df = pd.read_csv(file_path)
    
    elif file_extension == '.json':
        # Load JSON data
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
    
    elif file_extension == '.jsonl':
        # Load JSONL data
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        df = pd.DataFrame(data)
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))  # Handle missing input column
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n"
            if input_text and str(input_text).strip() and str(input_text) != 'nan':  # Only add input if it exists and is not empty
                text += f"### Input:\n{input_text}\n\n"
            text += f"### Response:\n{output}"
            texts.append(text)
        
        return {"text": texts}
    
    # Convert DataFrame to HuggingFace Dataset
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def prepare_dataset_from_csv(csv_path: str, tokenizer):
    """Prepare dataset from CSV file (legacy function)"""
    return prepare_dataset_from_file(csv_path, tokenizer)

def prepare_dataset(tokenizer):
    """Prepare your dataset (legacy method)"""
    # Replace with your actual dataset loading
    dataset = load_dataset("your_dataset", split="train", token="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")
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                text += f"### Input:\n{input_text}\n\n"
            text += f"### Response:\n{output}"
            texts.append(text)
        
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def train_with_config(csv_path: str = None, config: dict = None):
    """Train model with configurable parameters and optional CSV data"""
    
    # Set default config if not provided
    if config is None:
        config = {
            "model_name": "unsloth/llama-3-8b-bnb-4bit",
            "max_seq_length": 2048,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_steps": 60,
            "warmup_steps": 5,
            "save_steps": 25,
            "logging_steps": 1,
            "output_dir": "./results",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0
        }
    
    print("Starting training with real-time logging...")
    print("Dashboard available at: http://localhost:8000/dashboard")
    
    # Log configuration
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "config_loaded",
        "level": "INFO",
        "message": "🔧 Training configuration loaded",
        "step": 0,
        "epoch": 0,
        "config": config
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Log initialization steps
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "setup_start",
        "level": "INFO",
        "message": "🚀 Setting up model and tokenizer...",
        "step": 0,
        "epoch": 0
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Setup model with configurable parameters
    # Get quantization setting from config
    quantization = config.get("quantization", "4bit")
    
    # Set quantization parameters based on user choice
    if quantization == "4bit":
        load_in_4bit = True
        load_in_8bit = False
    elif quantization == "8bit":
        load_in_4bit = False
        load_in_8bit = True
    else:  # "none"
        load_in_4bit = False
        load_in_8bit = False
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.get("model_name", "unsloth/llama-3-8b-bnb-4bit"),
        max_seq_length=config.get("max_seq_length", 2048),
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Log model setup completion
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "model_loaded",
        "level": "INFO",
        "message": "✅ Model and tokenizer loaded successfully",
        "step": 0,
        "epoch": 0
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Log dataset preparation
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "dataset_prep",
        "level": "INFO",
        "message": "📊 Preparing dataset...",
        "step": 0,
        "epoch": 0
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Prepare dataset (file or default)
    if csv_path and os.path.exists(csv_path):
        dataset = prepare_dataset_from_file(csv_path, tokenizer)
        file_extension = os.path.splitext(csv_path)[1].upper()
        dataset_source = f"{file_extension[1:]} file: {os.path.basename(csv_path)}"
    else:
        dataset = prepare_dataset(tokenizer)
        dataset_source = "Default HuggingFace dataset"
    
    # Log dataset preparation completion
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "dataset_ready",
        "level": "INFO",
        "message": f"✅ Dataset prepared and ready for training from {dataset_source}",
        "step": 0,
        "epoch": 0,
        "dataset_size": len(dataset) if dataset else 0
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Training arguments with configurable parameters
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./results"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        optim="adamw_8bit",
        warmup_steps=config.get("warmup_steps", 5),
        max_steps=config.get("max_steps", 60),
        learning_rate=config.get("learning_rate", 2e-4),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config.get("logging_steps", 1),
        logging_dir="./logs",
        save_steps=config.get("save_steps", 25),
        save_total_limit=3,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Log training arguments
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "training_args",
        "level": "INFO",
        "message": "⚙️ Training arguments configured",
        "step": 0,
        "epoch": 0,
        "training_args": {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "max_steps": training_args.max_steps,
            "warmup_steps": training_args.warmup_steps
        }
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Create trainer with custom callback
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[DetailedLoggingCallback(logging_steps=config.get("logging_steps", 1))]
    )
    
    # Show model info
    trainer.model.print_trainable_parameters()
    
    # Start training
    print("Training started! Check the dashboard for real-time updates.")
    trainer_stats = trainer.train()
    
    # Save model
    model_output_dir = config.get("output_dir", "./results")
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    # Log completion
    completion_log = {
        "timestamp": datetime.now().isoformat(),
        "type": "training_complete",
        "level": "INFO",
        "message": f"🎉 Training completed successfully! Model saved to {model_output_dir}",
        "step": trainer_stats.global_step if hasattr(trainer_stats, 'global_step') else 0,
        "epoch": 0,
        "stats": {
            "train_runtime": trainer_stats.metrics.get("train_runtime"),
            "train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second"),
            "train_steps_per_second": trainer_stats.metrics.get("train_steps_per_second"),
            "total_flos": trainer_stats.metrics.get("total_flos"),
            "train_loss": trainer_stats.metrics.get("train_loss")
        }
    }
    
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(completion_log) + '\n')
    
    print(f"Training completed! Final model saved to '{model_output_dir}'")

def main():
    """Legacy main function for backward compatibility"""
    train_with_config()

if __name__ == "__main__":
    main()

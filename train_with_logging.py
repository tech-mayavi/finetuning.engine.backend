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
from log_monitor import DetailedLoggingCallback, start_log_monitoring

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

def prepare_dataset(tokenizer):
    """Prepare your dataset"""
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

def main():
    # Start log monitoring server in a separate thread
    log_server_thread = threading.Thread(target=start_log_monitoring, daemon=True)
    log_server_thread.start()
    
    print("Starting training with real-time logging...")
    print("Dashboard available at: http://localhost:5000")
    
    # Log initialization steps
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "setup_start",
        "level": "INFO",
        "message": "🔧 Setting up model and tokenizer...",
        "step": 0,
        "epoch": 0
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
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
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Log dataset preparation completion
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "dataset_ready",
        "level": "INFO",
        "message": "✅ Dataset prepared and ready for training",
        "step": 0,
        "epoch": 0
    }
    with open('training_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Training arguments with detailed logging
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        warmup_steps=5,
        max_steps=60,  # Reduce for testing
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,  # Log every step
        logging_dir="./logs",
        save_steps=25,
        save_total_limit=3,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Create trainer with custom callback
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[DetailedLoggingCallback()]
    )
    
    # Show model info
    trainer.model.print_trainable_parameters()
    
    # Start training
    print("Training started! Check the dashboard for real-time updates.")
    trainer_stats = trainer.train()
    
    # Save model
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    
    # Log completion
    completion_log = {
        "timestamp": datetime.now().isoformat(),
        "type": "training_complete",
        "level": "INFO",
        "message": "Training completed successfully",
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
    
    print("Training completed! Final model saved to 'lora_model'")

if __name__ == "__main__":
    main()

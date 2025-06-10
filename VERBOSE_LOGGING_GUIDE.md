# Verbose Logging System Guide

## Overview

The verbose logging system provides comprehensive, configurable logging for model training with detailed system monitoring, resource usage tracking, and performance analysis. This enhanced logging system captures everything from basic training progress to ultra-detailed system metrics.

## Features

### 🔍 **Multi-Level Verbosity**
- **Low**: Basic training progress with minimal overhead
- **Medium**: Standard progress with moderate resource monitoring
- **High**: Detailed progress with comprehensive monitoring (default)
- **Ultra**: Maximum verbosity with complete system analysis

### 📊 **Comprehensive Monitoring**
- **System Information**: CPU, memory, GPU, disk, network
- **Resource Usage**: Real-time tracking with trend analysis
- **Performance Metrics**: Timing, throughput, efficiency analysis
- **Model Information**: Parameter counts, size estimation, configuration
- **Dataset Analysis**: Structure, size, sample information
- **Training Progress**: Step-by-step progress with ETA calculations

### ⚠️ **Smart Alerting**
- Memory usage warnings (>90%)
- GPU memory warnings (>95%)
- Performance anomaly detection
- Resource trend analysis

## Quick Start

### Basic Usage

```python
from verbose_logging import VerboseLoggingCallback

# Create callback with high verbosity (default)
callback = VerboseLoggingCallback(verbosity_level="high")

# Use in training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    callbacks=[callback]
)
```

### Configuration-Based Usage

```python
# Configure training with verbose logging
config = {
    "use_verbose_logging": True,
    "verbosity_level": "high",  # or "low", "medium", "ultra"
    "model_name": "unsloth/llama-3-8b-bnb-4bit",
    "max_steps": 100,
    # ... other training parameters
}

# Start training with verbose logging
train_with_config(config=config)
```

## Verbosity Levels

### 📊 **Low Verbosity**
- **Purpose**: Minimal logging for production environments
- **Frequency**: Major milestones only
- **Includes**:
  - Training start/end
  - Epoch completion
  - Model saves
  - Final statistics

### 📈 **Medium Verbosity**
- **Purpose**: Balanced logging for development
- **Frequency**: Moderate logging frequency
- **Includes**:
  - All low-level logs
  - Resource usage every 5 steps
  - Performance metrics
  - Basic trend analysis

### 🔍 **High Verbosity** (Default)
- **Purpose**: Detailed monitoring for debugging
- **Frequency**: High logging frequency
- **Includes**:
  - All medium-level logs
  - Step-by-step resource usage
  - Comprehensive trend analysis
  - Resource warnings
  - Detailed timing statistics

### 🚀 **Ultra Verbosity**
- **Purpose**: Maximum detail for research/analysis
- **Frequency**: Every event logged
- **Includes**:
  - All high-level logs
  - Batch-level information
  - All trainer logs
  - System temperatures (if available)
  - Complete system state

## Log File Structure

### Default Log Files
- **Standard Logging**: `training_logs.jsonl`
- **Verbose Logging**: `verbose_training_logs.jsonl`
- **Error Logs**: `verbose_logging_errors.log`

### Log Entry Format

```json
{
  "timestamp": "2025-01-10T11:30:00.123456",
  "type": "training_step_complete",
  "level": "INFO",
  "message": "✅ Step 42 completed - Loss: 0.234567",
  "step": 42,
  "epoch": 1,
  "timing": {
    "step_time_seconds": 2.345,
    "avg_step_time_seconds": 2.123,
    "eta_minutes": 45.67,
    "estimated_completion": "2025-01-10T12:15:00.000000"
  },
  "metrics": {
    "loss": 0.234567,
    "learning_rate": 0.0001,
    "grad_norm": 1.234
  },
  "resource_usage": {
    "cpu": {"usage_percent": 75.5},
    "memory": {"usage_percent": 68.2, "used_gb": 10.9},
    "gpu": {"devices": [{"memory_usage_percent": 85.3}]}
  },
  "trends": {
    "memory_usage_trend": {
      "current": 68.2,
      "avg_last_5": 67.8,
      "trend_direction": "increasing"
    }
  }
}
```

## Configuration Options

### Training Configuration

```python
config = {
    # Verbose Logging Settings
    "use_verbose_logging": True,        # Enable/disable verbose logging
    "verbosity_level": "high",          # "low", "medium", "high", "ultra"
    
    # Standard Training Settings
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
    "output_dir": "./results"
}
```

### Custom Log File

```python
callback = VerboseLoggingCallback(
    verbosity_level="ultra",
    log_file="custom_training_logs.jsonl"
)
```

## Log Analysis

### Reading Logs

```python
import json

# Read verbose logs
logs = []
with open('verbose_training_logs.jsonl', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

# Filter by log type
training_steps = [log for log in logs if log['type'] == 'training_step_complete']
resource_warnings = [log for log in logs if log['level'] == 'WARNING']
```

### Performance Analysis

```python
# Extract timing data
step_times = [log['timing']['step_time_seconds'] 
              for log in logs 
              if log['type'] == 'training_step_complete']

# Calculate statistics
import numpy as np
avg_time = np.mean(step_times)
std_time = np.std(step_times)
print(f"Average step time: {avg_time:.3f}s ± {std_time:.3f}s")
```

### Resource Usage Trends

```python
# Extract memory usage
memory_usage = [log['resource_usage']['memory']['usage_percent']
                for log in logs 
                if 'resource_usage' in log and 'memory' in log['resource_usage']]

# Plot trends
import matplotlib.pyplot as plt
plt.plot(memory_usage)
plt.title('Memory Usage Over Time')
plt.ylabel('Memory Usage (%)')
plt.xlabel('Training Steps')
plt.show()
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from verbose_logging import VerboseLoggingCallback

app = FastAPI()

@app.post("/train")
async def start_training(config: dict):
    # Configure verbose logging
    config.update({
        "use_verbose_logging": True,
        "verbosity_level": "high"
    })
    
    # Start training with verbose logging
    train_with_config(config=config)
    
    return {"status": "Training started with verbose logging"}
```

### With Custom Monitoring

```python
import threading
import time

def monitor_logs():
    """Monitor logs in real-time"""
    while True:
        try:
            with open('verbose_training_logs.jsonl', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_log = json.loads(lines[-1])
                    if last_log['level'] == 'WARNING':
                        print(f"⚠️ Warning: {last_log['message']}")
        except:
            pass
        time.sleep(5)

# Start monitoring in background
monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
monitor_thread.start()
```

## Performance Impact

### Verbosity Level Performance

| Level  | CPU Overhead | Disk I/O | Log Size | Recommended Use |
|--------|-------------|----------|----------|-----------------|
| Low    | <1%         | Minimal  | Small    | Production      |
| Medium | 1-2%        | Low      | Medium   | Development     |
| High   | 2-3%        | Medium   | Large    | Debugging       |
| Ultra  | 3-5%        | High     | Very Large | Research      |

### Optimization Tips

1. **Use appropriate verbosity level** for your use case
2. **Monitor disk space** when using high/ultra verbosity
3. **Consider log rotation** for long training sessions
4. **Use log filtering** for analysis to reduce memory usage

## Troubleshooting

### Common Issues

1. **High disk usage**: Reduce verbosity level or implement log rotation
2. **Performance impact**: Use lower verbosity for production training
3. **Missing dependencies**: Install `psutil` and `numpy`
4. **Permission errors**: Ensure write permissions for log files

### Error Handling

The verbose logging system includes robust error handling:
- Failed log writes are captured in `verbose_logging_errors.log`
- System monitoring failures don't interrupt training
- Graceful degradation when system information is unavailable

## Testing

### Run Tests

```bash
# Test verbose logging functionality
python test_verbose_logging.py

# This will create test log files for each verbosity level
```

### Validate Logs

```python
import json

def validate_log_file(filename):
    """Validate log file format"""
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                log_entry = json.loads(line)
                required_fields = ['timestamp', 'type', 'level', 'message']
                for field in required_fields:
                    assert field in log_entry, f"Missing {field} in line {i+1}"
        print(f"✅ {filename} is valid")
    except Exception as e:
        print(f"❌ {filename} validation failed: {e}")

validate_log_file('verbose_training_logs.jsonl')
```

## Best Practices

### 1. **Choose Appropriate Verbosity**
- Use `low` for production training
- Use `medium` for development
- Use `high` for debugging issues
- Use `ultra` for research and detailed analysis

### 2. **Monitor Resources**
- Watch for memory/GPU warnings
- Monitor disk space usage
- Check log file sizes regularly

### 3. **Log Management**
- Implement log rotation for long training
- Archive old logs to save space
- Use compression for stored logs

### 4. **Analysis Workflow**
- Filter logs by type for specific analysis
- Use visualization tools for trend analysis
- Export metrics to monitoring systems

## API Reference

### VerboseLoggingCallback

```python
class VerboseLoggingCallback(TrainerCallback):
    def __init__(self, verbosity_level="high", log_file="verbose_training_logs.jsonl"):
        """
        Initialize verbose logging callback
        
        Args:
            verbosity_level: "low", "medium", "high", "ultra"
            log_file: Path to log file
        """
```

### Key Methods

- `_get_comprehensive_system_info()`: Get detailed system information
- `_get_current_resource_usage()`: Get current resource usage
- `_get_model_info(model)`: Get model information
- `_write_log(log_entry)`: Write log entry to file

## Support

For issues or questions about the verbose logging system:

1. Check the troubleshooting section
2. Review log files for error messages
3. Test with different verbosity levels
4. Validate log file format

The verbose logging system is designed to provide comprehensive insights into your model training process while maintaining flexibility and performance.

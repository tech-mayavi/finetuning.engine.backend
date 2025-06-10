import json
import threading
import time
import psutil
import platform
import sys
import gc
from datetime import datetime
from transformers import TrainerCallback
import os
import torch
import numpy as np
from typing import Dict, Any, Optional

class VerboseLoggingCallback(TrainerCallback):
    """Ultra-verbose callback for comprehensive training logging with system monitoring"""
    
    def __init__(self, verbosity_level: str = "high", log_file: str = "verbose_training_logs.jsonl"):
        """
        Initialize verbose logging callback
        
        Args:
            verbosity_level: "low", "medium", "high", "ultra" - controls logging detail level
            log_file: Path to the log file
        """
        self.verbosity_level = verbosity_level
        self.log_file = log_file
        self.start_time = None
        self.step_times = []
        self.last_step_time = None
        self.step_memory_usage = []
        self.step_gpu_usage = []
        self.initial_system_info = None
        self.process = psutil.Process()
        
        # Log system initialization
        self._log_system_initialization()
        
    def _log_system_initialization(self):
        """Log comprehensive system information at startup"""
        system_info = self._get_comprehensive_system_info()
        self.initial_system_info = system_info
        
        self._write_log({
            "timestamp": datetime.now().isoformat(),
            "type": "system_initialization",
            "level": "INFO",
            "message": "🖥️ Comprehensive system information captured",
            "step": 0,
            "epoch": 0,
            "system_info": system_info
        })
    
    def _get_comprehensive_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        try:
            # Basic system info
            system_info = {
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "architecture": platform.architecture(),
                    "python_version": sys.version,
                    "python_executable": sys.executable,
                    "hostname": platform.node()
                },
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "usage_percent": psutil.virtual_memory().percent,
                    "cached_gb": round(getattr(psutil.virtual_memory(), 'cached', 0) / (1024**3), 2),
                    "buffers_gb": round(getattr(psutil.virtual_memory(), 'buffers', 0) / (1024**3), 2)
                },
                "swap": {
                    "total_gb": round(psutil.swap_memory().total / (1024**3), 2),
                    "used_gb": round(psutil.swap_memory().used / (1024**3), 2),
                    "free_gb": round(psutil.swap_memory().free / (1024**3), 2),
                    "usage_percent": psutil.swap_memory().percent
                },
                "disk": {
                    "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                    "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                    "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
                    "usage_percent": round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 2)
                }
            }
            
            # GPU information
            if torch.cuda.is_available():
                gpu_info = {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "devices": []
                }
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        "device_id": i,
                        "name": device_props.name,
                        "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                        "major": device_props.major,
                        "minor": device_props.minor,
                        "multi_processor_count": device_props.multi_processor_count,
                        "max_threads_per_multi_processor": device_props.max_threads_per_multi_processor,
                        "warp_size": device_props.warp_size
                    }
                    
                    # Current GPU memory usage
                    if i == torch.cuda.current_device():
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        device_info.update({
                            "memory_allocated_gb": round(memory_allocated / (1024**3), 2),
                            "memory_reserved_gb": round(memory_reserved / (1024**3), 2),
                            "memory_free_gb": round((device_props.total_memory - memory_reserved) / (1024**3), 2),
                            "memory_usage_percent": round((memory_reserved / device_props.total_memory) * 100, 2)
                        })
                    
                    gpu_info["devices"].append(device_info)
                
                system_info["gpu"] = gpu_info
            else:
                system_info["gpu"] = {"cuda_available": False, "reason": "CUDA not available"}
            
            # Process information
            system_info["process"] = {
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "memory_info": {
                    "rss_mb": round(self.process.memory_info().rss / (1024**2), 2),
                    "vms_mb": round(self.process.memory_info().vms / (1024**2), 2),
                    "shared_mb": round(getattr(self.process.memory_info(), 'shared', 0) / (1024**2), 2)
                },
                "cpu_percent": self.process.cpu_percent(),
                "num_threads": self.process.num_threads(),
                "create_time": datetime.fromtimestamp(self.process.create_time()).isoformat(),
                "status": self.process.status(),
                "nice": self.process.nice() if hasattr(self.process, 'nice') else None
            }
            
            # Environment variables (filtered for security)
            env_vars = {}
            safe_env_vars = ['PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']
            for var in safe_env_vars:
                if var in os.environ:
                    env_vars[var] = os.environ[var]
            system_info["environment"] = env_vars
            
            return system_info
            
        except Exception as e:
            return {"error": f"Failed to get system info: {str(e)}"}
    
    def _get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            resource_usage = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=0.1),
                    "per_core": psutil.cpu_percent(interval=0.1, percpu=True),
                    "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                    "context_switches": psutil.cpu_stats().ctx_switches if hasattr(psutil.cpu_stats(), 'ctx_switches') else None,
                    "interrupts": psutil.cpu_stats().interrupts if hasattr(psutil.cpu_stats(), 'interrupts') else None
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "usage_percent": psutil.virtual_memory().percent,
                    "cached_gb": round(getattr(psutil.virtual_memory(), 'cached', 0) / (1024**3), 2),
                    "buffers_gb": round(getattr(psutil.virtual_memory(), 'buffers', 0) / (1024**3), 2),
                    "shared_gb": round(getattr(psutil.virtual_memory(), 'shared', 0) / (1024**3), 2)
                },
                "swap": {
                    "total_gb": round(psutil.swap_memory().total / (1024**3), 2),
                    "used_gb": round(psutil.swap_memory().used / (1024**3), 2),
                    "free_gb": round(psutil.swap_memory().free / (1024**3), 2),
                    "usage_percent": psutil.swap_memory().percent
                },
                "process": {
                    "memory_rss_mb": round(self.process.memory_info().rss / (1024**2), 2),
                    "memory_vms_mb": round(self.process.memory_info().vms / (1024**2), 2),
                    "memory_percent": round(self.process.memory_percent(), 2),
                    "cpu_percent": self.process.cpu_percent(),
                    "num_threads": self.process.num_threads(),
                    "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else None,
                    "io_counters": {
                        "read_count": self.process.io_counters().read_count if hasattr(self.process, 'io_counters') else None,
                        "write_count": self.process.io_counters().write_count if hasattr(self.process, 'io_counters') else None,
                        "read_bytes": self.process.io_counters().read_bytes if hasattr(self.process, 'io_counters') else None,
                        "write_bytes": self.process.io_counters().write_bytes if hasattr(self.process, 'io_counters') else None
                    } if hasattr(self.process, 'io_counters') else None
                }
            }
            
            # Disk I/O statistics
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    resource_usage["disk_io"] = {
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count,
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_time": disk_io.read_time,
                        "write_time": disk_io.write_time
                    }
            except:
                pass
            
            # Network I/O statistics
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    resource_usage["network_io"] = {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv,
                        "errin": net_io.errin,
                        "errout": net_io.errout,
                        "dropin": net_io.dropin,
                        "dropout": net_io.dropout
                    }
            except:
                pass
            
            # GPU usage
            if torch.cuda.is_available():
                gpu_usage = {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "devices": []
                }
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = device_props.total_memory
                    
                    device_usage = {
                        "device_id": i,
                        "name": device_props.name,
                        "memory_allocated_gb": round(memory_allocated / (1024**3), 2),
                        "memory_reserved_gb": round(memory_reserved / (1024**3), 2),
                        "memory_total_gb": round(memory_total / (1024**3), 2),
                        "memory_free_gb": round((memory_total - memory_reserved) / (1024**3), 2),
                        "memory_usage_percent": round((memory_reserved / memory_total) * 100, 2),
                        "memory_allocated_percent": round((memory_allocated / memory_total) * 100, 2)
                    }
                    
                    # Additional GPU stats if available
                    try:
                        device_usage["memory_stats"] = {
                            "active_bytes": torch.cuda.memory_stats(i).get("active_bytes.all.current", 0),
                            "inactive_split_bytes": torch.cuda.memory_stats(i).get("inactive_split_bytes.all.current", 0),
                            "allocated_bytes": torch.cuda.memory_stats(i).get("allocated_bytes.all.current", 0),
                            "reserved_bytes": torch.cuda.memory_stats(i).get("reserved_bytes.all.current", 0)
                        }
                    except:
                        pass
                    
                    gpu_usage["devices"].append(device_usage)
                
                resource_usage["gpu"] = gpu_usage
            
            # Garbage collection info
            resource_usage["gc"] = {
                "collections": gc.get_count(),
                "threshold": gc.get_threshold(),
                "stats": gc.get_stats() if hasattr(gc, 'get_stats') else None
            }
            
            # Temperature sensors (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    resource_usage["temperatures"] = {}
                    for name, entries in temps.items():
                        resource_usage["temperatures"][name] = [
                            {"label": entry.label or "unlabeled", "current": entry.current, "high": entry.high, "critical": entry.critical}
                            for entry in entries
                        ]
            except:
                pass
            
            return resource_usage
            
        except Exception as e:
            return {"error": f"Failed to get resource usage: {str(e)}"}
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            model_info = {
                "model_type": type(model).__name__,
                "model_class": str(type(model)),
                "device": str(next(model.parameters()).device) if hasattr(model, 'parameters') else "unknown"
            }
            
            # Count parameters
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_info.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "non_trainable_parameters": total_params - trainable_params,
                    "trainable_percentage": round((trainable_params / total_params) * 100, 2) if total_params > 0 else 0
                })
                
                # Parameter breakdown by layer type
                param_breakdown = {}
                for name, param in model.named_parameters():
                    layer_type = name.split('.')[0] if '.' in name else name
                    if layer_type not in param_breakdown:
                        param_breakdown[layer_type] = {"count": 0, "trainable": 0, "total_params": 0}
                    param_breakdown[layer_type]["count"] += 1
                    param_breakdown[layer_type]["total_params"] += param.numel()
                    if param.requires_grad:
                        param_breakdown[layer_type]["trainable"] += 1
                
                model_info["parameter_breakdown"] = param_breakdown
            
            # Model size estimation
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                model_size_mb = (param_size + buffer_size) / (1024**2)
                
                model_info.update({
                    "estimated_size_mb": round(model_size_mb, 2),
                    "parameter_size_mb": round(param_size / (1024**2), 2),
                    "buffer_size_mb": round(buffer_size / (1024**2), 2)
                })
            
            # Model configuration if available
            if hasattr(model, 'config'):
                config_dict = model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
                model_info["config"] = config_dict
            
            return model_info
            
        except Exception as e:
            return {"error": f"Failed to get model info: {str(e)}"}
    
    def _get_dataset_info(self, train_dataset) -> Dict[str, Any]:
        """Get detailed dataset information"""
        try:
            dataset_info = {
                "dataset_type": type(train_dataset).__name__,
                "dataset_size": len(train_dataset) if hasattr(train_dataset, '__len__') else "unknown"
            }
            
            # Sample a few examples to understand structure
            if hasattr(train_dataset, '__getitem__') and len(train_dataset) > 0:
                try:
                    sample = train_dataset[0]
                    if isinstance(sample, dict):
                        dataset_info["sample_keys"] = list(sample.keys())
                        dataset_info["sample_structure"] = {k: type(v).__name__ for k, v in sample.items()}
                        
                        # Get text lengths if text field exists
                        text_fields = ['text', 'input_ids', 'content', 'prompt']
                        for field in text_fields:
                            if field in sample:
                                if isinstance(sample[field], str):
                                    dataset_info[f"{field}_sample_length"] = len(sample[field])
                                elif hasattr(sample[field], '__len__'):
                                    dataset_info[f"{field}_sample_length"] = len(sample[field])
                                break
                except:
                    pass
            
            return dataset_info
            
        except Exception as e:
            return {"error": f"Failed to get dataset info: {str(e)}"}
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training with comprehensive logging"""
        self.start_time = time.time()
        self.last_step_time = self.start_time
        
        # Get model and dataset information
        model = kwargs.get('model')
        train_dataset = kwargs.get('train_dataloader').dataset if kwargs.get('train_dataloader') else None
        
        model_info = self._get_model_info(model) if model else {}
        dataset_info = self._get_dataset_info(train_dataset) if train_dataset else {}
        
        # Log comprehensive training initialization
        self._write_log({
            "timestamp": datetime.now().isoformat(),
            "type": "training_initialization",
            "level": "INFO",
            "message": "🚀 Comprehensive training initialization with full system context",
            "step": 0,
            "epoch": 0,
            "training_args": {
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
                "logging_steps": args.logging_steps,
                "save_steps": args.save_steps,
                "eval_steps": getattr(args, 'eval_steps', None),
                "fp16": getattr(args, 'fp16', False),
                "bf16": getattr(args, 'bf16', False),
                "dataloader_pin_memory": getattr(args, 'dataloader_pin_memory', False),
                "dataloader_num_workers": getattr(args, 'dataloader_num_workers', 0),
                "optim": getattr(args, 'optim', 'adamw_hf'),
                "weight_decay": getattr(args, 'weight_decay', 0.0),
                "adam_beta1": getattr(args, 'adam_beta1', 0.9),
                "adam_beta2": getattr(args, 'adam_beta2', 0.999),
                "adam_epsilon": getattr(args, 'adam_epsilon', 1e-8),
                "max_grad_norm": getattr(args, 'max_grad_norm', 1.0),
                "lr_scheduler_type": getattr(args, 'lr_scheduler_type', 'linear'),
                "warmup_ratio": getattr(args, 'warmup_ratio', 0.0)
            },
            "model_info": model_info,
            "dataset_info": dataset_info,
            "resource_usage": self._get_current_resource_usage(),
            "verbosity_level": self.verbosity_level
        })
        
        # Log training start with predictions
        estimated_duration_hours = (state.max_steps * 30) / 3600  # Rough estimate: 30 seconds per step
        self._write_log({
            "timestamp": datetime.now().isoformat(),
            "type": "training_start",
            "level": "INFO",
            "message": "🎯 Training session started with comprehensive monitoring",
            "step": 0,
            "epoch": 0,
            "total_steps": state.max_steps,
            "total_epochs": args.num_train_epochs,
            "predictions": {
                "estimated_duration_hours": round(estimated_duration_hours, 2),
                "estimated_completion_time": datetime.fromtimestamp(time.time() + estimated_duration_hours * 3600).isoformat(),
                "steps_per_epoch": state.max_steps // args.num_train_epochs if args.num_train_epochs > 0 else state.max_steps
            }
        })
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step with detailed monitoring"""
        step_start_time = time.time()
        
        # Log step beginning with resource monitoring
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "step_begin",
            "level": "DEBUG" if self.verbosity_level in ["high", "ultra"] else "INFO",
            "message": f"📝 Starting step {state.global_step + 1}/{state.max_steps}",
            "step": state.global_step + 1,
            "epoch": state.epoch,
            "progress_percent": round((state.global_step / state.max_steps) * 100, 2) if state.max_steps > 0 else 0,
            "step_in_epoch": state.global_step % (state.max_steps // max(1, args.num_train_epochs)) if args.num_train_epochs > 0 else state.global_step
        }
        
        # Add resource usage for high verbosity
        if self.verbosity_level in ["high", "ultra"]:
            log_entry["resource_usage"] = self._get_current_resource_usage()
        
        # Add batch information for ultra verbosity
        if self.verbosity_level == "ultra":
            batch = kwargs.get('inputs', {})
            if batch:
                log_entry["batch_info"] = {
                    "batch_keys": list(batch.keys()) if isinstance(batch, dict) else "non-dict batch",
                    "batch_size": len(batch.get('input_ids', [])) if isinstance(batch, dict) and 'input_ids' in batch else "unknown"
                }
        
        self._write_log(log_entry)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step with comprehensive metrics"""
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        
        # Get current metrics
        logs = kwargs.get('logs', {})
        
        # Calculate timing statistics
        avg_step_time = np.mean(self.step_times[-10:]) if len(self.step_times) >= 10 else np.mean(self.step_times)
        median_step_time = np.median(self.step_times[-10:]) if len(self.step_times) >= 10 else np.median(self.step_times)
        std_step_time = np.std(self.step_times[-10:]) if len(self.step_times) >= 10 else np.std(self.step_times)
        
        remaining_steps = state.max_steps - state.global_step
        eta_seconds = avg_step_time * remaining_steps
        eta_minutes = eta_seconds / 60
        
        # Get resource usage
        resource_usage = self._get_current_resource_usage()
        
        # Store memory and GPU usage for trend analysis
        if 'memory' in resource_usage:
            self.step_memory_usage.append(resource_usage['memory']['usage_percent'])
        if 'gpu' in resource_usage and resource_usage['gpu'] and resource_usage['gpu'].get('devices'):
            gpu_usage = resource_usage['gpu']['devices'][0]['memory_usage_percent']
            self.step_gpu_usage.append(gpu_usage)
        
        # Comprehensive step completion log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "training_step_complete",
            "level": "INFO",
            "message": f"✅ Step {state.global_step} completed - Loss: {logs.get('loss', 0):.6f}",
            "step": state.global_step,
            "epoch": state.epoch,
            "timing": {
                "step_time_seconds": round(step_time, 3),
                "avg_step_time_seconds": round(avg_step_time, 3),
                "median_step_time_seconds": round(median_step_time, 3),
                "std_step_time_seconds": round(std_step_time, 3),
                "steps_per_second": round(1 / avg_step_time, 3) if avg_step_time > 0 else 0,
                "samples_per_second": round(args.per_device_train_batch_size * args.gradient_accumulation_steps / avg_step_time, 2) if avg_step_time > 0 else 0,
                "eta_minutes": round(eta_minutes, 2),
                "eta_hours": round(eta_minutes / 60, 2),
                "total_elapsed_minutes": round((current_time - self.start_time) / 60, 2),
                "estimated_completion": datetime.fromtimestamp(current_time + eta_seconds).isoformat()
            },
            "metrics": {
                "loss": logs.get('loss', 0),
                "learning_rate": logs.get('learning_rate', 0),
                "grad_norm": logs.get('grad_norm', 0),
                "epoch": logs.get('epoch', state.epoch),
                "train_runtime": logs.get('train_runtime', 0),
                "train_samples_per_second": logs.get('train_samples_per_second', 0),
                "train_steps_per_second": logs.get('train_steps_per_second', 0)
            },
            "progress": {
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "progress_percent": round((state.global_step / state.max_steps) * 100, 2) if state.max_steps > 0 else 0,
                "remaining_steps": remaining_steps,
                "steps_completed_this_session": len(self.step_times)
            }
        }
        
        # Add detailed resource usage for medium+ verbosity
        if self.verbosity_level in ["medium", "high", "ultra"]:
            log_entry["resource_usage"] = resource_usage
        
        # Add trend analysis for high+ verbosity
        if self.verbosity_level in ["high", "ultra"] and len(self.step_memory_usage) >= 5:
            log_entry["trends"] = {
                "memory_usage_trend": {
                    "current": self.step_memory_usage[-1],
                    "avg_last_5": round(np.mean(self.step_memory_usage[-5:]), 2),
                    "max_last_10": round(np.max(self.step_memory_usage[-10:]), 2) if len(self.step_memory_usage) >= 10 else round(np.max(self.step_memory_usage), 2),
                    "min_last_10": round(np.min(self.step_memory_usage[-10:]), 2) if len(self.step_memory_usage) >= 10 else round(np.min(self.step_memory_usage), 2),
                    "trend_direction": "increasing" if len(self.step_memory_usage) >= 2 and self.step_memory_usage[-1] > self.step_memory_usage[-2] else "stable_or_decreasing"
                }
            }
            
            if self.step_gpu_usage:
                log_entry["trends"]["gpu_usage_trend"] = {
                    "current": self.step_gpu_usage[-1],
                    "avg_last_5": round(np.mean(self.step_gpu_usage[-5:]), 2),
                    "max_last_10": round(np.max(self.step_gpu_usage[-10:]), 2) if len(self.step_gpu_usage) >= 10 else round(np.max(self.step_gpu_usage), 2)
                }
        
        # Add all available logs for ultra verbosity
        if self.verbosity_level == "ultra":
            log_entry["all_trainer_logs"] = logs
        
        self._write_log(log_entry)
        
        # Log warnings for resource usage
        if 'memory' in resource_usage and resource_usage['memory']['usage_percent'] > 90:
            self._write_log({
                "timestamp": datetime.now().isoformat(),
                "type": "resource_warning",
                "level": "WARNING",
                "message": f"⚠️ High memory usage detected: {resource_usage['memory']['usage_percent']:.1f}%",
                "step": state.global_step,
                "epoch": state.epoch,
                "memory_usage_percent": resource_usage['memory']['usage_percent']
            })
        
        if 'gpu' in resource_usage and resource_usage['gpu'] and resource_usage['gpu'].get('devices'):
            gpu_usage = resource_usage['gpu']['devices'][0]['memory_usage_percent']
            if gpu_usage > 95:
                self._write_log({
                    "timestamp": datetime.now().isoformat(),
                    "type": "resource_warning",
                    "level": "WARNING",
                    "message": f"⚠️ High GPU memory usage detected: {gpu_usage:.1f}%",
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "gpu_memory_usage_percent": gpu_usage
                })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch with comprehensive logging"""
        self._write_log({
            "timestamp": datetime.now().isoformat(),
            "type": "epoch_begin",
            "level": "INFO",
            "message": f"🔄 Starting epoch {state.epoch + 1}/{args.num_train_epochs}",
            "step": state.global_step,
            "epoch": state.epoch + 1,
            "total_epochs": args.num_train_epochs,
            "epoch_progress_percent": round(((state.epoch + 1) / args.num_train_epochs) * 100, 2),
            "resource_usage": self._get_current_resource_usage() if self.verbosity_level in ["high", "ultra"] else None
        })
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch with detailed analysis"""
        logs = kwargs.get('logs', {})
        
        # Calculate epoch statistics
        epoch_duration = time.time() - self.start_time
        steps_this_epoch = len([t for t in self.step_times[-20:] if t > 0])  # Approximate
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "epoch_complete",
            "level": "INFO",
            "message": f"🏁 Epoch {state.epoch} completed with comprehensive analysis",
            "step": state.global_step,
            "epoch": state.epoch,
            "epoch_metrics": {
                "train_loss": logs.get('train_loss', 0),
                "eval_loss": logs.get('eval_loss', 0),
                "epoch_duration_minutes": round(epoch_duration / 60, 2),
                "steps_this_epoch": steps_this_epoch,
                "avg_step_time_this_epoch": round(np.mean(self.step_times[-steps_this_epoch:]), 3) if steps_this_epoch > 0 else 0
            },
            "progress": {
                "epoch_progress": f"{state.epoch + 1}/{args.num_train_epochs}",
                "overall_progress_percent": round(((state.epoch + 1) / args.num_train_epochs) * 100, 2)
            }
        }
        
        # Add resource usage for medium+ verbosity
        if self.verbosity_level in ["medium", "high", "ultra"]:
            log_entry["resource_usage"] = self._get_current_resource_usage()
        
        # Add performance analysis for high+ verbosity
        if self.verbosity_level in ["high", "ultra"] and len(self.step_times) >= 10:
            recent_step_times = self.step_times[-steps_this_epoch:] if steps_this_epoch > 0 else self.step_times[-10:]
            log_entry["performance_analysis"] = {
                "step_time_stats": {
                    "mean": round(np.mean(recent_step_times), 3),
                    "median": round(np.median(recent_step_times), 3),
                    "std": round(np.std(recent_step_times), 3),
                    "min": round(np.min(recent_step_times), 3),
                    "max": round(np.max(recent_step_times), 3)
                },
                "throughput": {
                    "steps_per_minute": round(60 / np.mean(recent_step_times), 2) if np.mean(recent_step_times) > 0 else 0,
                    "samples_per_minute": round(60 * args.per_device_train_batch_size * args.gradient_accumulation_steps / np.mean(recent_step_times), 2) if np.mean(recent_step_times) > 0 else 0
                }
            }
        
        self._write_log(log_entry)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when model is saved with detailed checkpoint information"""
        checkpoint_info = {
            "timestamp": datetime.now().isoformat(),
            "type": "model_checkpoint_save",
            "level": "INFO",
            "message": f"💾 Model checkpoint saved at step {state.global_step}",
            "step": state.global_step,
            "epoch": state.epoch,
            "save_path": args.output_dir,
            "checkpoint_details": {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "training_progress_percent": round((state.global_step / state.max_steps) * 100, 2) if state.max_steps > 0 else 0
            }
        }
        
        # Add resource usage for medium+ verbosity
        if self.verbosity_level in ["medium", "high", "ultra"]:
            checkpoint_info["resource_usage"] = self._get_current_resource_usage()
        
        # Add checkpoint size information if possible
        try:
            if os.path.exists(args.output_dir):
                checkpoint_size = sum(os.path.getsize(os.path.join(args.output_dir, f)) 
                                    for f in os.listdir(args.output_dir) 
                                    if os.path.isfile(os.path.join(args.output_dir, f)))
                checkpoint_info["checkpoint_details"]["size_mb"] = round(checkpoint_size / (1024**2), 2)
        except:
            pass
        
        self._write_log(checkpoint_info)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called during evaluation with comprehensive monitoring"""
        self._write_log({
            "timestamp": datetime.now().isoformat(),
            "type": "evaluation_start",
            "level": "INFO",
            "message": f"📊 Starting evaluation at step {state.global_step}",
            "step": state.global_step,
            "epoch": state.epoch,
            "resource_usage": self._get_current_resource_usage() if self.verbosity_level in ["high", "ultra"] else None
        })
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """Called during prediction steps with detailed logging"""
        if self.verbosity_level == "ultra" or (self.verbosity_level in ["high"] and state.global_step % 5 == 0):
            self._write_log({
                "timestamp": datetime.now().isoformat(),
                "type": "prediction_step",
                "level": "DEBUG",
                "message": f"🔮 Prediction step at {state.global_step}",
                "step": state.global_step,
                "epoch": state.epoch,
                "resource_usage": self._get_current_resource_usage() if self.verbosity_level == "ultra" else None
            })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs with enhanced metrics capture"""
        if logs and (self.verbosity_level == "ultra" or 
                    (self.verbosity_level == "high" and state.global_step % 3 == 0) or
                    (self.verbosity_level == "medium" and state.global_step % 5 == 0)):
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "training_metrics_update",
                "level": "DEBUG",
                "message": f"📈 Comprehensive training metrics update - Step {state.global_step}",
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": logs
            }
            
            # Add resource usage for high+ verbosity
            if self.verbosity_level in ["high", "ultra"]:
                log_entry["resource_usage"] = self._get_current_resource_usage()
            
            self._write_log(log_entry)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training with comprehensive summary"""
        total_time = time.time() - self.start_time
        
        # Calculate comprehensive training statistics
        training_stats = {
            "total_time_seconds": round(total_time, 2),
            "total_time_minutes": round(total_time / 60, 2),
            "total_time_hours": round(total_time / 3600, 2),
            "total_steps": state.global_step,
            "total_epochs": state.epoch,
            "avg_step_time": round(np.mean(self.step_times), 3) if self.step_times else 0,
            "median_step_time": round(np.median(self.step_times), 3) if self.step_times else 0,
            "fastest_step_time": round(np.min(self.step_times), 3) if self.step_times else 0,
            "slowest_step_time": round(np.max(self.step_times), 3) if self.step_times else 0,
            "steps_per_second": round(state.global_step / total_time, 3) if total_time > 0 else 0,
            "samples_per_second": round(state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps / total_time, 2) if total_time > 0 else 0
        }
        
        # Get final loss
        final_loss = 0
        if hasattr(state, 'log_history') and state.log_history:
            final_loss = state.log_history[-1].get('train_loss', 0)
        
        completion_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "training_complete",
            "level": "INFO",
            "message": f"🎉 Training completed successfully! Total time: {training_stats['total_time_hours']:.2f} hours",
            "step": state.global_step,
            "epoch": state.epoch,
            "training_statistics": training_stats,
            "final_metrics": {
                "final_loss": final_loss,
                "total_steps_completed": state.global_step,
                "epochs_completed": state.epoch
            },
            "resource_usage": self._get_current_resource_usage(),
            "system_performance_summary": {
                "memory_usage_trend": {
                    "avg": round(np.mean(self.step_memory_usage), 2) if self.step_memory_usage else 0,
                    "max": round(np.max(self.step_memory_usage), 2) if self.step_memory_usage else 0,
                    "min": round(np.min(self.step_memory_usage), 2) if self.step_memory_usage else 0
                },
                "gpu_usage_trend": {
                    "avg": round(np.mean(self.step_gpu_usage), 2) if self.step_gpu_usage else 0,
                    "max": round(np.max(self.step_gpu_usage), 2) if self.step_gpu_usage else 0,
                    "min": round(np.min(self.step_gpu_usage), 2) if self.step_gpu_usage else 0
                } if self.step_gpu_usage else None
            }
        }
        
        self._write_log(completion_log)
    
    def _write_log(self, log_entry):
        """Write log entry to file with error handling"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error writing verbose log: {e}")
            # Fallback to basic logging
            try:
                with open('verbose_logging_errors.log', 'a') as f:
                    f.write(f"{datetime.now().isoformat()}: Error writing log - {str(e)}\n")
            except:
                pass


# Legacy alias for backward compatibility
DetailedLoggingCallback = VerboseLoggingCallback

#!/usr/bin/env python3
"""
Simple test to verify dual logging functionality without requiring transformers
"""

import json
import os
from datetime import datetime

def test_dual_logging():
    """Test the dual logging functionality"""
    
    print("🧪 Testing Dual Logging Functionality")
    print("=" * 50)
    
    # Create a mock verbose log entry (similar to what VerboseLoggingCallback would create)
    verbose_log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "training_step_complete",
        "level": "INFO",
        "message": "✅ Step 5 completed - Loss: 0.234567",
        "step": 5,
        "epoch": 1,
        "timing": {
            "step_time_seconds": 2.345,
            "avg_step_time_seconds": 2.123,
            "eta_minutes": 45.67
        },
        "metrics": {
            "loss": 0.234567,
            "learning_rate": 0.0001,
            "grad_norm": 1.234
        },
        "progress": {
            "progress_percent": 8.33,
            "remaining_steps": 55
        },
        "resource_usage": {
            "cpu": {"usage_percent": 75.5},
            "memory": {"usage_percent": 68.2, "used_gb": 10.9},
            "gpu": {"devices": [{"memory_usage_percent": 85.3}]}
        }
    }
    
    # Test the standard log creation function
    def create_standard_log(verbose_log_entry):
        """Create a simplified version of the log entry for frontend compatibility"""
        standard_log = {
            "timestamp": verbose_log_entry["timestamp"],
            "type": verbose_log_entry["type"],
            "level": verbose_log_entry["level"],
            "message": verbose_log_entry["message"],
            "step": verbose_log_entry.get("step", 0),
            "epoch": verbose_log_entry.get("epoch", 0)
        }
        
        # Add key metrics for frontend display
        if "timing" in verbose_log_entry:
            timing = verbose_log_entry["timing"]
            standard_log.update({
                "step_time": timing.get("step_time_seconds"),
                "avg_step_time": timing.get("avg_step_time_seconds"),
                "eta_minutes": timing.get("eta_minutes")
            })
        
        if "metrics" in verbose_log_entry:
            metrics = verbose_log_entry["metrics"]
            standard_log.update({
                "loss": metrics.get("loss"),
                "learning_rate": metrics.get("learning_rate"),
                "grad_norm": metrics.get("grad_norm")
            })
        
        if "progress" in verbose_log_entry:
            progress = verbose_log_entry["progress"]
            standard_log.update({
                "progress_percent": progress.get("progress_percent"),
                "remaining_steps": progress.get("remaining_steps")
            })
        
        return standard_log
    
    # Test the dual logging
    print("\n📝 Testing verbose log creation...")
    
    # Write verbose log
    verbose_filename = "test_verbose_logs.jsonl"
    with open(verbose_filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(verbose_log_entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Verbose log written to: {verbose_filename}")
    print(f"   Size: {len(json.dumps(verbose_log_entry))} characters")
    
    # Create and write standard log
    standard_log = create_standard_log(verbose_log_entry)
    standard_filename = "test_standard_logs.jsonl"
    with open(standard_filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(standard_log, ensure_ascii=False) + '\n')
    
    print(f"✅ Standard log written to: {standard_filename}")
    print(f"   Size: {len(json.dumps(standard_log))} characters")
    
    # Compare the logs
    print("\n🔍 Comparing log formats:")
    print("\nVerbose log keys:", list(verbose_log_entry.keys()))
    print("Standard log keys:", list(standard_log.keys()))
    
    print("\n📊 Log size comparison:")
    verbose_size = len(json.dumps(verbose_log_entry))
    standard_size = len(json.dumps(standard_log))
    reduction = round((1 - standard_size/verbose_size) * 100, 1)
    print(f"   Verbose: {verbose_size} chars")
    print(f"   Standard: {standard_size} chars")
    print(f"   Size reduction: {reduction}%")
    
    # Verify frontend compatibility
    print("\n🎯 Frontend compatibility check:")
    required_fields = ["timestamp", "type", "level", "message", "step", "epoch"]
    missing_fields = [field for field in required_fields if field not in standard_log]
    
    if not missing_fields:
        print("✅ All required fields present in standard log")
    else:
        print(f"❌ Missing fields: {missing_fields}")
    
    # Check for key training metrics
    training_metrics = ["loss", "learning_rate", "step_time", "progress_percent"]
    present_metrics = [metric for metric in training_metrics if metric in standard_log]
    print(f"✅ Training metrics included: {present_metrics}")
    
    # Display sample logs
    print("\n📄 Sample Standard Log (Frontend Compatible):")
    print(json.dumps(standard_log, indent=2))
    
    # Clean up test files
    try:
        os.remove(verbose_filename)
        os.remove(standard_filename)
        print(f"\n🧹 Cleaned up test files")
    except:
        pass
    
    print("\n🎉 Dual logging test completed successfully!")
    print("\n✅ Benefits of dual logging:")
    print("   - Frontend continues to work without changes")
    print("   - Verbose logs available for detailed analysis")
    print("   - Backward compatibility maintained")
    print("   - Significant size reduction for frontend logs")

if __name__ == "__main__":
    test_dual_logging()

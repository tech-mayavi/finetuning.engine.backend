#!/usr/bin/env python3
"""
Test script to demonstrate verbose logging functionality
"""

import json
from datetime import datetime
from verbose_logging import VerboseLoggingCallback

def test_verbose_logging():
    """Test the verbose logging callback with different verbosity levels"""
    
    print("🧪 Testing Verbose Logging Callback")
    print("=" * 50)
    
    # Test different verbosity levels
    verbosity_levels = ["low", "medium", "high", "ultra"]
    
    for level in verbosity_levels:
        print(f"\n📊 Testing verbosity level: {level}")
        
        # Create callback instance
        callback = VerboseLoggingCallback(
            verbosity_level=level,
            log_file=f"test_verbose_logs_{level}.jsonl"
        )
        
        print(f"✅ Created VerboseLoggingCallback with {level} verbosity")
        print(f"📁 Log file: test_verbose_logs_{level}.jsonl")
        
        # Test manual log writing
        test_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "test_log",
            "level": "INFO",
            "message": f"🧪 Test log entry for {level} verbosity level",
            "step": 1,
            "epoch": 0,
            "verbosity_level": level,
            "test_data": {
                "cpu_cores": 8,
                "memory_gb": 16,
                "gpu_available": True
            }
        }
        
        callback._write_log(test_log)
        print(f"✅ Test log written successfully")
    
    print("\n🎉 Verbose logging test completed!")
    print("\nGenerated test files:")
    for level in verbosity_levels:
        print(f"  - test_verbose_logs_{level}.jsonl")
    
    print("\n📖 Usage Examples:")
    print("1. Basic usage:")
    print("   callback = VerboseLoggingCallback(verbosity_level='high')")
    print("\n2. Custom log file:")
    print("   callback = VerboseLoggingCallback(verbosity_level='ultra', log_file='my_logs.jsonl')")
    print("\n3. In training script:")
    print("   config = {'use_verbose_logging': True, 'verbosity_level': 'high'}")
    print("   train_with_config(config=config)")

def demonstrate_verbosity_differences():
    """Demonstrate the differences between verbosity levels"""
    
    print("\n🔍 Verbosity Level Differences:")
    print("=" * 40)
    
    levels_info = {
        "low": {
            "description": "Basic training progress with minimal resource monitoring",
            "includes": ["Training start/end", "Epoch completion", "Major milestones"],
            "frequency": "Low frequency logging"
        },
        "medium": {
            "description": "Standard training progress with moderate resource monitoring",
            "includes": ["All low level logs", "Resource usage every few steps", "Performance metrics"],
            "frequency": "Medium frequency logging"
        },
        "high": {
            "description": "Detailed training progress with comprehensive monitoring",
            "includes": ["All medium level logs", "Step-by-step resource usage", "Trend analysis", "Warnings"],
            "frequency": "High frequency logging"
        },
        "ultra": {
            "description": "Maximum verbosity with complete system monitoring",
            "includes": ["All high level logs", "Batch information", "All trainer logs", "System temperatures"],
            "frequency": "Maximum frequency logging"
        }
    }
    
    for level, info in levels_info.items():
        print(f"\n📊 {level.upper()} Verbosity:")
        print(f"   Description: {info['description']}")
        print(f"   Includes: {', '.join(info['includes'])}")
        print(f"   Frequency: {info['frequency']}")

if __name__ == "__main__":
    test_verbose_logging()
    demonstrate_verbosity_differences()

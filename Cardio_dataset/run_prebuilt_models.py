#!/usr/bin/env python3
"""
Script to run prebuilt models with progress indicators and completion messages.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_progress(message, epoch=None, total_epochs=None):
    """Print progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if epoch is not None and total_epochs is not None:
        print(f"[{timestamp}] {message} - Epoch {epoch}/{total_epochs}")
    else:
        print(f"[{timestamp}] {message}")

def run_with_progress(cmd, description, total_epochs=None):
    """Run a command with progress monitoring."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        epoch_count = 0
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                
                # Check for epoch completion
                if "epoch" in line.lower() or "step" in line.lower():
                    if "%" in line:
                        # Extract progress percentage
                        try:
                            percent_part = line.split("%")[0].split()[-1]
                            if "/" in percent_part:
                                current, total = percent_part.split("/")
                                current_epoch = int(current)
                                total_epochs = int(total)
                                if current_epoch > epoch_count:
                                    epoch_count = current_epoch
                                    print_progress(f"✅ Completed epoch {current_epoch}", current_epoch, total_epochs)
                        except:
                            pass
        
        return_code = process.poll()
        if return_code == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False

def main():
    print("🎯 Starting Prebuilt Models with Progress Indicators")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run library implementations
    print_progress("Starting library implementations...")
    cmd = [
        sys.executable,
        "prebuilt_models/library_implementations/train_evaluate_gans.py",
        "--ctgan-epochs", "10",
        "--pategan-epochs", "10", 
        "--dpctgan-epochs", "10"
    ]
    results["library_implementations"] = run_with_progress(
        cmd, 
        "Library Implementations (CTGAN, PATE-GAN, DP-CTGAN)",
        total_epochs=30  # 10 epochs each for 3 models
    )
    
    # Run old CTGAN implementation
    print_progress("Starting old CTGAN implementation...")
    cmd = [
        sys.executable,
        "prebuilt_models/old_implementations/ctgan_train.py",
        "--data", "prebuilt_models/datasets/input/cardio_train_dataset.csv",
        "--epochs", "10",
        "--samples", "1000"
    ]
    results["old_ctgan"] = run_with_progress(
        cmd,
        "Old CTGAN Implementation",
        total_epochs=10
    )
    
    # Run old DP-CTGAN implementation
    print_progress("Starting old DP-CTGAN implementation...")
    cmd = [
        sys.executable,
        "prebuilt_models/old_implementations/dp_ctgan_train.py",
        "--data", "prebuilt_models/datasets/input/cardio_train_dataset.csv",
        "--epochs", "10",
        "--samples", "1000"
    ]
    results["old_dp_ctgan"] = run_with_progress(
        cmd,
        "Old DP-CTGAN Implementation (Placeholder)",
        total_epochs=10
    )
    
    # Run old PATE-CTGAN implementation
    print_progress("Starting old PATE-CTGAN implementation...")
    cmd = [
        sys.executable,
        "prebuilt_models/old_implementations/pate_ctgan_train.py",
        "--data", "prebuilt_models/datasets/input/cardio_train_dataset.csv",
        "--epochs", "10",
        "--samples", "1000"
    ]
    results["old_pate_ctgan"] = run_with_progress(
        cmd,
        "Old PATE-CTGAN Implementation (Placeholder)",
        total_epochs=10
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("📋 EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for model, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {model}: {status}")
    
    print(f"\n📁 Check the following folders for generated datasets:")
    print(f"   • prebuilt_models/datasets/generated/ctgan/")
    print(f"   • prebuilt_models/datasets/generated/pategan/")
    print(f"   • prebuilt_models/datasets/generated/dp_ctgan/")
    print(f"   • prebuilt_models/old_implementations/output/")

if __name__ == "__main__":
    main()

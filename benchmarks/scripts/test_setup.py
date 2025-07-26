#!/usr/bin/env python3
"""
Test script to verify the benchmarking system setup
"""

import os
import subprocess
import sys

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        print("✅ Docker is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ Docker is not available")
        return False

def check_models():
    """Check if models are available"""
    models_dir = './models'
    models = []
    
    for ext in ['.gguf', '.bin']:
        models.extend([f for f in os.listdir(models_dir) if f.endswith(ext)])
    
    if models:
        print(f"✅ Found {len(models)} models: {', '.join(models)}")
        return True
    else:
        print("⚠️  No models found in ./models directory")
        print("   Please add .gguf or .bin model files to continue")
        return False

def main():
    print("🔍 Testing LLM Benchmarking System Setup...")
    
    checks = [
        ("Docker", check_docker),
        ("Models", check_models)
    ]
    
    passed = 0
    for name, check_func in checks:
        print(f"\n{name} check:")
        if check_func():
            passed += 1
    
    print(f"\n📊 Setup test results: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("🎉 System is ready for benchmarking!")
        print("\nNext steps:")
        print("1. Add model files to ./models/")
        print("2. Run: python scripts/benchmark.py --build-images")
    else:
        print("⚠️  Please address the issues above before running benchmarks")

if __name__ == "__main__":
    main()

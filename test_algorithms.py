#!/usr/bin/env python
"""Test script to verify algorithm availability."""

import sys
import subprocess

print("🧪 Testing Jitterbug Algorithm Availability")
print("=" * 50)

# Test imports
print("\n1. Testing imports...")
try:
    from jitterbug import JitterbugAnalyzer, JitterbugConfig
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Core import failed: {e}")
    sys.exit(1)

# Test available algorithms
print("\n2. Checking available algorithms...")
try:
    from jitterbug.detection import get_available_algorithms
    algorithms = get_available_algorithms()
    print(f"✅ Available algorithms: {', '.join(algorithms)}")
except Exception as e:
    print(f"⚠️  Could not get algorithm list: {e}")

# Test each algorithm
print("\n3. Testing algorithm imports...")

# Ruptures (should always work)
try:
    from jitterbug.detection.algorithms import RupturesDetector
    print("✅ Ruptures algorithm available")
except ImportError as e:
    print(f"❌ Ruptures import failed: {e}")

# Bayesian
try:
    import bayesian_changepoint_detection
    print("✅ bayesian_changepoint_detection package found")
    try:
        from jitterbug.detection.algorithms import BayesianChangePointDetector
        print("✅ Bayesian algorithm available")
    except ImportError as e:
        print(f"❌ Bayesian algorithm import failed: {e}")
except ImportError:
    print("⚠️  bayesian_changepoint_detection package not installed")
    print("   Install with: pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git")

# PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} found")
    try:
        from jitterbug.detection.algorithms import TorchChangePointDetector
        print("✅ PyTorch algorithm available")
    except ImportError as e:
        print(f"❌ PyTorch algorithm import failed: {e}")
except ImportError:
    print("⚠️  PyTorch not installed")
    print("   Install with: pip install torch")

# Test CLI availability
print("\n4. Testing CLI commands...")
try:
    result = subprocess.run(['jitterbug', '--help'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ CLI is available")
    else:
        print(f"❌ CLI failed: {result.stderr}")
except Exception as e:
    print(f"❌ Could not run CLI: {e}")

# Test with example data
print("\n5. Testing analysis with example data...")
try:
    config = JitterbugConfig()
    analyzer = JitterbugAnalyzer(config)
    print("✅ Default analyzer created (uses ruptures)")
    
    # Try each algorithm
    for algo in ['ruptures', 'bcp', 'torch']:
        if algo in algorithms:
            try:
                config.change_point_detection.algorithm = algo
                analyzer = JitterbugAnalyzer(config)
                print(f"✅ {algo} analyzer created successfully")
            except Exception as e:
                print(f"❌ {algo} analyzer failed: {e}")
        else:
            print(f"⚠️  {algo} not available")
            
except Exception as e:
    print(f"❌ Analysis test failed: {e}")

print("\n" + "=" * 50)
print("Test complete!")
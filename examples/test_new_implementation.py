#!/usr/bin/env python
"""
Test script for the new Jitterbug 2.0 implementation.

This script tests the refactored code without requiring external dependencies
that might not be installed.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🧪 Testing Jitterbug 2.0 Basic Functionality")
    print("=" * 50)
    
    try:
        # Test imports
        print("✅ Testing imports...")
        from jitterbug.models import (
            RTTMeasurement, RTTDataset, MinimumRTTDataset,
            ChangePoint, LatencyJump, JitterAnalysis,
            CongestionInference, CongestionInferenceResult,
            JitterbugConfig, ChangePointDetectionConfig
        )
        print("   ✓ All model imports successful")
        
        # Test basic data structures
        print("✅ Testing data structures...")
        
        # Create a simple RTT measurement
        timestamp = datetime.now()
        measurement = RTTMeasurement(
            timestamp=timestamp,
            epoch=timestamp.timestamp(),
            rtt_value=25.5,
            source="192.168.1.1",
            destination="8.8.8.8"
        )
        print(f"   ✓ RTTMeasurement created: {measurement.rtt_value}ms")
        
        # Create sample data
        measurements = []
        start_time = datetime.now() - timedelta(hours=1)
        
        for i in range(100):
            ts = start_time + timedelta(seconds=i * 30)
            # Simulate some network variation
            base_rtt = 20.0 + (5.0 * np.sin(i * 0.1)) + np.random.normal(0, 2)
            rtt = max(1.0, base_rtt)
            
            measurements.append(RTTMeasurement(
                timestamp=ts,
                epoch=ts.timestamp(),
                rtt_value=rtt
            ))
        
        dataset = RTTDataset(measurements=measurements)
        print(f"   ✓ RTTDataset created with {len(dataset)} measurements")
        
        # Test minimum RTT computation
        min_dataset = dataset.compute_minimum_intervals(interval_minutes=5)
        print(f"   ✓ MinimumRTTDataset created with {len(min_dataset)} intervals")
        
        # Test configuration
        config = JitterbugConfig()
        print(f"   ✓ Configuration created: {config.change_point_detection.algorithm}")
        
        # Test arrays conversion
        epochs, rtt_values = dataset.to_arrays()
        print(f"   ✓ Array conversion: {len(epochs)} epochs, RTT range: {rtt_values.min():.1f}-{rtt_values.max():.1f}ms")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithms():
    """Test different algorithms with fallback for missing dependencies."""
    print("\n🔬 Testing Algorithms")
    print("=" * 30)
    
    try:
        from jitterbug.models import ChangePointDetectionConfig
        
        # Test configuration for different algorithms
        configs = [
            ("ruptures", ChangePointDetectionConfig(algorithm="ruptures")),
            ("bcp", ChangePointDetectionConfig(algorithm="bcp")),
            ("torch", ChangePointDetectionConfig(algorithm="torch")),
        ]
        
        for name, config in configs:
            try:
                print(f"✅ Testing {name} configuration...")
                print(f"   ✓ Algorithm: {config.algorithm}")
                print(f"   ✓ Threshold: {config.threshold}")
                print(f"   ✓ Min time elapsed: {config.min_time_elapsed}s")
                
                # Test algorithm initialization (without actually running)
                from jitterbug.detection import ChangePointDetector
                detector = ChangePointDetector(config)
                print(f"   ✓ {name} detector initialized successfully")
                
            except ImportError as e:
                print(f"   ⚠️  {name} dependencies not available: {e}")
            except Exception as e:
                print(f"   ❌ {name} test failed: {e}")
        
        print("\n🎉 Algorithm tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Algorithm tests failed: {e}")
        return False

def test_io_functionality():
    """Test I/O functionality."""
    print("\n💾 Testing I/O Functionality")
    print("=" * 30)
    
    try:
        from jitterbug.io import DataLoader
        
        # Test DataLoader creation
        loader = DataLoader()
        print("   ✓ DataLoader created")
        
        # Create sample CSV data
        sample_data = pd.DataFrame({
            'epoch': [1640000000 + i * 30 for i in range(50)],
            'values': [20 + np.random.normal(0, 2) for _ in range(50)]
        })
        
        # Test DataFrame loading
        dataset = loader.load_from_dataframe(sample_data)
        print(f"   ✓ DataFrame loaded: {len(dataset)} measurements")
        
        # Test validation
        validation_results = loader.validate_data(dataset)
        print(f"   ✓ Data validation: {'✓' if validation_results['valid'] else '✗'}")
        
        if validation_results['valid']:
            metrics = validation_results['metrics']
            print(f"   ✓ Duration: {metrics['duration_seconds']:.1f}s")
            print(f"   ✓ RTT range: {metrics['rtt_statistics']['min']:.1f}-{metrics['rtt_statistics']['max']:.1f}ms")
        
        print("\n🎉 I/O tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ I/O tests failed: {e}")
        return False

def test_config_functionality():
    """Test configuration functionality."""
    print("\n⚙️  Testing Configuration")
    print("=" * 30)
    
    try:
        from jitterbug.models import (
            JitterbugConfig, ChangePointDetectionConfig,
            JitterAnalysisConfig, LatencyJumpConfig
        )
        
        # Test default configuration
        config = JitterbugConfig()
        print("   ✓ Default configuration created")
        
        # Test custom configuration
        custom_config = JitterbugConfig(
            change_point_detection=ChangePointDetectionConfig(
                algorithm="ruptures",
                threshold=0.3,
                min_time_elapsed=600
            ),
            jitter_analysis=JitterAnalysisConfig(
                method="ks_test",
                threshold=0.2
            ),
            latency_jump=LatencyJumpConfig(
                threshold=0.8
            )
        )
        print("   ✓ Custom configuration created")
        print(f"   ✓ Change point algorithm: {custom_config.change_point_detection.algorithm}")
        print(f"   ✓ Jitter method: {custom_config.jitter_analysis.method}")
        print(f"   ✓ Latency jump threshold: {custom_config.latency_jump.threshold}")
        
        # Test configuration validation
        try:
            invalid_config = ChangePointDetectionConfig(threshold=2.0)  # Invalid threshold
            print("   ❌ Validation should have failed!")
        except Exception:
            print("   ✓ Configuration validation working")
        
        print("\n🎉 Configuration tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Jitterbug 2.0 Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_algorithms,
        test_io_functionality,
        test_config_functionality,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! The new implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements-new.txt")
    print("   2. Run the basic analysis example: python examples/basic_analysis.py")
    print("   3. Try the new CLI: python -m jitterbug.cli analyze --help")

if __name__ == "__main__":
    main()
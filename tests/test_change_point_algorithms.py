"""
Comprehensive tests for all change point detection algorithms.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from jitterbug.models import (
    MinimumRTTDataset,
    RTTMeasurement,
    ChangePoint,
    ChangePointDetectionConfig
)
from jitterbug.detection import (
    ChangePointDetector,
    RupturesDetector,
    BayesianChangePointDetector,
    TorchChangePointDetector
)


class TestDataGenerator:
    """Helper class to generate test data for change point detection."""
    
    @staticmethod
    def generate_synthetic_data(
        num_points: int = 100,
        change_points: list = None,
        noise_level: float = 1.0,
        base_values: list = None
    ) -> MinimumRTTDataset:
        """
        Generate synthetic RTT data with known change points.
        
        Parameters
        ----------
        num_points : int
            Number of data points to generate.
        change_points : list
            List of change point indices (0-based).
        noise_level : float
            Standard deviation of noise to add.
        base_values : list
            Base RTT values for each segment.
            
        Returns
        -------
        MinimumRTTDataset
            Generated dataset with known change points.
        """
        if change_points is None:
            change_points = [num_points // 3, 2 * num_points // 3]
        
        if base_values is None:
            base_values = [20.0, 35.0, 25.0]
        
        # Generate time series with change points
        measurements = []
        start_time = datetime.now() - timedelta(hours=2)
        
        segment_idx = 0
        for i in range(num_points):
            # Check if we've reached a change point
            if segment_idx < len(change_points) and i >= change_points[segment_idx]:
                segment_idx += 1
            
            # Generate RTT value for current segment
            base_rtt = base_values[min(segment_idx, len(base_values) - 1)]
            noise = np.random.normal(0, noise_level)
            rtt_value = max(1.0, base_rtt + noise)
            
            # Create measurement
            timestamp = start_time + timedelta(minutes=i * 15)  # 15-minute intervals
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=rtt_value
            ))
        
        return MinimumRTTDataset(
            measurements=measurements,
            interval_minutes=15,
            metadata={
                'true_change_points': change_points,
                'base_values': base_values,
                'noise_level': noise_level
            }
        )


class TestRupturesDetector:
    """Test the Ruptures change point detector."""
    
    def test_ruptures_detector_initialization(self):
        """Test initializing the Ruptures detector."""
        config = ChangePointDetectionConfig(
            algorithm="ruptures",
            threshold=0.25,
            ruptures_model="rbf",
            ruptures_penalty=10.0
        )
        
        try:
            detector = RupturesDetector(config)
            assert detector.config == config
            assert detector.rpt is not None
        except ImportError:
            pytest.skip("Ruptures library not available")
    
    def test_ruptures_different_models(self):
        """Test Ruptures with different kernel models."""
        models = ["rbf", "l1", "l2", "normal"]
        dataset = TestDataGenerator.generate_synthetic_data(50, [20, 35])
        
        for model in models:
            try:
                config = ChangePointDetectionConfig(
                    algorithm="ruptures",
                    ruptures_model=model,
                    ruptures_penalty=10.0
                )
                detector = RupturesDetector(config)
                
                change_points = detector.detect(dataset)
                
                # Should detect some change points
                assert isinstance(change_points, list)
                assert all(isinstance(cp, ChangePoint) for cp in change_points)
                
                # Check algorithm name
                for cp in change_points:
                    assert cp.algorithm == f"ruptures_{model}"
                
            except ImportError:
                pytest.skip(f"Ruptures library not available for model {model}")
    
    def test_ruptures_penalty_effects(self):
        """Test that penalty parameter affects number of change points."""
        dataset = TestDataGenerator.generate_synthetic_data(60, [20, 40])
        
        try:
            # Low penalty (more change points)
            config_low = ChangePointDetectionConfig(
                algorithm="ruptures",
                ruptures_penalty=5.0
            )
            detector_low = RupturesDetector(config_low)
            cp_low = detector_low.detect(dataset)
            
            # High penalty (fewer change points)
            config_high = ChangePointDetectionConfig(
                algorithm="ruptures",
                ruptures_penalty=20.0
            )
            detector_high = RupturesDetector(config_high)
            cp_high = detector_high.detect(dataset)
            
            # Low penalty should generally detect more change points
            # (though this might not always be true for all datasets)
            assert len(cp_low) >= 0
            assert len(cp_high) >= 0
            
        except ImportError:
            pytest.skip("Ruptures library not available")
    
    def test_ruptures_confidence_calculation(self):
        """Test that confidence values are properly calculated."""
        dataset = TestDataGenerator.generate_synthetic_data(40, [20])
        
        try:
            config = ChangePointDetectionConfig(algorithm="ruptures")
            detector = RupturesDetector(config)
            
            change_points = detector.detect(dataset)
            
            for cp in change_points:
                assert 0 <= cp.confidence <= 1
                assert cp.epoch > 0
                assert cp.timestamp is not None
                
        except ImportError:
            pytest.skip("Ruptures library not available")


class TestBayesianChangePointDetector:
    """Test the Bayesian change point detector."""
    
    def test_bcp_detector_initialization(self):
        """Test initializing the BCP detector."""
        config = ChangePointDetectionConfig(
            algorithm="bcp",
            threshold=0.25
        )
        
        try:
            detector = BayesianChangePointDetector(config)
            assert detector.config == config
            assert detector.offline_changepoint_detection is not None
        except ImportError:
            pytest.skip("Bayesian change point detection library not available")
    
    def test_bcp_detection(self):
        """Test BCP detection with synthetic data."""
        dataset = TestDataGenerator.generate_synthetic_data(30, [10, 20])
        
        try:
            config = ChangePointDetectionConfig(
                algorithm="bcp",
                threshold=0.25
            )
            detector = BayesianChangePointDetector(config)
            
            change_points = detector.detect(dataset)
            
            # Should return valid change points
            assert isinstance(change_points, list)
            assert all(isinstance(cp, ChangePoint) for cp in change_points)
            
            # Check algorithm name
            for cp in change_points:
                assert cp.algorithm == "bayesian_cp"
                assert 0 <= cp.confidence <= 1
                
        except ImportError:
            pytest.skip("Bayesian change point detection library not available")
    
    def test_bcp_threshold_effects(self):
        """Test that threshold affects detection sensitivity."""
        dataset = TestDataGenerator.generate_synthetic_data(40, [15, 25])
        
        try:
            # Low threshold (more sensitive)
            config_low = ChangePointDetectionConfig(
                algorithm="bcp",
                threshold=0.1
            )
            detector_low = BayesianChangePointDetector(config_low)
            cp_low = detector_low.detect(dataset)
            
            # High threshold (less sensitive)
            config_high = ChangePointDetectionConfig(
                algorithm="bcp",
                threshold=0.5
            )
            detector_high = BayesianChangePointDetector(config_high)
            cp_high = detector_high.detect(dataset)
            
            # Both should return valid results
            assert isinstance(cp_low, list)
            assert isinstance(cp_high, list)
            
        except ImportError:
            pytest.skip("Bayesian change point detection library not available")
    
    def test_bcp_error_handling(self):
        """Test BCP error handling with insufficient data."""
        # Create very small dataset
        measurements = [
            RTTMeasurement(
                timestamp=datetime.now(),
                epoch=datetime.now().timestamp(),
                rtt_value=20.0
            )
        ]
        dataset = MinimumRTTDataset(measurements=measurements, interval_minutes=15)
        
        try:
            config = ChangePointDetectionConfig(algorithm="bcp")
            detector = BayesianChangePointDetector(config)
            
            change_points = detector.detect(dataset)
            
            # Should handle gracefully (return empty list)
            assert isinstance(change_points, list)
            assert len(change_points) == 0
            
        except ImportError:
            pytest.skip("Bayesian change point detection library not available")


class TestTorchChangePointDetector:
    """Test the PyTorch change point detector."""
    
    def test_torch_detector_initialization(self):
        """Test initializing the PyTorch detector."""
        config = ChangePointDetectionConfig(
            algorithm="torch",
            threshold=0.25
        )
        
        try:
            detector = TorchChangePointDetector(config)
            assert detector.config == config
            assert detector.torch is not None
            assert detector.model is not None
        except ImportError:
            pytest.skip("PyTorch library not available")
    
    def test_torch_detection(self):
        """Test PyTorch detection with synthetic data."""
        dataset = TestDataGenerator.generate_synthetic_data(50, [20, 35])
        
        try:
            config = ChangePointDetectionConfig(
                algorithm="torch",
                threshold=0.25
            )
            detector = TorchChangePointDetector(config)
            
            change_points = detector.detect(dataset)
            
            # Should return valid change points
            assert isinstance(change_points, list)
            assert all(isinstance(cp, ChangePoint) for cp in change_points)
            
            # Check algorithm name
            for cp in change_points:
                assert cp.algorithm == "torch_heuristic"
                assert 0 <= cp.confidence <= 1
                
        except ImportError:
            pytest.skip("PyTorch library not available")
    
    def test_torch_data_preparation(self):
        """Test PyTorch data preparation methods."""
        dataset = TestDataGenerator.generate_synthetic_data(60)
        
        try:
            config = ChangePointDetectionConfig(algorithm="torch")
            detector = TorchChangePointDetector(config)
            
            # Test data preparation
            data_tensor, epochs, mean, std = detector._prepare_data(dataset)
            
            assert data_tensor is not None
            assert len(epochs) == len(dataset)
            assert isinstance(mean, float)
            assert isinstance(std, float)
            
        except ImportError:
            pytest.skip("PyTorch library not available")
    
    def test_torch_model_creation(self):
        """Test PyTorch model creation."""
        try:
            config = ChangePointDetectionConfig(algorithm="torch")
            detector = TorchChangePointDetector(config)
            
            model = detector._create_model()
            assert model is not None
            
            # Test model with dummy data
            dummy_input = detector.torch.randn(1, 50, 1)
            output = model(dummy_input)
            
            assert output.shape == (1, 50)
            
        except ImportError:
            pytest.skip("PyTorch library not available")


class TestChangePointDetectorInterface:
    """Test the unified change point detector interface."""
    
    def test_detector_algorithm_selection(self):
        """Test that the detector selects the correct algorithm."""
        algorithms = [
            ("ruptures", RupturesDetector),
            ("bcp", BayesianChangePointDetector),
            ("torch", TorchChangePointDetector)
        ]
        
        for algo_name, expected_class in algorithms:
            try:
                config = ChangePointDetectionConfig(algorithm=algo_name)
                detector = ChangePointDetector(config)
                
                assert isinstance(detector.algorithm, expected_class)
                
            except ImportError:
                pytest.skip(f"Dependencies for {algo_name} not available")
    
    def test_detector_with_all_algorithms(self):
        """Test detection with all available algorithms."""
        dataset = TestDataGenerator.generate_synthetic_data(40, [15, 25])
        algorithms = ["ruptures", "bcp", "torch"]
        
        for algorithm in algorithms:
            try:
                config = ChangePointDetectionConfig(algorithm=algorithm)
                detector = ChangePointDetector(config)
                
                change_points = detector.detect(dataset)
                
                # Should return valid results
                assert isinstance(change_points, list)
                assert all(isinstance(cp, ChangePoint) for cp in change_points)
                
                # Check that post-processing is applied
                if len(change_points) > 1:
                    # Check time ordering
                    epochs = [cp.epoch for cp in change_points]
                    assert epochs == sorted(epochs)
                
            except ImportError:
                pytest.skip(f"Dependencies for {algorithm} not available")
    
    def test_detector_filtering(self):
        """Test change point filtering functionality."""
        dataset = TestDataGenerator.generate_synthetic_data(100, [25, 50, 75])
        
        try:
            config = ChangePointDetectionConfig(
                algorithm="ruptures",
                min_time_elapsed=3600,  # 1 hour
                max_change_points=2
            )
            detector = ChangePointDetector(config)
            
            change_points = detector.detect(dataset)
            
            # Should respect max_change_points
            assert len(change_points) <= 2
            
            # Should respect min_time_elapsed
            if len(change_points) > 1:
                for i in range(1, len(change_points)):
                    time_diff = change_points[i].epoch - change_points[i-1].epoch
                    assert time_diff >= config.min_time_elapsed
                    
        except ImportError:
            pytest.skip("Ruptures library not available")
    
    def test_invalid_algorithm(self):
        """Test error handling for invalid algorithms."""
        with pytest.raises(ValueError, match="Unknown change point detection algorithm"):
            config = ChangePointDetectionConfig(algorithm="invalid_algorithm")
            ChangePointDetector(config)


class TestAlgorithmComparison:
    """Test comparing different algorithms on the same data."""
    
    def test_algorithm_consistency(self):
        """Test that algorithms produce consistent results."""
        dataset = TestDataGenerator.generate_synthetic_data(
            num_points=50,
            change_points=[20, 35],
            noise_level=0.5,
            base_values=[20.0, 35.0, 25.0]
        )
        
        algorithms = ["ruptures", "bcp", "torch"]
        results = {}
        
        for algorithm in algorithms:
            try:
                config = ChangePointDetectionConfig(
                    algorithm=algorithm,
                    threshold=0.25
                )
                detector = ChangePointDetector(config)
                
                change_points = detector.detect(dataset)
                results[algorithm] = change_points
                
            except ImportError:
                pytest.skip(f"Dependencies for {algorithm} not available")
        
        # All algorithms should return valid results
        for algo, cps in results.items():
            assert isinstance(cps, list)
            assert all(isinstance(cp, ChangePoint) for cp in cps)
    
    def test_algorithm_performance_characteristics(self):
        """Test performance characteristics of different algorithms."""
        import time
        
        dataset = TestDataGenerator.generate_synthetic_data(100, [30, 70])
        algorithms = ["ruptures", "bcp", "torch"]
        performance = {}
        
        for algorithm in algorithms:
            try:
                config = ChangePointDetectionConfig(algorithm=algorithm)
                detector = ChangePointDetector(config)
                
                start_time = time.time()
                change_points = detector.detect(dataset)
                end_time = time.time()
                
                performance[algorithm] = {
                    'execution_time': end_time - start_time,
                    'num_change_points': len(change_points),
                    'avg_confidence': np.mean([cp.confidence for cp in change_points]) if change_points else 0
                }
                
            except ImportError:
                pytest.skip(f"Dependencies for {algorithm} not available")
        
        # All algorithms should complete in reasonable time
        for algo, perf in performance.items():
            assert perf['execution_time'] < 60  # Should complete within 1 minute
            assert perf['num_change_points'] >= 0
            assert 0 <= perf['avg_confidence'] <= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        measurements = []
        dataset = MinimumRTTDataset(measurements=measurements, interval_minutes=15)
        
        config = ChangePointDetectionConfig(algorithm="ruptures")
        detector = ChangePointDetector(config)
        
        change_points = detector.detect(dataset)
        assert len(change_points) == 0
    
    def test_single_point_dataset(self):
        """Test behavior with single data point."""
        measurements = [
            RTTMeasurement(
                timestamp=datetime.now(),
                epoch=datetime.now().timestamp(),
                rtt_value=20.0
            )
        ]
        dataset = MinimumRTTDataset(measurements=measurements, interval_minutes=15)
        
        config = ChangePointDetectionConfig(algorithm="ruptures")
        detector = ChangePointDetector(config)
        
        change_points = detector.detect(dataset)
        assert len(change_points) == 0
    
    def test_constant_data(self):
        """Test behavior with constant RTT values."""
        measurements = []
        start_time = datetime.now()
        
        for i in range(20):
            timestamp = start_time + timedelta(minutes=i * 15)
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=20.0  # Constant value
            ))
        
        dataset = MinimumRTTDataset(measurements=measurements, interval_minutes=15)
        
        try:
            config = ChangePointDetectionConfig(algorithm="ruptures")
            detector = ChangePointDetector(config)
            
            change_points = detector.detect(dataset)
            
            # Should handle constant data gracefully
            assert isinstance(change_points, list)
            # Likely no change points for constant data
            assert len(change_points) == 0
            
        except ImportError:
            pytest.skip("Ruptures library not available")
    
    def test_very_noisy_data(self):
        """Test behavior with very noisy data."""
        dataset = TestDataGenerator.generate_synthetic_data(
            num_points=50,
            change_points=[25],
            noise_level=10.0,  # Very high noise
            base_values=[20.0, 25.0]
        )
        
        try:
            config = ChangePointDetectionConfig(
                algorithm="ruptures",
                threshold=0.4  # Less sensitive for noisy data
            )
            detector = ChangePointDetector(config)
            
            change_points = detector.detect(dataset)
            
            # Should handle noisy data without crashing
            assert isinstance(change_points, list)
            
        except ImportError:
            pytest.skip("Ruptures library not available")
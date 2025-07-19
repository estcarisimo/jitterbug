"""
Tests for Jitterbug data models.
"""

import pytest
from datetime import datetime
import numpy as np
import pandas as pd

from jitterbug.models import (
    RTTMeasurement,
    RTTDataset,
    MinimumRTTDataset,
    ChangePoint,
    LatencyJump,
    JitterAnalysis,
    CongestionInference,
    JitterbugConfig,
    ChangePointDetectionConfig,
    JitterAnalysisConfig,
)


class TestRTTMeasurement:
    """Test RTT measurement model."""
    
    def test_valid_measurement(self):
        """Test creating a valid RTT measurement."""
        timestamp = datetime.now()
        measurement = RTTMeasurement(
            timestamp=timestamp,
            epoch=timestamp.timestamp(),
            rtt_value=25.5,
            source="192.168.1.1",
            destination="8.8.8.8"
        )
        assert measurement.rtt_value == 25.5
        assert measurement.source == "192.168.1.1"
        assert measurement.destination == "8.8.8.8"
    
    def test_invalid_rtt_value(self):
        """Test that invalid RTT values raise validation errors."""
        timestamp = datetime.now()
        
        with pytest.raises(ValueError, match="RTT value must be positive"):
            RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=-1.0
            )
        
        with pytest.raises(ValueError, match="RTT value seems unreasonably high"):
            RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=15000.0  # 15 seconds
            )


class TestRTTDataset:
    """Test RTT dataset model."""
    
    def create_sample_measurements(self, count=10):
        """Create sample RTT measurements."""
        measurements = []
        base_time = datetime.now()
        
        for i in range(count):
            timestamp = base_time.replace(second=i)
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=20.0 + i * 0.5
            ))
        
        return measurements
    
    def test_valid_dataset(self):
        """Test creating a valid RTT dataset."""
        measurements = self.create_sample_measurements(5)
        dataset = RTTDataset(measurements=measurements)
        
        assert len(dataset) == 5
        assert len(dataset.measurements) == 5
    
    def test_empty_dataset(self):
        """Test that empty datasets raise validation errors."""
        with pytest.raises(ValueError, match="Dataset must contain at least one measurement"):
            RTTDataset(measurements=[])
    
    def test_to_arrays(self):
        """Test converting dataset to numpy arrays."""
        measurements = self.create_sample_measurements(3)
        dataset = RTTDataset(measurements=measurements)
        
        epochs, rtt_values = dataset.to_arrays()
        
        assert len(epochs) == 3
        assert len(rtt_values) == 3
        assert isinstance(epochs, np.ndarray)
        assert isinstance(rtt_values, np.ndarray)
    
    def test_to_dataframe(self):
        """Test converting dataset to pandas DataFrame."""
        measurements = self.create_sample_measurements(3)
        dataset = RTTDataset(measurements=measurements)
        
        df = dataset.to_dataframe()
        
        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'epoch' in df.columns
        assert 'rtt_value' in df.columns
    
    def test_compute_minimum_intervals(self):
        """Test computing minimum RTT intervals."""
        measurements = self.create_sample_measurements(20)
        dataset = RTTDataset(measurements=measurements)
        
        min_dataset = dataset.compute_minimum_intervals(interval_minutes=15)
        
        assert isinstance(min_dataset, MinimumRTTDataset)
        assert min_dataset.interval_minutes == 15
        assert len(min_dataset) <= len(dataset)


class TestChangePoint:
    """Test change point model."""
    
    def test_valid_change_point(self):
        """Test creating a valid change point."""
        timestamp = datetime.now()
        cp = ChangePoint(
            timestamp=timestamp,
            epoch=timestamp.timestamp(),
            confidence=0.85,
            algorithm="ruptures"
        )
        
        assert cp.confidence == 0.85
        assert cp.algorithm == "ruptures"
    
    def test_invalid_confidence(self):
        """Test that invalid confidence values raise validation errors."""
        timestamp = datetime.now()
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            ChangePoint(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                confidence=1.5,
                algorithm="ruptures"
            )


class TestLatencyJump:
    """Test latency jump model."""
    
    def test_valid_latency_jump(self):
        """Test creating a valid latency jump."""
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + 15)
        
        jump = LatencyJump(
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_epoch=start_time.timestamp(),
            end_epoch=end_time.timestamp(),
            has_jump=True,
            magnitude=5.2,
            threshold=0.5
        )
        
        assert jump.has_jump is True
        assert jump.magnitude == 5.2
        assert jump.threshold == 0.5
    
    def test_invalid_time_order(self):
        """Test that invalid time ordering raises validation errors."""
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute - 15)  # Earlier than start
        
        with pytest.raises(ValueError, match="End epoch must be after start epoch"):
            LatencyJump(
                start_timestamp=start_time,
                end_timestamp=end_time,
                start_epoch=start_time.timestamp(),
                end_epoch=end_time.timestamp(),
                has_jump=True,
                magnitude=5.2,
                threshold=0.5
            )


class TestJitterAnalysis:
    """Test jitter analysis model."""
    
    def test_valid_jitter_analysis(self):
        """Test creating a valid jitter analysis."""
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + 15)
        
        analysis = JitterAnalysis(
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_epoch=start_time.timestamp(),
            end_epoch=end_time.timestamp(),
            has_significant_jitter=True,
            jitter_metric=0.35,
            method="jitter_dispersion",
            threshold=0.25
        )
        
        assert analysis.has_significant_jitter is True
        assert analysis.jitter_metric == 0.35
        assert analysis.method == "jitter_dispersion"
    
    def test_ks_test_with_p_value(self):
        """Test jitter analysis with KS test and p-value."""
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + 15)
        
        analysis = JitterAnalysis(
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_epoch=start_time.timestamp(),
            end_epoch=end_time.timestamp(),
            has_significant_jitter=True,
            jitter_metric=0.85,
            method="ks_test",
            threshold=0.05,
            p_value=0.003
        )
        
        assert analysis.method == "ks_test"
        assert analysis.p_value == 0.003


class TestCongestionInference:
    """Test congestion inference model."""
    
    def test_valid_congestion_inference(self):
        """Test creating a valid congestion inference."""
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + 15)
        
        inference = CongestionInference(
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_epoch=start_time.timestamp(),
            end_epoch=end_time.timestamp(),
            is_congested=True,
            confidence=0.78
        )
        
        assert inference.is_congested is True
        assert inference.confidence == 0.78
    
    def test_to_dict(self):
        """Test converting congestion inference to dictionary."""
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + 15)
        
        inference = CongestionInference(
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_epoch=start_time.timestamp(),
            end_epoch=end_time.timestamp(),
            is_congested=True,
            confidence=0.78
        )
        
        result_dict = inference.to_dict()
        
        assert 'starts' in result_dict
        assert 'ends' in result_dict
        assert 'congestion' in result_dict
        assert result_dict['congestion'] is True


class TestJitterbugConfig:
    """Test Jitterbug configuration model."""
    
    def test_default_config(self):
        """Test creating default configuration."""
        config = JitterbugConfig()
        
        assert config.change_point_detection.algorithm == "ruptures"
        assert config.jitter_analysis.method == "jitter_dispersion"
        assert config.output_format == "json"
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test creating custom configuration."""
        config = JitterbugConfig(
            change_point_detection=ChangePointDetectionConfig(
                algorithm="bcp",
                threshold=0.3
            ),
            jitter_analysis=JitterAnalysisConfig(
                method="ks_test",
                threshold=0.15
            ),
            verbose=True
        )
        
        assert config.change_point_detection.algorithm == "bcp"
        assert config.change_point_detection.threshold == 0.3
        assert config.jitter_analysis.method == "ks_test"
        assert config.jitter_analysis.threshold == 0.15
        assert config.verbose is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid threshold
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            ChangePointDetectionConfig(threshold=2.0)
        
        # Test invalid moving average order
        with pytest.raises(ValueError, match="Moving average order must be even"):
            JitterAnalysisConfig(moving_average_order=5)  # Odd number
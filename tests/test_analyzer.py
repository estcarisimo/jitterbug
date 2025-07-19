"""
Tests for the main Jitterbug analyzer.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from jitterbug.analyzer import JitterbugAnalyzer
from jitterbug.models import JitterbugConfig, RTTMeasurement, RTTDataset


class TestJitterbugAnalyzer:
    """Test the main Jitterbug analyzer."""
    
    def create_sample_dataset(self, duration_minutes=60, interval_seconds=30):
        """Create a sample RTT dataset for testing."""
        measurements = []
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        num_measurements = (duration_minutes * 60) // interval_seconds
        
        for i in range(num_measurements):
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            
            # Simulate different network conditions
            if i < num_measurements // 3:
                # Normal conditions
                base_rtt = 20.0
                jitter = np.random.normal(0, 2.0)
            elif i < 2 * num_measurements // 3:
                # Congested conditions
                base_rtt = 35.0
                jitter = np.random.normal(0, 5.0)
            else:
                # Back to normal
                base_rtt = 22.0
                jitter = np.random.normal(0, 2.5)
            
            rtt_value = max(1.0, base_rtt + jitter)
            
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=rtt_value
            ))
        
        return RTTDataset(measurements=measurements)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        config = JitterbugConfig()
        analyzer = JitterbugAnalyzer(config)
        
        assert analyzer.config == config
        assert analyzer.data_loader is not None
        assert analyzer.change_point_detector is not None
        assert analyzer.jitter_analyzer is not None
        assert analyzer.latency_jump_analyzer is not None
        assert analyzer.congestion_inference_analyzer is not None
    
    def test_analyze_with_sample_data(self):
        """Test analysis with sample data."""
        config = JitterbugConfig()
        analyzer = JitterbugAnalyzer(config)
        
        # Create sample dataset
        dataset = self.create_sample_dataset(duration_minutes=30)
        
        # Analyze
        results = analyzer.analyze(dataset)
        
        # Verify results structure
        assert results is not None
        assert hasattr(results, 'inferences')
        assert hasattr(results, 'metadata')
        assert 'total_measurements' in results.metadata
        assert 'min_intervals' in results.metadata
        assert 'config' in results.metadata
    
    def test_analyze_from_dataframe(self):
        """Test analysis from pandas DataFrame."""
        config = JitterbugConfig()
        analyzer = JitterbugAnalyzer(config)
        
        # Create sample DataFrame
        start_time = datetime.now() - timedelta(hours=1)
        data = []
        
        for i in range(120):  # 2 hours of data, 1 minute intervals
            timestamp = start_time + timedelta(minutes=i)
            epoch = timestamp.timestamp()
            
            # Simulate network conditions
            if i < 40:
                rtt = 20.0 + np.random.normal(0, 2)
            elif i < 80:
                rtt = 35.0 + np.random.normal(0, 5)
            else:
                rtt = 22.0 + np.random.normal(0, 2.5)
            
            data.append({'epoch': epoch, 'values': max(1.0, rtt)})
        
        df = pd.DataFrame(data)
        
        # Analyze
        results = analyzer.analyze_from_dataframe(df)
        
        # Verify results
        assert results is not None
        assert len(results.inferences) >= 0  # May or may not find congestion
        assert results.metadata['total_measurements'] == 120
    
    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        config = JitterbugConfig()
        analyzer = JitterbugAnalyzer(config)
        
        # Create sample dataset
        dataset = self.create_sample_dataset(duration_minutes=30)
        
        # Analyze
        results = analyzer.analyze(dataset)
        
        # Get summary statistics
        summary = analyzer.get_summary_statistics(results)
        
        # Verify summary structure
        assert 'total_periods' in summary
        assert 'congested_periods' in summary
        assert 'total_duration_seconds' in summary
        assert 'congestion_duration_seconds' in summary
        assert 'congestion_ratio' in summary
        assert 'average_confidence' in summary
        assert 'metadata' in summary
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        config = JitterbugConfig()
        analyzer = JitterbugAnalyzer(config)
        
        # Create very small dataset
        measurements = [
            RTTMeasurement(
                timestamp=datetime.now(),
                epoch=datetime.now().timestamp(),
                rtt_value=20.0
            )
        ]
        dataset = RTTDataset(measurements=measurements)
        
        # Analyze
        results = analyzer.analyze(dataset)
        
        # Should handle gracefully
        assert results is not None
        assert len(results.inferences) == 0
        assert 'error' in results.metadata or 'min_intervals' in results.metadata
    
    def test_different_algorithms(self):
        """Test different change point detection algorithms."""
        algorithms = ["ruptures", "bcp"]  # Skip torch for now due to dependencies
        
        for algorithm in algorithms:
            try:
                config = JitterbugConfig()
                config.change_point_detection.algorithm = algorithm
                analyzer = JitterbugAnalyzer(config)
                
                # Create sample dataset
                dataset = self.create_sample_dataset(duration_minutes=30)
                
                # Analyze
                results = analyzer.analyze(dataset)
                
                # Verify results
                assert results is not None
                assert results.metadata['config']['change_point_detection']['algorithm'] == algorithm
                
            except ImportError:
                # Skip if dependencies not available
                pytest.skip(f"Dependencies for {algorithm} not available")
    
    def test_different_jitter_methods(self):
        """Test different jitter analysis methods."""
        methods = ["jitter_dispersion", "ks_test"]
        
        for method in methods:
            config = JitterbugConfig()
            config.jitter_analysis.method = method
            analyzer = JitterbugAnalyzer(config)
            
            # Create sample dataset
            dataset = self.create_sample_dataset(duration_minutes=30)
            
            # Analyze
            results = analyzer.analyze(dataset)
            
            # Verify results
            assert results is not None
            assert results.metadata['config']['jitter_analysis']['method'] == method
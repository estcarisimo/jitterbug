"""
Jitterbug: Framework for Jitter-Based Congestion Inference

A modern Python framework for detecting network congestion through jitter analysis
and change point detection in Round-Trip Time (RTT) measurements.

Key Features:
- Multiple change point detection algorithms
- Jitter dispersion and KS-test analysis methods
- Flexible data input formats (CSV, JSON, InfluxDB)
- Comprehensive configuration management
- Type-safe Pydantic models
- Modern CLI interface

Example Usage:
    from jitterbug import JitterbugAnalyzer
    from jitterbug.models import JitterbugConfig
    
    # Load configuration
    config = JitterbugConfig()
    
    # Create analyzer
    analyzer = JitterbugAnalyzer(config)
    
    # Analyze RTT data
    results = analyzer.analyze_from_file('rtts.csv')
    
    # Get congestion periods
    congested_periods = results.get_congested_periods()
"""

from .analyzer import JitterbugAnalyzer
from .models import (
    RTTMeasurement,
    RTTDataset,
    MinimumRTTDataset,
    ChangePoint,
    LatencyJump,
    JitterAnalysis,
    CongestionInference,
    CongestionInferenceResult,
    JitterbugConfig,
    ChangePointDetectionConfig,
    JitterAnalysisConfig,
    LatencyJumpConfig,
    DataProcessingConfig,
)

__version__ = "2.0.0"
__author__ = "Esteban Carisimo"
__email__ = "esteban.carisimo@northwestern.edu"

__all__ = [
    # Main analyzer
    "JitterbugAnalyzer",
    
    # Data models
    "RTTMeasurement",
    "RTTDataset",
    "MinimumRTTDataset",
    
    # Analysis models
    "ChangePoint",
    "LatencyJump", 
    "JitterAnalysis",
    "CongestionInference",
    "CongestionInferenceResult",
    
    # Configuration models
    "JitterbugConfig",
    "ChangePointDetectionConfig",
    "JitterAnalysisConfig",
    "LatencyJumpConfig", 
    "DataProcessingConfig",
]
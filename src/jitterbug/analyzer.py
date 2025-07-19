"""
Main analyzer class for Jitterbug framework.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np

from .models import (
    RTTDataset,
    MinimumRTTDataset,
    CongestionInferenceResult,
    CongestionInference,
    JitterbugConfig,
    ChangePoint,
    LatencyJump,
    JitterAnalysis
)
from .io import DataLoader
from .detection import ChangePointDetector
from .analysis import JitterAnalyzer, LatencyJumpAnalyzer, CongestionInferenceAnalyzer


logger = logging.getLogger(__name__)


class JitterbugAnalyzer:
    """
    Main analyzer class that orchestrates the entire Jitterbug analysis pipeline.
    
    This class coordinates data loading, change point detection, jitter analysis,
    latency jump detection, and final congestion inference.
    
    Parameters
    ----------
    config : JitterbugConfig
        Configuration object containing all analysis parameters.
    
    Attributes
    ----------
    config : JitterbugConfig
        Configuration object.
    data_loader : DataLoader
        Data loading component.
    change_point_detector : ChangePointDetector
        Change point detection component.
    jitter_analyzer : JitterAnalyzer
        Jitter analysis component.
    latency_jump_analyzer : LatencyJumpAnalyzer
        Latency jump analysis component.
    congestion_inference_analyzer : CongestionInferenceAnalyzer
        Congestion inference component.
    """
    
    def __init__(self, config: JitterbugConfig):
        """
        Initialize the Jitterbug analyzer.
        
        Parameters
        ----------
        config : JitterbugConfig
            Configuration object containing all analysis parameters.
        """
        self.config = config
        self._setup_logging()
        self._initialize_components()
        
        # Store analysis results for access after analysis
        self.raw_data = None
        self.min_rtt_data = None
        self.change_points = None
    
    def _setup_logging(self) -> None:
        """Set up logging based on configuration."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self) -> None:
        """Initialize all analysis components."""
        self.data_loader = DataLoader()
        self.change_point_detector = ChangePointDetector(self.config.change_point_detection)
        self.jitter_analyzer = JitterAnalyzer(self.config.jitter_analysis)
        self.latency_jump_analyzer = LatencyJumpAnalyzer(self.config.latency_jump)
        self.congestion_inference_analyzer = CongestionInferenceAnalyzer()
    
    def analyze_from_file(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None
    ) -> CongestionInferenceResult:
        """
        Analyze RTT data from a file.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the RTT data file.
        file_format : Optional[str]
            Format of the file ('csv', 'json', 'influx'). If None, will be inferred.
            
        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info(f"Loading RTT data from {file_path}")
        
        # Load RTT data
        rtt_data = self.data_loader.load_from_file(file_path, file_format)
        
        # Store raw data for visualization
        self.raw_data = rtt_data
        
        # Perform analysis
        return self.analyze(rtt_data)
    
    def analyze_from_dataframe(self, df: pd.DataFrame) -> CongestionInferenceResult:
        """
        Analyze RTT data from a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing RTT data with columns 'epoch' and 'values'.
            
        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info("Loading RTT data from DataFrame")
        
        # Convert DataFrame to RTTDataset
        rtt_data = self.data_loader.load_from_dataframe(df)
        
        # Store raw data for visualization
        self.raw_data = rtt_data
        
        # Perform analysis
        return self.analyze(rtt_data)
    
    def analyze(self, rtt_data: RTTDataset) -> CongestionInferenceResult:
        """
        Perform complete Jitterbug analysis on RTT data.
        
        Parameters
        ----------
        rtt_data : RTTDataset
            RTT measurement dataset.
            
        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info(f"Starting Jitterbug analysis on {len(rtt_data)} RTT measurements")
        
        # Step 1: Compute minimum RTT intervals
        logger.info("Computing minimum RTT intervals")
        min_rtt_data = rtt_data.compute_minimum_intervals(
            self.config.data_processing.minimum_interval_minutes
        )
        
        # Store min_rtt_data for visualization
        self.min_rtt_data = min_rtt_data
        
        if len(min_rtt_data) < 2:
            logger.warning("Insufficient data for analysis (need at least 2 intervals)")
            return CongestionInferenceResult(
                inferences=[],
                metadata={
                    'error': 'Insufficient data for analysis',
                    'min_intervals': len(min_rtt_data),
                    'config': self.config.dict()
                }
            )
        
        # Step 2: Detect change points
        logger.info("Detecting change points")
        change_points = self.change_point_detector.detect(min_rtt_data)
        
        # Store change points for visualization
        self.change_points = change_points
        
        if not change_points:
            logger.info("No change points detected")
            return CongestionInferenceResult(
                inferences=[],
                metadata={
                    'change_points': 0,
                    'config': self.config.dict()
                }
            )
        
        logger.info(f"Detected {len(change_points)} change points")
        
        # Step 3: Analyze latency jumps
        logger.info("Analyzing latency jumps")
        latency_jumps = self.latency_jump_analyzer.analyze(min_rtt_data, change_points)
        
        # Step 4: Analyze jitter
        logger.info("Analyzing jitter")
        if self.config.jitter_analysis.method == 'jitter_dispersion':
            jitter_results = self.jitter_analyzer.analyze_jitter_dispersion(
                min_rtt_data, change_points
            )
        else:  # ks_test
            jitter_results = self.jitter_analyzer.analyze_ks_test(
                rtt_data, change_points
            )
        
        # Step 5: Infer congestion
        logger.info("Inferring congestion periods")
        congestion_inferences = self.congestion_inference_analyzer.infer(
            latency_jumps, jitter_results
        )
        
        # Prepare results
        result = CongestionInferenceResult(
            inferences=congestion_inferences,
            metadata={
                'total_measurements': len(rtt_data),
                'min_intervals': len(min_rtt_data),
                'change_points': len(change_points),
                'latency_jumps': len(latency_jumps),
                'jitter_analyses': len(jitter_results),
                'congestion_periods': len([ci for ci in congestion_inferences if ci.is_congested]),
                'config': self.config.dict()
            }
        )
        
        logger.info(f"Analysis complete: {len(congestion_inferences)} periods analyzed, "
                   f"{len(result.get_congested_periods())} congestion periods found")
        
        return result
    
    def save_results(
        self,
        results: CongestionInferenceResult,
        output_path: Union[str, Path],
        format: Optional[str] = None
    ) -> None:
        """
        Save analysis results to file.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to save.
        output_path : Union[str, Path]
            Path to save results to.
        format : Optional[str]
            Output format ('json', 'csv', 'parquet'). If None, uses config default.
        """
        output_path = Path(output_path)
        format = format or self.config.output_format
        
        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results.dict(), f, indent=2, default=str)
        
        elif format == 'csv':
            df = results.to_dataframe()
            df.to_csv(output_path, index=False)
        
        elif format == 'parquet':
            df = results.to_dataframe()
            df.to_parquet(output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Results saved to {output_path}")
    
    def get_summary_statistics(self, results: CongestionInferenceResult) -> dict:
        """
        Get summary statistics from analysis results.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results.
            
        Returns
        -------
        dict
            Summary statistics.
        """
        congested_periods = results.get_congested_periods()
        total_duration = results.get_total_congestion_duration()
        
        if results.inferences:
            total_analysis_time = (
                max(inf.end_epoch for inf in results.inferences) -
                min(inf.start_epoch for inf in results.inferences)
            )
            congestion_ratio = total_duration / total_analysis_time if total_analysis_time > 0 else 0
        else:
            total_analysis_time = 0
            congestion_ratio = 0
        
        return {
            'total_periods': len(results.inferences),
            'congested_periods': len(congested_periods),
            'total_duration_seconds': total_analysis_time,
            'congestion_duration_seconds': total_duration,
            'congestion_ratio': congestion_ratio,
            'average_confidence': (
                sum(inf.confidence for inf in congested_periods) / len(congested_periods)
                if congested_periods else 0
            ),
            'metadata': results.metadata
        }
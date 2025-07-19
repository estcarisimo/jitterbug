"""
Jitter analysis implementation.
"""

import logging
from typing import List
from datetime import datetime
import numpy as np
import scipy.stats

from ..models import (
    RTTDataset,
    MinimumRTTDataset,
    ChangePoint,
    JitterAnalysis,
    JitterAnalysisConfig
)


logger = logging.getLogger(__name__)


class JitterAnalyzer:
    """
    Analyzer for jitter-based congestion detection.
    
    Supports two methods:
    1. Jitter dispersion analysis
    2. Kolmogorov-Smirnov test
    
    Parameters
    ----------
    config : JitterAnalysisConfig
        Configuration for jitter analysis.
    """
    
    def __init__(self, config: JitterAnalysisConfig):
        """
        Initialize the jitter analyzer.
        
        Parameters
        ----------
        config : JitterAnalysisConfig
            Configuration for jitter analysis.
        """
        self.config = config
    
    def analyze_jitter_dispersion(
        self,
        dataset: MinimumRTTDataset,
        change_points: List[ChangePoint]
    ) -> List[JitterAnalysis]:
        """
        Analyze jitter using dispersion method.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset containing minimum RTT values.
        change_points : List[ChangePoint]
            List of detected change points.
            
        Returns
        -------
        List[JitterAnalysis]
            List of jitter analysis results.
        """
        logger.info("Analyzing jitter using dispersion method")
        
        if len(change_points) < 2:
            logger.warning("Need at least 2 change points for jitter analysis")
            return []
        
        # Convert to arrays
        epochs, rtt_values = dataset.to_arrays()
        
        # Compute jitter dispersion
        jitter_epochs, jitter_dispersion = self._compute_jitter_dispersion(
            epochs, rtt_values
        )
        
        # Analyze jitter for each period between change points
        results = []
        
        for i in range(len(change_points) - 1):
            start_cp = change_points[i]
            end_cp = change_points[i + 1]
            
            # Get jitter values for this period
            period_mask = (jitter_epochs >= start_cp.epoch) & (jitter_epochs <= end_cp.epoch)
            
            if np.sum(period_mask) < 2:
                continue
            
            period_jitter = jitter_dispersion[period_mask]
            
            # Compare with previous period if available
            if i > 0:
                prev_cp = change_points[i - 1]
                prev_mask = (jitter_epochs >= prev_cp.epoch) & (jitter_epochs < start_cp.epoch)
                
                if np.sum(prev_mask) > 0:
                    prev_jitter = jitter_dispersion[prev_mask]
                    
                    # Test for significant increase (original v1 logic)
                    mean_increase = np.mean(period_jitter) > (np.mean(prev_jitter) + self.config.threshold)
                    jitter_metric = np.mean(period_jitter) - np.mean(prev_jitter)
                    
                    results.append(JitterAnalysis(
                        start_timestamp=start_cp.timestamp,
                        end_timestamp=end_cp.timestamp,
                        start_epoch=start_cp.epoch,
                        end_epoch=end_cp.epoch,
                        has_significant_jitter=mean_increase,
                        jitter_metric=float(jitter_metric),
                        method='jitter_dispersion',
                        threshold=self.config.threshold
                    ))
        
        return results
    
    def analyze_ks_test(
        self,
        dataset: RTTDataset,
        change_points: List[ChangePoint]
    ) -> List[JitterAnalysis]:
        """
        Analyze jitter using Kolmogorov-Smirnov test.
        
        Parameters
        ----------
        dataset : RTTDataset
            Dataset containing RTT measurements.
        change_points : List[ChangePoint]
            List of detected change points.
            
        Returns
        -------
        List[JitterAnalysis]
            List of jitter analysis results.
        """
        logger.info("Analyzing jitter using Kolmogorov-Smirnov test")
        
        if len(change_points) < 2:
            logger.warning("Need at least 2 change points for jitter analysis")
            return []
        
        # Convert to arrays
        epochs, rtt_values = dataset.to_arrays()
        
        # Compute jitter
        jitter_epochs, jitter_values = self._compute_jitter(epochs, rtt_values)
        
        # Analyze jitter for each period between change points
        results = []
        
        for i in range(len(change_points) - 1):
            start_cp = change_points[i]
            end_cp = change_points[i + 1]
            
            # Get jitter values for this period
            period_mask = (jitter_epochs >= start_cp.epoch) & (jitter_epochs <= end_cp.epoch)
            
            if np.sum(period_mask) < 2:
                continue
            
            period_jitter = jitter_values[period_mask]
            
            # Compare with previous period if available
            if i > 0:
                prev_cp = change_points[i - 1]
                prev_mask = (jitter_epochs >= prev_cp.epoch) & (jitter_epochs < start_cp.epoch)
                
                if np.sum(prev_mask) > 0:
                    prev_jitter = jitter_values[prev_mask]
                    
                    # Perform KS test (original v1 logic)
                    try:
                        ks_stat, p_value = scipy.stats.ks_2samp(prev_jitter, period_jitter)
                        has_significant_jitter = p_value < self.config.significance_level
                        
                        results.append(JitterAnalysis(
                            start_timestamp=start_cp.timestamp,
                            end_timestamp=end_cp.timestamp,
                            start_epoch=start_cp.epoch,
                            end_epoch=end_cp.epoch,
                            has_significant_jitter=has_significant_jitter,
                            jitter_metric=float(ks_stat),
                            method='ks_test',
                            threshold=self.config.significance_level,
                            p_value=float(p_value)
                        ))
                    
                    except Exception as e:
                        logger.warning(f"KS test failed for period {i}: {e}")
                        continue
        
        return results
    
    def _compute_jitter(self, epochs: np.ndarray, rtt_values: np.ndarray) -> tuple:
        """
        Compute jitter values from RTT measurements.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoch timestamps.
        rtt_values : np.ndarray
            RTT values.
            
        Returns
        -------
        tuple
            Tuple of (jitter_epochs, jitter_values).
        """
        if len(epochs) < 2:
            raise ValueError("Need at least 2 samples to compute jitter")
        
        # Jitter is the difference between consecutive RTT measurements
        jitter_epochs = epochs[1:]
        jitter_values = rtt_values[1:] - rtt_values[:-1]
        
        return jitter_epochs, jitter_values
    
    def _compute_jitter_dispersion(
        self,
        epochs: np.ndarray,
        rtt_values: np.ndarray
    ) -> tuple:
        """
        Compute jitter dispersion using moving average and IQR filtering.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoch timestamps.
        rtt_values : np.ndarray
            RTT values.
            
        Returns
        -------
        tuple
            Tuple of (dispersion_epochs, dispersion_values).
        """
        # First compute basic jitter
        jitter_epochs, jitter_values = self._compute_jitter(epochs, rtt_values)
        
        # Apply moving IQR filter
        filtered_jitter = self._moving_iqr_filter(
            jitter_values, self.config.moving_iqr_order
        )
        
        # Apply moving average filter
        final_jitter = self._moving_average_filter(
            filtered_jitter, self.config.moving_average_order
        )
        
        # Adjust epochs for filter lengths
        filter_offset = self.config.moving_iqr_order + self.config.moving_average_order // 2
        final_epochs = jitter_epochs[filter_offset:filter_offset + len(final_jitter)]
        
        return final_epochs, final_jitter
    
    def _moving_iqr_filter(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving IQR filter to values.
        
        Parameters
        ----------
        values : np.ndarray
            Input values.
        window_size : int
            Window size for IQR calculation.
            
        Returns
        -------
        np.ndarray
            Filtered values.
        """
        if len(values) < window_size * 2 + 1:
            return np.array([])
        
        iqr_values = []
        
        for i in range(window_size, len(values) - window_size):
            window = values[i - window_size:i + window_size + 1]
            q1, q3 = np.percentile(window, [25, 75])
            iqr_values.append(q3 - q1)
        
        return np.array(iqr_values)
    
    def _moving_average_filter(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving average filter to values.
        
        Parameters
        ----------
        values : np.ndarray
            Input values.
        window_size : int
            Window size for moving average.
            
        Returns
        -------
        np.ndarray
            Filtered values.
        """
        if len(values) < window_size:
            return np.array([])
        
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')
"""
Latency jump analysis implementation.
"""

import logging
from typing import List
import numpy as np

from ..models import (
    MinimumRTTDataset,
    ChangePoint,
    LatencyJump,
    LatencyJumpConfig,
)


logger = logging.getLogger(__name__)


class LatencyJumpAnalyzer:
    """
    Analyzer for detecting significant latency jumps.
    
    Identifies periods where the baseline latency increases significantly
    compared to the previous period.
    
    Parameters
    ----------
    config : LatencyJumpConfig
        Configuration for latency jump analysis.
    """
    
    def __init__(self, config: LatencyJumpConfig):
        """
        Initialize the latency jump analyzer.
        
        Parameters
        ----------
        config : LatencyJumpConfig
            Configuration for latency jump analysis.
        """
        self.config = config
    
    def analyze(
        self,
        dataset: MinimumRTTDataset,
        change_points: List[ChangePoint]
    ) -> List[LatencyJump]:
        """
        Analyze latency jumps between change points.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset containing minimum RTT values.
        change_points : List[ChangePoint]
            List of detected change points.
            
        Returns
        -------
        List[LatencyJump]
            List of latency jump analysis results.
        """
        logger.info("Analyzing latency jumps")
        
        if len(change_points) < 2:
            logger.warning("Need at least 2 change points for latency jump analysis")
            return []
        
        # Convert to arrays
        epochs, rtt_values = dataset.to_arrays()
        
        # Analyze jumps for each period between change points
        results = []
        
        for i in range(len(change_points) - 1):
            start_cp = change_points[i]
            end_cp = change_points[i + 1]
            
            # Get RTT values for this period
            period_mask = (epochs >= start_cp.epoch) & (epochs <= end_cp.epoch)
            
            if np.sum(period_mask) < 1:
                continue
            
            period_rtt = rtt_values[period_mask]
            
            # Compare with previous period if available
            if i > 0:
                prev_cp = change_points[i - 1]
                prev_mask = (epochs >= prev_cp.epoch) & (epochs < start_cp.epoch)
                
                if np.sum(prev_mask) > 0:
                    prev_rtt = rtt_values[prev_mask]
                    
                    # Calculate mean RTT for both periods
                    mean_prev = np.mean(prev_rtt)
                    mean_current = np.mean(period_rtt)
                    
                    # Check for significant jump
                    magnitude = mean_current - mean_prev
                    has_jump = magnitude > self.config.threshold
                    
                    results.append(LatencyJump(
                        start_timestamp=start_cp.timestamp,
                        end_timestamp=end_cp.timestamp,
                        start_epoch=start_cp.epoch,
                        end_epoch=end_cp.epoch,
                        has_jump=has_jump,
                        magnitude=float(magnitude),
                        threshold=self.config.threshold
                    ))
        
        return results
    
    def analyze_detailed(
        self,
        dataset: MinimumRTTDataset,
        change_points: List[ChangePoint]
    ) -> List[LatencyJump]:
        """
        Perform detailed latency jump analysis with additional statistics.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset containing minimum RTT values.
        change_points : List[ChangePoint]
            List of detected change points.
            
        Returns
        -------
        List[LatencyJump]
            List of detailed latency jump analysis results.
        """
        logger.info("Performing detailed latency jump analysis")
        
        if len(change_points) < 2:
            logger.warning("Need at least 2 change points for latency jump analysis")
            return []
        
        # Convert to arrays
        epochs, rtt_values = dataset.to_arrays()
        
        # Analyze jumps for each period between change points
        results = []
        
        for i in range(len(change_points) - 1):
            start_cp = change_points[i]
            end_cp = change_points[i + 1]
            
            # Get RTT values for this period
            period_mask = (epochs >= start_cp.epoch) & (epochs <= end_cp.epoch)
            
            if np.sum(period_mask) < 1:
                continue
            
            period_rtt = rtt_values[period_mask]
            
            # Compare with previous period if available
            if i > 0:
                prev_cp = change_points[i - 1]
                prev_mask = (epochs >= prev_cp.epoch) & (epochs < start_cp.epoch)
                
                if np.sum(prev_mask) > 0:
                    prev_rtt = rtt_values[prev_mask]
                    
                    # Calculate detailed statistics
                    mean_prev = np.mean(prev_rtt)
                    mean_current = np.mean(period_rtt)
                    std_prev = np.std(prev_rtt)
                    
                    # Calculate magnitude and significance
                    magnitude = mean_current - mean_prev
                    
                    # Use a more sophisticated jump detection
                    # Consider both mean change and variance
                    normalized_magnitude = magnitude / (std_prev + 1e-6)  # Avoid division by zero
                    
                    # Adaptive threshold based on variance
                    adaptive_threshold = self.config.threshold * (1 + std_prev / mean_prev)
                    
                    has_jump = (magnitude > adaptive_threshold) and (normalized_magnitude > 1.0)
                    
                    # Create extended LatencyJump with additional metadata
                    jump = LatencyJump(
                        start_timestamp=start_cp.timestamp,
                        end_timestamp=end_cp.timestamp,
                        start_epoch=start_cp.epoch,
                        end_epoch=end_cp.epoch,
                        has_jump=has_jump,
                        magnitude=float(magnitude),
                        threshold=self.config.threshold
                    )
                    
                    results.append(jump)
        
        return results
    
    def get_jump_statistics(self, latency_jumps: List[LatencyJump]) -> dict:
        """
        Calculate statistics for latency jumps.
        
        Parameters
        ----------
        latency_jumps : List[LatencyJump]
            List of latency jump results.
            
        Returns
        -------
        dict
            Dictionary containing jump statistics.
        """
        if not latency_jumps:
            return {
                'total_periods': 0,
                'jump_periods': 0,
                'jump_ratio': 0.0,
                'average_magnitude': 0.0,
                'max_magnitude': 0.0,
                'min_magnitude': 0.0
            }
        
        jump_periods = [jump for jump in latency_jumps if jump.has_jump]
        magnitudes = [jump.magnitude for jump in latency_jumps if jump.has_jump]
        
        return {
            'total_periods': len(latency_jumps),
            'jump_periods': len(jump_periods),
            'jump_ratio': len(jump_periods) / len(latency_jumps),
            'average_magnitude': float(np.mean(magnitudes)) if magnitudes else 0.0,
            'max_magnitude': float(np.max(magnitudes)) if magnitudes else 0.0,
            'min_magnitude': float(np.min(magnitudes)) if magnitudes else 0.0,
            'threshold_used': self.config.threshold
        }
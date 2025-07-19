"""
Main change point detection interface.
"""

import logging
from typing import List
from datetime import datetime

from ..models import MinimumRTTDataset, ChangePoint, ChangePointDetectionConfig
from .algorithms import RupturesDetector


logger = logging.getLogger(__name__)


class ChangePointDetector:
    """
    Main interface for change point detection algorithms.
    
    This class provides a unified interface for different change point detection
    algorithms and handles algorithm selection based on configuration.
    
    Parameters
    ----------
    config : ChangePointDetectionConfig
        Configuration for change point detection.
        
    Attributes
    ----------
    config : ChangePointDetectionConfig
        Configuration object.
    algorithm : BaseChangePointDetector
        Selected change point detection algorithm.
    """
    
    def __init__(self, config: ChangePointDetectionConfig):
        """
        Initialize the change point detector.
        
        Parameters
        ----------
        config : ChangePointDetectionConfig
            Configuration for change point detection.
        """
        self.config = config
        self.algorithm = self._create_algorithm()
    
    def _create_algorithm(self):
        """
        Create the appropriate change point detection algorithm.
        
        Returns
        -------
        BaseChangePointDetector
            Change point detection algorithm instance.
        """
        if self.config.algorithm == 'ruptures':
            return RupturesDetector(self.config)
        elif self.config.algorithm == 'bcp':
            try:
                from .algorithms import BayesianChangePointDetector
                return BayesianChangePointDetector(self.config)
            except ImportError as e:
                raise ImportError(
                    "bayesian_changepoint_detection package is required for BCP detection. "
                    "Install it with: pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git"
                ) from e
        elif self.config.algorithm == 'torch':
            try:
                from .algorithms import TorchChangePointDetector
                return TorchChangePointDetector(self.config)
            except ImportError as e:
                raise ImportError(
                    "PyTorch is required for neural network detection. "
                    "Install it with: pip install jitterbug[torch]"
                ) from e
        elif self.config.algorithm == 'rbeast':
            try:
                from .algorithms import RbeastDetector
                return RbeastDetector(self.config)
            except ImportError as e:
                raise ImportError(
                    "Rbeast is required for Rbeast detection. "
                    "Install it with: pip install Rbeast"
                ) from e
        elif self.config.algorithm == 'adtk':
            try:
                from .algorithms import ADTKDetector
                return ADTKDetector(self.config)
            except ImportError as e:
                raise ImportError(
                    "ADTK is required for ADTK detection. "
                    "Install it with: pip install adtk"
                ) from e
        else:
            raise ValueError(f"Unknown change point detection algorithm: {self.config.algorithm}")
    
    def detect(self, dataset: MinimumRTTDataset) -> List[ChangePoint]:
        """
        Detect change points in the RTT dataset.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset containing minimum RTT values.
            
        Returns
        -------
        List[ChangePoint]
            List of detected change points.
        """
        logger.info(f"Detecting change points using {self.config.algorithm} algorithm")
        
        if len(dataset) < 2:
            logger.warning("Insufficient data for change point detection")
            return []
        
        # Run the algorithm
        change_points = self.algorithm.detect(dataset)
        
        # Apply post-processing filters
        change_points = self._filter_change_points(change_points)
        
        logger.info(f"Detected {len(change_points)} change points")
        return change_points
    
    def _filter_change_points(self, change_points: List[ChangePoint]) -> List[ChangePoint]:
        """
        Apply post-processing filters to change points.
        
        Parameters
        ----------
        change_points : List[ChangePoint]
            Raw change points from algorithm.
            
        Returns
        -------
        List[ChangePoint]
            Filtered change points.
        """
        if not change_points:
            return change_points
        
        # Sort by timestamp
        change_points.sort(key=lambda cp: cp.epoch)
        
        # Filter by minimum time elapsed
        filtered_points = []
        
        for cp in change_points:
            if not filtered_points:
                # First change point
                filtered_points.append(cp)
            else:
                # Check minimum time elapsed
                time_diff = cp.epoch - filtered_points[-1].epoch
                if time_diff >= self.config.min_time_elapsed:
                    filtered_points.append(cp)
                else:
                    # Keep the one with higher confidence
                    if cp.confidence > filtered_points[-1].confidence:
                        filtered_points[-1] = cp
        
        # Limit maximum number of change points
        if self.config.max_change_points is not None:
            if len(filtered_points) > self.config.max_change_points:
                # Keep the ones with highest confidence
                filtered_points.sort(key=lambda cp: cp.confidence, reverse=True)
                filtered_points = filtered_points[:self.config.max_change_points]
                # Re-sort by timestamp
                filtered_points.sort(key=lambda cp: cp.epoch)
        
        return filtered_points
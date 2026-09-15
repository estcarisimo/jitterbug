"""
Change point detection algorithm implementations.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np

from ..models import ChangePoint, ChangePointDetectionConfig, MinimumRTTDataset

logger = logging.getLogger(__name__)


class BaseChangePointDetector(ABC):
    """
    Base class for change point detection algorithms.

    Parameters
    ----------
    config : ChangePointDetectionConfig
        Configuration for change point detection.
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

    @abstractmethod
    def detect(self, dataset: MinimumRTTDataset) -> list[ChangePoint]:
        """
        Detect change points in the dataset.

        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset to analyze.

        Returns
        -------
        List[ChangePoint]
            List of detected change points.
        """
        pass


class RupturesDetector(BaseChangePointDetector):
    """
    Change point detection using the ruptures library.

    This implementation uses the ruptures library which provides multiple
    algorithms for change point detection including Pelt, BottomUp, and others.
    """

    def __init__(self, config: ChangePointDetectionConfig):
        """
        Initialize the ruptures detector.

        Parameters
        ----------
        config : ChangePointDetectionConfig
            Configuration for change point detection.
        """
        super().__init__(config)

        try:
            import ruptures as rpt

            self.rpt = rpt
        except ImportError as e:
            raise ImportError(
                "ruptures package is required for ruptures change point detection. "
                "Install it with: pip install ruptures"
            ) from e

    def detect(self, dataset: MinimumRTTDataset) -> list[ChangePoint]:
        """
        Detect change points using ruptures library.

        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset to analyze.

        Returns
        -------
        List[ChangePoint]
            List of detected change points.
        """
        epochs, rtt_values = dataset.to_arrays()

        # Prepare data for ruptures
        signal = rtt_values.reshape(-1, 1)

        # Choose algorithm based on configuration
        if self.config.ruptures_model == "rbf":
            algo = self.rpt.Pelt(model="rbf").fit(signal)
        elif self.config.ruptures_model == "l1":
            algo = self.rpt.Pelt(model="l1").fit(signal)
        elif self.config.ruptures_model == "l2":
            algo = self.rpt.Pelt(model="l2").fit(signal)
        elif self.config.ruptures_model == "normal":
            algo = self.rpt.Pelt(model="normal").fit(signal)
        else:
            logger.warning(f"Unknown ruptures model {self.config.ruptures_model}, using 'rbf'")
            algo = self.rpt.Pelt(model="rbf").fit(signal)

        # Detect change points with adaptive penalty
        # Scale penalty based on threshold - lower threshold means lower penalty
        adaptive_penalty = self.config.ruptures_penalty * (1.0 - self.config.threshold)
        adaptive_penalty = max(0.1, adaptive_penalty)  # Ensure minimum penalty

        change_point_indices = algo.predict(pen=adaptive_penalty)

        # Convert indices to change points with filtering
        change_points = []
        last_change_time = -float("inf")

        for idx in change_point_indices:
            if idx < len(epochs):  # ruptures sometimes returns len(signal) as last point
                epoch = epochs[idx]

                # Apply minimum time elapsed constraint
                if epoch - last_change_time >= self.config.min_time_elapsed:
                    timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)

                    # Calculate confidence based on signal variance around the change point
                    confidence = self._calculate_confidence(signal, idx)

                    change_points.append(
                        ChangePoint(
                            timestamp=timestamp,
                            epoch=epoch,
                            confidence=confidence,
                            algorithm=f"ruptures_{self.config.ruptures_model}",
                        )
                    )

                    last_change_time = epoch

        # If we still have too few change points, try with lower penalty
        if len(change_points) < 10:
            logger.info(
                f"Only {len(change_points)} change points found, retrying with lower penalty"
            )
            lower_penalty = adaptive_penalty * 0.3
            change_point_indices = algo.predict(pen=lower_penalty)

            change_points = []
            last_change_time = -float("inf")

            for idx in change_point_indices:
                if idx < len(epochs):
                    epoch = epochs[idx]

                    # Apply minimum time elapsed constraint
                    if epoch - last_change_time >= self.config.min_time_elapsed:
                        timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
                        confidence = self._calculate_confidence(signal, idx)

                        change_points.append(
                            ChangePoint(
                                timestamp=timestamp,
                                epoch=epoch,
                                confidence=confidence,
                                algorithm=f"ruptures_{self.config.ruptures_model}_lowpen",
                            )
                        )

                        last_change_time = epoch

        return change_points

    def _calculate_confidence(self, signal: np.ndarray, idx: int) -> float:
        """
        Calculate confidence score for a change point.

        Parameters
        ----------
        signal : np.ndarray
            Signal data.
        idx : int
            Index of the change point.

        Returns
        -------
        float
            Confidence score between 0 and 1.
        """
        # Simple confidence calculation based on variance difference
        # before and after the change point
        window_size = min(10, idx, len(signal) - idx)

        if window_size < 2:
            return 0.5  # Default confidence

        before = signal[max(0, idx - window_size) : idx]
        after = signal[idx : idx + window_size]

        if len(before) == 0 or len(after) == 0:
            return 0.5

        var_before = np.var(before)
        var_after = np.var(after)
        mean_before = np.mean(before)
        mean_after = np.mean(after)

        # Confidence based on mean difference and variance change
        mean_diff = abs(mean_after - mean_before)
        var_change = abs(var_after - var_before)

        # Normalize to [0, 1] range
        confidence = min(1.0, (mean_diff + var_change) / 10.0)

        return float(max(0.0, confidence))


class BayesianChangePointDetector(BaseChangePointDetector):
    """
    Change point detection using Bayesian methods.

    This is a modernized version of the original BCP implementation
    with improved error handling and configuration.
    """

    def __init__(self, config: ChangePointDetectionConfig):
        """
        Initialize the Bayesian change point detector.

        Parameters
        ----------
        config : ChangePointDetectionConfig
            Configuration for change point detection.
        """
        super().__init__(config)

        try:
            import bayesian_changepoint_detection.bayesian_models as bm
            import bayesian_changepoint_detection.offline_likelihoods as ol
            import bayesian_changepoint_detection.priors as pr

            self.offline_changepoint_detection = bm.offline_changepoint_detection
            self.offline_likelihoods = ol
            self.priors = pr

        except ImportError as e:
            raise ImportError(
                "bayesian_changepoint_detection package is required for BCP detection. "
                "Install it from: https://github.com/estcarisimo/bayesian_changepoint_detection"
            ) from e

    def detect(self, dataset: MinimumRTTDataset) -> list[ChangePoint]:
        """
        Detect change points using Bayesian change point detection.

        Parameters
        ----------
        dataset : MinimumRTTDataset
            Dataset to analyze.

        Returns
        -------
        List[ChangePoint]
            List of detected change points.
        """
        epochs, rtt_values = dataset.to_arrays()

        # Minimum samples check
        if len(rtt_values) < 2:
            logger.warning("Insufficient samples for Bayesian change point detection")
            return []

        try:
            # Set up prior
            def prior_function(x: int) -> float:
                return float(self.priors.const_prior(x, p=1 / (len(rtt_values) + 1)))

            # Run offline change point detection. The library defaults to a GPU when one
            # is visible (MPS on Apple Silicon), which is an order of magnitude slower than
            # CPU for series of this size, so the device is explicit.
            device = self.config.bcp_device
            _Q, _P, Pcp = self.offline_changepoint_detection(
                rtt_values,
                prior_function,
                self.offline_likelihoods.StudentT(device=device),
                truncate=-40,
                device=device,
            )

            # Calculate change point probabilities (Pcp is log-scale, one row per run length)
            if hasattr(Pcp, "detach"):
                Pcp = Pcp.detach().cpu().numpy()
            change_point_probs = np.exp(Pcp).sum(0)

            # Find significant change points
            significant_indices = np.where(change_point_probs > self.config.threshold)[0]

            # Convert to change points
            change_points = []
            for idx in significant_indices:
                if idx < len(epochs):
                    epoch = epochs[idx]
                    timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    confidence = float(change_point_probs[idx])

                    change_points.append(
                        ChangePoint(
                            timestamp=timestamp,
                            epoch=epoch,
                            confidence=confidence,
                            algorithm="bayesian_cp",
                        )
                    )

            return change_points

        except Exception as e:
            # Do not turn a crash into "no change points found": surface it.
            raise RuntimeError(f"Bayesian change point detection failed: {e}") from e

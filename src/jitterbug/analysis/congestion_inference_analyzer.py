"""
Congestion inference analysis implementation.
"""

import logging

import numpy as np

from ..models import CongestionInference, JitterAnalysis, LatencyJump

logger = logging.getLogger(__name__)


class CongestionInferenceAnalyzer:
    """
    Analyzer for inferring congestion based on latency jumps and jitter analysis.

    Combines results from latency jump detection and jitter analysis to make
    final congestion inferences.
    """

    def __init__(self) -> None:
        """Initialize the congestion inference analyzer."""

    def infer(
        self, latency_jumps: list[LatencyJump], jitter_analyses: list[JitterAnalysis]
    ) -> list[CongestionInference]:
        """
        Infer congestion periods based on latency jumps and jitter analysis.

        Uses the original v1 stateful congestion inference logic:
        - Congestion = True when BOTH latency jump AND jitter are detected
        - Congestion = False when there's NO latency jump (regardless of jitter)
        - Maintains congestion state between periods

        Parameters
        ----------
        latency_jumps : List[LatencyJump]
            List of latency jump analysis results.
        jitter_analyses : List[JitterAnalysis]
            List of jitter analysis results.

        Returns
        -------
        List[CongestionInference]
            List of congestion inference results.
        """
        logger.info("Inferring congestion periods")

        if not latency_jumps or not jitter_analyses:
            logger.warning("No latency jumps or jitter analyses available")
            return []

        # Match latency jumps with jitter analyses
        matched_pairs = []
        for jump in latency_jumps:
            # Find corresponding jitter analysis
            corresponding_jitter = None
            for jitter in jitter_analyses:
                if (
                    abs(jump.start_epoch - jitter.start_epoch) < 1
                    and abs(jump.end_epoch - jitter.end_epoch) < 1
                ):
                    corresponding_jitter = jitter
                    break

            if corresponding_jitter is None:
                logger.warning(
                    "No corresponding jitter analysis found for jump "
                    f"{jump.start_epoch}-{jump.end_epoch}"
                )
                continue

            matched_pairs.append((jump, corresponding_jitter))

        if not matched_pairs:
            logger.warning("No matched pairs of latency jumps and jitter analyses")
            return []

        # Apply exact v1-style stateful congestion inference
        inferences = []
        congestion_state = False  # Track congestion state across periods

        for jump, jitter in matched_pairs:
            # Exact v1 logic from cong_inference.py lines 49-52
            if jump.has_jump and jitter.has_significant_jitter:
                # Both latency jump AND jitter detected -> congestion = True
                congestion_state = True
            elif not jump.has_jump:
                # No latency jump -> congestion = False (regardless of jitter)
                congestion_state = False
            # else: jump but no jitter -> maintain previous congestion state (no change)

            # Calculate confidence
            if congestion_state:
                confidence = 0.8
                if jump.magnitude > jump.threshold * 2:
                    confidence += 0.1
                confidence = min(1.0, confidence)
            else:
                confidence = 0.0

            inference = CongestionInference(
                start_timestamp=jump.start_timestamp,
                end_timestamp=jump.end_timestamp,
                start_epoch=jump.start_epoch,
                end_epoch=jump.end_epoch,
                is_congested=congestion_state,
                confidence=confidence,
                latency_jump=jump,
                jitter_analysis=jitter,
            )

            inferences.append(inference)

        return inferences

    def get_inference_statistics(self, inferences: list[CongestionInference]) -> dict:
        """
        Calculate statistics for congestion inferences.

        Parameters
        ----------
        inferences : List[CongestionInference]
            List of congestion inference results.

        Returns
        -------
        dict
            Dictionary containing inference statistics.
        """
        if not inferences:
            return {
                "total_periods": 0,
                "congested_periods": 0,
                "congestion_ratio": 0.0,
                "average_confidence": 0.0,
                "total_duration": 0.0,
                "congestion_duration": 0.0,
            }

        congested_periods = [inf for inf in inferences if inf.is_congested]
        total_duration = sum(inf.end_epoch - inf.start_epoch for inf in inferences)
        congestion_duration = sum(inf.end_epoch - inf.start_epoch for inf in congested_periods)

        return {
            "total_periods": len(inferences),
            "congested_periods": len(congested_periods),
            "congestion_ratio": len(congested_periods) / len(inferences),
            "average_confidence": float(np.mean([inf.confidence for inf in congested_periods]))
            if congested_periods
            else 0.0,
            "total_duration": total_duration,
            "congestion_duration": congestion_duration,
            "congestion_time_ratio": congestion_duration / total_duration
            if total_duration > 0
            else 0.0,
        }

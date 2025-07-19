"""
Congestion inference analysis implementation.
"""

import logging
from typing import List
from datetime import datetime
import numpy as np

from ..models import (
    LatencyJump,
    JitterAnalysis,
    CongestionInference
)


logger = logging.getLogger(__name__)


class CongestionInferenceAnalyzer:
    """
    Analyzer for inferring congestion based on latency jumps and jitter analysis.
    
    Combines results from latency jump detection and jitter analysis to make
    final congestion inferences.
    """
    
    def __init__(self):
        """Initialize the congestion inference analyzer."""
        pass
    
    def infer(
        self,
        latency_jumps: List[LatencyJump],
        jitter_analyses: List[JitterAnalysis]
    ) -> List[CongestionInference]:
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
                if (abs(jump.start_epoch - jitter.start_epoch) < 1 and 
                    abs(jump.end_epoch - jitter.end_epoch) < 1):
                    corresponding_jitter = jitter
                    break
            
            if corresponding_jitter is None:
                logger.warning(f"No corresponding jitter analysis found for jump {jump.start_epoch}-{jump.end_epoch}")
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
                jitter_analysis=jitter
            )
            
            inferences.append(inference)
        
        return inferences
    
    def _apply_inference_logic(
        self,
        jump: LatencyJump,
        jitter: JitterAnalysis
    ) -> tuple[bool, float]:
        """
        Apply congestion inference logic to determine if a period is congested.
        
        Parameters
        ----------
        jump : LatencyJump
            Latency jump analysis result.
        jitter : JitterAnalysis
            Jitter analysis result.
            
        Returns
        -------
        tuple[bool, float]
            Tuple of (is_congested, confidence).
        """
        # Basic inference logic: congestion is inferred when both
        # latency jump and jitter increase are detected
        
        # Primary condition: both jump and jitter must be significant
        if jump.has_jump and jitter.has_significant_jitter:
            # High confidence when both indicators agree
            confidence = 0.8
            
            # Increase confidence based on magnitude
            if jump.magnitude > jump.threshold * 2:
                confidence += 0.1
            
            if jitter.method == 'ks_test' and jitter.p_value is not None:
                # Lower p-value increases confidence
                confidence += 0.1 * (1 - jitter.p_value * 20)  # Scale p-value
            
            # Cap confidence at 1.0
            confidence = min(1.0, confidence)
            
            return True, confidence
        
        # Secondary condition: only latency jump without jitter change
        elif jump.has_jump and not jitter.has_significant_jitter:
            # Medium confidence - might be congestion but less certain
            confidence = 0.4
            
            # Increase confidence for large jumps
            if jump.magnitude > jump.threshold * 3:
                confidence += 0.2
            
            return True, confidence
        
        # Tertiary condition: only jitter increase without latency jump
        elif not jump.has_jump and jitter.has_significant_jitter:
            # Low confidence - jitter alone might not indicate congestion
            confidence = 0.2
            
            # Increase confidence for very significant jitter changes
            if jitter.method == 'ks_test' and jitter.p_value is not None:
                if jitter.p_value < 0.001:  # Very significant
                    confidence += 0.3
            elif jitter.method == 'jitter_dispersion':
                if jitter.jitter_metric > jitter.threshold * 2:
                    confidence += 0.3
            
            return True, confidence
        
        # No congestion detected
        else:
            return False, 0.0
    
    def _post_process_inferences(
        self,
        inferences: List[CongestionInference]
    ) -> List[CongestionInference]:
        """
        Post-process congestion inferences to smooth out isolated decisions.
        
        Parameters
        ----------
        inferences : List[CongestionInference]
            Raw congestion inferences.
            
        Returns
        -------
        List[CongestionInference]
            Post-processed congestion inferences.
        """
        if len(inferences) <= 1:
            return inferences
        
        # Sort by start time
        inferences.sort(key=lambda x: x.start_epoch)
        
        # Apply smoothing: if a non-congested period is surrounded by
        # congested periods, consider it congested too (and vice versa)
        processed = inferences.copy()
        
        for i in range(1, len(processed) - 1):
            current = processed[i]
            prev = processed[i - 1]
            next_inf = processed[i + 1]
            
            # Check if current inference is isolated
            if (current.is_congested != prev.is_congested and 
                current.is_congested != next_inf.is_congested):
                
                # Flip the decision but with reduced confidence
                new_is_congested = not current.is_congested
                new_confidence = min(current.confidence, 0.3)
                
                # Create new inference with updated decision
                processed[i] = CongestionInference(
                    start_timestamp=current.start_timestamp,
                    end_timestamp=current.end_timestamp,
                    start_epoch=current.start_epoch,
                    end_epoch=current.end_epoch,
                    is_congested=new_is_congested,
                    confidence=new_confidence,
                    latency_jump=current.latency_jump,
                    jitter_analysis=current.jitter_analysis
                )
        
        return processed
    
    def get_inference_statistics(self, inferences: List[CongestionInference]) -> dict:
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
                'total_periods': 0,
                'congested_periods': 0,
                'congestion_ratio': 0.0,
                'average_confidence': 0.0,
                'total_duration': 0.0,
                'congestion_duration': 0.0
            }
        
        congested_periods = [inf for inf in inferences if inf.is_congested]
        total_duration = sum(inf.end_epoch - inf.start_epoch for inf in inferences)
        congestion_duration = sum(inf.end_epoch - inf.start_epoch for inf in congested_periods)
        
        return {
            'total_periods': len(inferences),
            'congested_periods': len(congested_periods),
            'congestion_ratio': len(congested_periods) / len(inferences),
            'average_confidence': float(np.mean([inf.confidence for inf in congested_periods])) if congested_periods else 0.0,
            'total_duration': total_duration,
            'congestion_duration': congestion_duration,
            'congestion_time_ratio': congestion_duration / total_duration if total_duration > 0 else 0.0
        }
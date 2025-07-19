"""
API models for Jitterbug REST API.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

from ..models import (
    RTTMeasurement as CoreRTTMeasurement,
    CongestionInference as CoreCongestionInference,
    ChangePoint as CoreChangePoint,
    JitterbugConfig as CoreJitterbugConfig
)


class RTTMeasurementAPI(BaseModel):
    """API model for RTT measurements."""
    timestamp: datetime = Field(..., description="Measurement timestamp")
    epoch: float = Field(..., description="Unix timestamp")
    rtt_value: float = Field(..., gt=0, description="RTT value in milliseconds")
    source: Optional[str] = Field(None, description="Source IP address")
    destination: Optional[str] = Field(None, description="Destination IP address")


class RTTDatasetAPI(BaseModel):
    """API model for RTT datasets."""
    measurements: List[RTTMeasurementAPI] = Field(..., description="List of RTT measurements")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Dataset metadata")


class ChangePointAPI(BaseModel):
    """API model for change points."""
    timestamp: datetime = Field(..., description="Change point timestamp")
    epoch: float = Field(..., description="Unix timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    algorithm: str = Field(..., description="Detection algorithm used")


class LatencyJumpAPI(BaseModel):
    """API model for latency jumps."""
    start_timestamp: datetime = Field(..., description="Start timestamp")
    end_timestamp: datetime = Field(..., description="End timestamp")
    start_epoch: float = Field(..., description="Start unix timestamp")
    end_epoch: float = Field(..., description="End unix timestamp")
    has_jump: bool = Field(..., description="Whether a jump was detected")
    magnitude: float = Field(..., description="Jump magnitude in milliseconds")
    threshold: float = Field(..., description="Detection threshold")


class JitterAnalysisAPI(BaseModel):
    """API model for jitter analysis."""
    start_timestamp: datetime = Field(..., description="Start timestamp")
    end_timestamp: datetime = Field(..., description="End timestamp")
    start_epoch: float = Field(..., description="Start unix timestamp")
    end_epoch: float = Field(..., description="End unix timestamp")
    has_significant_jitter: bool = Field(..., description="Whether significant jitter was detected")
    jitter_metric: float = Field(..., description="Jitter metric value")
    method: str = Field(..., description="Analysis method used")
    threshold: float = Field(..., description="Detection threshold")
    p_value: Optional[float] = Field(None, description="Statistical p-value if applicable")


class CongestionInferenceAPI(BaseModel):
    """API model for congestion inferences."""
    start_timestamp: datetime = Field(..., description="Period start timestamp")
    end_timestamp: datetime = Field(..., description="Period end timestamp")
    start_epoch: float = Field(..., description="Period start unix timestamp")
    end_epoch: float = Field(..., description="Period end unix timestamp")
    is_congested: bool = Field(..., description="Whether congestion was inferred")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    latency_jump: Optional[LatencyJumpAPI] = Field(None, description="Latency jump analysis")
    jitter_analysis: Optional[JitterAnalysisAPI] = Field(None, description="Jitter analysis")


class AnalysisResultAPI(BaseModel):
    """API model for complete analysis results."""
    inferences: List[CongestionInferenceAPI] = Field(..., description="Congestion inferences")
    change_points: List[ChangePointAPI] = Field(..., description="Detected change points")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")


class AnalysisRequestAPI(BaseModel):
    """API model for analysis requests."""
    data: RTTDatasetAPI = Field(..., description="RTT dataset to analyze")
    config: Optional[Dict[str, Any]] = Field(None, description="Analysis configuration")
    algorithm: Optional[str] = Field("ruptures", description="Change point detection algorithm")
    method: Optional[str] = Field("jitter_dispersion", description="Jitter analysis method")
    threshold: Optional[float] = Field(0.25, description="Detection threshold")
    min_time_elapsed: Optional[int] = Field(1800, description="Minimum time between change points")


class AnalysisResponseAPI(BaseModel):
    """API model for analysis responses."""
    success: bool = Field(..., description="Whether analysis succeeded")
    result: Optional[AnalysisResultAPI] = Field(None, description="Analysis results")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    request_id: str = Field(..., description="Unique request identifier")


class HealthCheckAPI(BaseModel):
    """API model for health check responses."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class ValidationRequestAPI(BaseModel):
    """API model for data validation requests."""
    data: RTTDatasetAPI = Field(..., description="RTT dataset to validate")


class ValidationResponseAPI(BaseModel):
    """API model for data validation responses."""
    valid: bool = Field(..., description="Whether data is valid")
    errors: List[str] = Field(..., description="Validation errors")
    warnings: List[str] = Field(..., description="Validation warnings")
    metrics: Dict[str, Any] = Field(..., description="Data quality metrics")


class AlgorithmComparisonRequestAPI(BaseModel):
    """API model for algorithm comparison requests."""
    data: RTTDatasetAPI = Field(..., description="RTT dataset to analyze")
    algorithms: List[str] = Field(..., description="Algorithms to compare")
    config: Optional[Dict[str, Any]] = Field(None, description="Common configuration")


class AlgorithmComparisonResponseAPI(BaseModel):
    """API model for algorithm comparison responses."""
    success: bool = Field(..., description="Whether comparison succeeded")
    results: Optional[Dict[str, List[ChangePointAPI]]] = Field(None, description="Results by algorithm")
    performance: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Total execution time in seconds")


class StatusAPI(BaseModel):
    """API model for service status."""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Status timestamp")
    requests_processed: int = Field(..., description="Total requests processed")
    errors: int = Field(..., description="Total errors")
    avg_response_time: float = Field(..., description="Average response time")


class ErrorResponseAPI(BaseModel):
    """API model for error responses."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


def convert_to_api_model(core_obj: Union[CoreRTTMeasurement, CoreCongestionInference, CoreChangePoint]) -> Union[RTTMeasurementAPI, CongestionInferenceAPI, ChangePointAPI]:
    """Convert core models to API models."""
    if isinstance(core_obj, CoreRTTMeasurement):
        return RTTMeasurementAPI(
            timestamp=core_obj.timestamp,
            epoch=core_obj.epoch,
            rtt_value=core_obj.rtt_value,
            source=core_obj.source,
            destination=core_obj.destination
        )
    elif isinstance(core_obj, CoreCongestionInference):
        return CongestionInferenceAPI(
            start_timestamp=core_obj.start_timestamp,
            end_timestamp=core_obj.end_timestamp,
            start_epoch=core_obj.start_epoch,
            end_epoch=core_obj.end_epoch,
            is_congested=core_obj.is_congested,
            confidence=core_obj.confidence,
            latency_jump=LatencyJumpAPI(
                start_timestamp=core_obj.latency_jump.start_timestamp,
                end_timestamp=core_obj.latency_jump.end_timestamp,
                start_epoch=core_obj.latency_jump.start_epoch,
                end_epoch=core_obj.latency_jump.end_epoch,
                has_jump=core_obj.latency_jump.has_jump,
                magnitude=core_obj.latency_jump.magnitude,
                threshold=core_obj.latency_jump.threshold
            ) if core_obj.latency_jump else None,
            jitter_analysis=JitterAnalysisAPI(
                start_timestamp=core_obj.jitter_analysis.start_timestamp,
                end_timestamp=core_obj.jitter_analysis.end_timestamp,
                start_epoch=core_obj.jitter_analysis.start_epoch,
                end_epoch=core_obj.jitter_analysis.end_epoch,
                has_significant_jitter=core_obj.jitter_analysis.has_significant_jitter,
                jitter_metric=core_obj.jitter_analysis.jitter_metric,
                method=core_obj.jitter_analysis.method,
                threshold=core_obj.jitter_analysis.threshold,
                p_value=core_obj.jitter_analysis.p_value
            ) if core_obj.jitter_analysis else None
        )
    elif isinstance(core_obj, CoreChangePoint):
        return ChangePointAPI(
            timestamp=core_obj.timestamp,
            epoch=core_obj.epoch,
            confidence=core_obj.confidence,
            algorithm=core_obj.algorithm
        )
    else:
        raise ValueError(f"Unknown core model type: {type(core_obj)}")


def convert_from_api_model(api_obj: Union[RTTMeasurementAPI, CongestionInferenceAPI, ChangePointAPI]) -> Union[CoreRTTMeasurement, CoreCongestionInference, CoreChangePoint]:
    """Convert API models to core models."""
    if isinstance(api_obj, RTTMeasurementAPI):
        return CoreRTTMeasurement(
            timestamp=api_obj.timestamp,
            epoch=api_obj.epoch,
            rtt_value=api_obj.rtt_value,
            source=api_obj.source,
            destination=api_obj.destination
        )
    elif isinstance(api_obj, ChangePointAPI):
        return CoreChangePoint(
            timestamp=api_obj.timestamp,
            epoch=api_obj.epoch,
            confidence=api_obj.confidence,
            algorithm=api_obj.algorithm
        )
    else:
        raise ValueError(f"Unknown API model type: {type(api_obj)}")
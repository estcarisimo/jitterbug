"""
Analysis result models using Pydantic for validation and serialization.
"""

from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ChangePoint(BaseModel):
    """
    Represents a detected change point in the time series.

    Attributes
    ----------
    timestamp : datetime
        When the change point occurred.
    epoch : float
        Unix timestamp of the change point.
    confidence : float
        Confidence score of the change point detection.
    algorithm : str
        Algorithm used to detect the change point.
    """

    timestamp: datetime
    epoch: float
    confidence: float = Field(ge=0, le=1, description="Confidence must be between 0 and 1")
    algorithm: str = Field(description="Algorithm used for detection")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class LatencyJump(BaseModel):
    """
    Represents a detected latency jump between two time periods.

    Attributes
    ----------
    start_timestamp : datetime
        Start of the period.
    end_timestamp : datetime
        End of the period.
    start_epoch : float
        Unix timestamp of period start.
    end_epoch : float
        Unix timestamp of period end.
    has_jump : bool
        Whether a significant latency jump was detected.
    magnitude : float
        Magnitude of the jump in RTT units.
    threshold : float
        Threshold used for jump detection.
    """

    start_timestamp: datetime
    end_timestamp: datetime
    start_epoch: float
    end_epoch: float
    has_jump: bool
    magnitude: float = Field(description="Jump magnitude in RTT units")
    threshold: float = Field(gt=0, description="Threshold used for detection")

    @field_validator("end_epoch")
    @classmethod
    def validate_time_order(cls, v: float, info: ValidationInfo) -> float:
        start = info.data.get("start_epoch")
        if start is not None and v <= start:
            raise ValueError("End epoch must be after start epoch")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class JitterAnalysis(BaseModel):
    """
    Represents results of jitter analysis.

    Attributes
    ----------
    start_timestamp : datetime
        Start of the analysis period.
    end_timestamp : datetime
        End of the analysis period.
    start_epoch : float
        Unix timestamp of period start.
    end_epoch : float
        Unix timestamp of period end.
    has_significant_jitter : bool
        Whether significant jitter was detected.
    jitter_metric : float
        Computed jitter metric value.
    method : Literal['jitter_dispersion', 'ks_test']
        Method used for jitter analysis.
    threshold : float
        Threshold used for significance testing.
    p_value : Optional[float]
        P-value if statistical test was used.
    """

    start_timestamp: datetime
    end_timestamp: datetime
    start_epoch: float
    end_epoch: float
    has_significant_jitter: bool
    jitter_metric: float
    method: Literal["jitter_dispersion", "ks_test"]
    threshold: float = Field(gt=0, description="Threshold for significance")
    p_value: float | None = Field(
        default=None, ge=0, le=1, description="P-value if statistical test used"
    )

    @field_validator("end_epoch")
    @classmethod
    def validate_time_order(cls, v: float, info: ValidationInfo) -> float:
        start = info.data.get("start_epoch")
        if start is not None and v <= start:
            raise ValueError("End epoch must be after start epoch")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v

    @field_validator("p_value")
    @classmethod
    def validate_p_value(cls, v: float | None) -> float | None:
        if v is not None and not 0 <= v <= 1:
            raise ValueError("P-value must be between 0 and 1")
        return v

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class CongestionInference(BaseModel):
    """
    Represents the final congestion inference for a time period.

    Attributes
    ----------
    start_timestamp : datetime
        Start of the congestion period.
    end_timestamp : datetime
        End of the congestion period.
    start_epoch : float
        Unix timestamp of period start.
    end_epoch : float
        Unix timestamp of period end.
    is_congested : bool
        Whether congestion was inferred.
    confidence : float
        Confidence in the congestion inference.
    latency_jump : Optional[LatencyJump]
        Associated latency jump analysis.
    jitter_analysis : Optional[JitterAnalysis]
        Associated jitter analysis.
    """

    start_timestamp: datetime
    end_timestamp: datetime
    start_epoch: float
    end_epoch: float
    is_congested: bool
    confidence: float = Field(ge=0, le=1, description="Confidence in inference")
    latency_jump: LatencyJump | None = None
    jitter_analysis: JitterAnalysis | None = None

    @field_validator("end_epoch")
    @classmethod
    def validate_time_order(cls, v: float, info: ValidationInfo) -> float:
        start = info.data.get("start_epoch")
        if start is not None and v <= start:
            raise ValueError("End epoch must be after start epoch")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    def to_dict(self) -> dict[str, float | bool]:
        """
        Convert to dictionary for backward compatibility.

        Returns
        -------
        dict
            Dictionary with keys: starts, ends, congestion.
        """
        return {"starts": self.start_epoch, "ends": self.end_epoch, "congestion": self.is_congested}

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class CongestionInferenceResult(BaseModel):
    """
    Container for multiple congestion inference results.

    Attributes
    ----------
    inferences : List[CongestionInference]
        List of congestion inferences.
    metadata : dict
        Additional metadata about the analysis.
    """

    inferences: list[CongestionInference]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame for backward compatibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: starts, ends, congestion.
        """
        data = []
        for inference in self.inferences:
            data.append(inference.to_dict())

        return pd.DataFrame(data)

    def get_congested_periods(self) -> list[CongestionInference]:
        """
        Get only the periods identified as congested.

        Returns
        -------
        List[CongestionInference]
            List of congested periods.
        """
        return [inf for inf in self.inferences if inf.is_congested]

    def get_total_congestion_duration(self) -> float:
        """
        Get total duration of congestion in seconds.

        Returns
        -------
        float
            Total congestion duration in seconds.
        """
        total_duration = 0.0
        for inference in self.inferences:
            if inference.is_congested:
                total_duration += inference.end_epoch - inference.start_epoch
        return total_duration

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

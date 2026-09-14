"""
Configuration models using Pydantic for validation and serialization.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChangePointDetectionConfig(BaseModel):
    """
    Configuration for change point detection algorithms.

    Attributes
    ----------
    algorithm : Literal['bcp', 'ruptures']
        Algorithm to use for change point detection.
    threshold : float
        Threshold for change point detection sensitivity.
    min_time_elapsed : int
        Minimum time in seconds between change points.
    max_change_points : Optional[int]
        Maximum number of change points to detect.
    """

    algorithm: Literal["bcp", "ruptures"] = Field(
        default="ruptures", description="Change point detection algorithm"
    )
    threshold: float = Field(default=0.25, ge=0, le=1, description="Detection threshold (0-1)")
    min_time_elapsed: int = Field(
        default=1800, gt=0, description="Minimum time between change points in seconds"
    )
    max_change_points: int | None = Field(
        default=None, gt=0, description="Maximum number of change points to detect"
    )

    # Algorithm-specific parameters
    ruptures_model: str = Field(default="rbf", description="Ruptures model type")
    ruptures_penalty: float = Field(default=10.0, gt=0, description="Ruptures penalty parameter")
    bcp_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description=(
            "Torch device for the Bayesian detector (cpu, cuda, mps). CPU is the fastest "
            "choice for series of a few thousand points; GPUs only pay off for much longer ones."
        ),
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    @field_validator("min_time_elapsed")
    @classmethod
    def validate_min_time_elapsed(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Minimum time elapsed must be positive")
        return v

    model_config = ConfigDict(validate_assignment=True)


class JitterAnalysisConfig(BaseModel):
    """
    Configuration for jitter analysis methods.

    Attributes
    ----------
    method : Literal['jitter_dispersion', 'ks_test']
        Method to use for jitter analysis.
    threshold : float
        Threshold for significance testing.
    moving_average_order : int
        Order of moving average filter (must be even).
    moving_iqr_order : int
        Order of moving IQR filter.
    significance_level : float
        Statistical significance level for tests.
    """

    method: Literal["jitter_dispersion", "ks_test"] = Field(
        default="jitter_dispersion", description="Jitter analysis method"
    )
    threshold: float = Field(default=0.25, gt=0, description="Threshold for significance testing")
    moving_average_order: int = Field(
        default=6, gt=0, description="Order of moving average filter (must be even)"
    )
    moving_iqr_order: int = Field(default=4, gt=0, description="Order of moving IQR filter")
    significance_level: float = Field(
        default=0.05, gt=0, lt=1, description="Statistical significance level"
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v

    @field_validator("moving_average_order")
    @classmethod
    def validate_moving_average_order(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Moving average order must be positive")
        if v % 2 != 0:
            raise ValueError("Moving average order must be even")
        return v

    @field_validator("moving_iqr_order")
    @classmethod
    def validate_moving_iqr_order(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Moving IQR order must be positive")
        return v

    @field_validator("significance_level")
    @classmethod
    def validate_significance_level(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("Significance level must be between 0 and 1")
        return v

    model_config = ConfigDict(validate_assignment=True)


class LatencyJumpConfig(BaseModel):
    """
    Configuration for latency jump detection.

    Attributes
    ----------
    threshold : float
        Threshold for latency jump detection.
    """

    threshold: float = Field(default=0.5, gt=0, description="Threshold for latency jump detection")

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v

    model_config = ConfigDict(validate_assignment=True)


class DataProcessingConfig(BaseModel):
    """
    Configuration for data processing.

    Attributes
    ----------
    minimum_interval_minutes : int
        Interval in minutes for computing minimum RTT values.
    min_samples_per_interval : int
        Minimum number of samples required per interval.
    outlier_detection : bool
        Whether to perform outlier detection.
    outlier_threshold : float
        Z-score threshold for outlier detection.
    """

    minimum_interval_minutes: int = Field(
        default=15, gt=0, description="Interval for minimum RTT computation"
    )
    min_samples_per_interval: int = Field(
        default=5, gt=0, description="Minimum samples required per interval"
    )
    outlier_detection: bool = Field(
        default=True, description="Whether to perform outlier detection"
    )
    outlier_threshold: float = Field(
        default=3.0, gt=0, description="Z-score threshold for outlier detection"
    )

    @field_validator("minimum_interval_minutes")
    @classmethod
    def validate_minimum_interval_minutes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Minimum interval minutes must be positive")
        return v

    @field_validator("min_samples_per_interval")
    @classmethod
    def validate_min_samples_per_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Minimum samples per interval must be positive")
        return v

    @field_validator("outlier_threshold")
    @classmethod
    def validate_outlier_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Outlier threshold must be positive")
        return v

    model_config = ConfigDict(validate_assignment=True)


class JitterbugConfig(BaseSettings):
    """
    Main configuration class for Jitterbug.

    Attributes
    ----------
    change_point_detection : ChangePointDetectionConfig
        Configuration for change point detection.
    jitter_analysis : JitterAnalysisConfig
        Configuration for jitter analysis.
    latency_jump : LatencyJumpConfig
        Configuration for latency jump detection.
    data_processing : DataProcessingConfig
        Configuration for data processing.
    output_format : Literal['json', 'csv', 'parquet']
        Output format for results.
    verbose : bool
        Whether to enable verbose logging.
    """

    change_point_detection: ChangePointDetectionConfig = Field(
        default_factory=ChangePointDetectionConfig
    )
    jitter_analysis: JitterAnalysisConfig = Field(default_factory=JitterAnalysisConfig)
    latency_jump: LatencyJumpConfig = Field(default_factory=LatencyJumpConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)

    output_format: Literal["json", "csv", "parquet"] = Field(
        default="json", description="Output format for results"
    )
    verbose: bool = Field(default=False, description="Enable verbose logging")

    @classmethod
    def from_file(cls, config_path: Path) -> "JitterbugConfig":
        """
        Load configuration from file.

        Parameters
        ----------
        config_path : Path
            Path to configuration file (YAML or JSON).

        Returns
        -------
        JitterbugConfig
            Loaded configuration.
        """
        import json

        import yaml

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open() as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        return cls(**data)

    def to_file(self, config_path: Path) -> None:
        """
        Save configuration to file.

        Parameters
        ----------
        config_path : Path
            Path to save configuration file.
        """
        import json

        import yaml

        data = self.model_dump()

        with config_path.open("w") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False)
            elif config_path.suffix.lower() == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    model_config = SettingsConfigDict(
        validate_assignment=True, env_prefix="JITTERBUG_", case_sensitive=False
    )

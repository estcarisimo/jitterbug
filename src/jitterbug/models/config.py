"""
Configuration models using Pydantic for validation and serialization.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


class ClusteringConfig(BaseModel):
    """
    Configuration for the non-sequential (clustering) analysis mode.

    Each minimum-RTT interval becomes one point with two features, its minimum RTT and
    the interquartile range of the jitter samples inside it. The points are clustered
    without regard to time; the cluster with the lowest median minimum RTT is the
    baseline, and every other cluster is congested when its median minimum RTT exceeds
    the baseline by more than the latency threshold and the jitter samples of the two
    clusters differ: a Kolmogorov-Smirnov test is significant at
    ``jitter_analysis.significance_level`` and its statistic is at least
    ``min_ks_statistic``.

    Attributes
    ----------
    algorithm : Literal['gmm', 'kmeans', 'kmeans_silhouette']
        ``gmm``: Gaussian mixture with the number of components chosen by BIC among
        1..``max_clusters`` (one component means no congestion signal). ``kmeans``:
        k-means with ``n_clusters`` clusters. ``kmeans_silhouette``: k-means with the
        number of clusters in 2..``max_clusters`` that maximizes the silhouette score.
    n_clusters : int
        Number of clusters for ``kmeans``.
    max_clusters : int
        Largest number of clusters tried by ``gmm`` and ``kmeans_silhouette``.
    min_period_intervals : int
        Temporal smoothing, in intervals: gaps of at most this many non-congested
        intervals inside congestion are filled, then congested runs of at most this
        many intervals are dropped. ``0`` keeps the raw per-interval labels.
    random_state : int
        Seed for the clustering algorithms, so results are reproducible.
    latency_threshold : float | None
        Minimum excess of a cluster's median minimum RTT over the baseline's (ms).
        ``None`` uses ``latency_jump.threshold``, the threshold of the sequential mode.
    min_ks_statistic : float
        Smallest Kolmogorov-Smirnov statistic (the largest gap between the two empirical
        jitter distributions, 0-1) that counts as a jitter change. Clusters pool
        thousands of samples, so the p-value alone is significant for negligible
        differences; this bounds the effect size. ``0`` relies on the p-value only.
    """

    algorithm: Literal["gmm", "kmeans", "kmeans_silhouette"] = Field(
        default="gmm", description="Clustering algorithm"
    )
    n_clusters: int = Field(default=2, ge=2, le=20, description="Clusters for kmeans")
    max_clusters: int = Field(
        default=6, ge=2, le=20, description="Largest number of clusters tried by model selection"
    )
    min_period_intervals: int = Field(
        default=2, ge=0, description="Temporal smoothing window, in minimum-RTT intervals"
    )
    random_state: int = Field(default=0, description="Random seed for clustering")
    latency_threshold: float | None = Field(
        default=None,
        gt=0,
        description="Latency jump threshold for this mode (ms); None uses latency_jump.threshold",
    )
    min_ks_statistic: float = Field(
        default=0.1, ge=0, le=1, description="Smallest KS statistic that counts as a jitter change"
    )

    model_config = ConfigDict(validate_assignment=True)


class StreamingConfig(BaseModel):
    """
    Settings of the online (streaming) mode, ``jitterbug.streaming.OnlineJitterbug``.

    The online mode reuses the sequential decision rule: minimum-RTT bins of
    ``data_processing.minimum_interval_minutes``, the latency jump threshold of
    ``latency_jump.threshold``, the significance level of
    ``jitter_analysis.significance_level`` and the device of
    ``change_point_detection.bcp_device``. The fields below are specific to the
    Bayesian online change point detector and to the two-stage verdicts.

    Attributes
    ----------
    backend : Literal['bocpd', 'window']
        ``bocpd`` (default): incremental Bayesian online change point detection, one bin
        at a time. ``window``: rerun the offline sequential pipeline on a trailing window
        of ``window_hours`` every ``rerun_every_bins`` closed bins and emit a verdict once
        it has been identical in ``stable_runs`` consecutive reruns. The baseline the
        incremental detector is compared with; heavier per bin, no new algorithm.
    window_hours : float
        Length of the trailing window of the ``window`` back end, in hours.
    rerun_every_bins : int
        Closed bins between two reruns of the ``window`` back end.
    stable_runs : int
        Consecutive reruns in which a change point or a verdict must be identical before
        the ``window`` back end emits it.
    hazard_lambda : float
        Expected run length in bins (constant hazard ``1 / hazard_lambda``).
    decision : Literal['map', 'lag', 'window']
        Change point rule. ``map`` (default) fires when the most probable run length
        drops; ``lag`` thresholds the posterior that a segment started exactly ``lag`` bins
        ago; ``window`` thresholds the posterior that it started within the last ``lag``
        bins and places the change at the most probable run length.
    lag : int
        Decision delay in bins for the ``lag`` and ``window`` rules.
    threshold : float
        Posterior threshold for the ``lag`` and ``window`` rules.
    max_run_length : int or None
        Run lengths kept by the detector (bounds memory and time per bin); ``None`` keeps
        the exact posterior.
    min_time_elapsed : int
        Minimum seconds between change points. The MAP rule refines a boundary a few
        bins after first reporting it; one hour absorbs those refinements.
    prior_alpha, prior_beta, prior_kappa : float
        Normal-Gamma prior of the Student-t predictive (see ``online_likelihoods.StudentT``).
    prior_mu : float or None
        Prior mean; ``None`` uses the first minimum RTT seen.
    min_ks_statistic : float
        Smallest Kolmogorov-Smirnov statistic that counts as a jitter change (effect-size
        guard, 0-1); ``0`` relies on the p-value only.
    min_period_samples : int
        Jitter samples the open period must hold before a provisional verdict. About
        three 15-minute bins on the paper dataset; 30 gives noisy p-values. With
        ``jitter_analysis.method: jitter_dispersion`` the verdict also waits for enough
        causal dispersion values (``JitterAnalyzer.causal_dispersion_delay`` plus
        ``moving_average_order`` bins).
    """

    backend: Literal["bocpd", "window"] = Field(
        default="bocpd",
        description="bocpd: incremental Bayesian detector; window: offline pipeline on a window",
    )
    window_hours: float = Field(default=72.0, gt=0, description="Trailing window (window backend)")
    rerun_every_bins: int = Field(
        default=1, ge=1, description="Closed bins between reruns (window backend)"
    )
    stable_runs: int = Field(
        default=2, ge=1, description="Identical consecutive reruns before emitting (window backend)"
    )
    hazard_lambda: float = Field(
        default=50.0, ge=1, description="Expected run length in bins (constant hazard)"
    )
    decision: Literal["map", "lag", "window"] = Field(
        default="map", description="Change point rule: MAP run-length drop, fixed lag, or window"
    )
    lag: int = Field(default=4, ge=1, description="Decision delay in bins (lag and window rules)")
    threshold: float = Field(
        default=0.5, gt=0, le=1, description="Posterior threshold (lag and window rules)"
    )
    max_run_length: int | None = Field(
        default=1000, ge=1, description="Run lengths kept; None keeps the exact posterior"
    )
    min_time_elapsed: int = Field(
        default=3600, gt=0, description="Minimum seconds between change points"
    )
    prior_alpha: float = Field(default=0.1, gt=0, description="Gamma prior shape on precision")
    prior_beta: float = Field(default=0.1, gt=0, description="Gamma prior rate on precision")
    prior_kappa: float = Field(default=1.0, gt=0, description="Normal prior precision on mean")
    prior_mu: float | None = Field(
        default=None, description="Prior mean; None uses the first minimum RTT seen"
    )
    min_ks_statistic: float = Field(
        default=0.0, ge=0, le=1, description="Smallest KS statistic that counts as a jitter change"
    )
    min_period_samples: int = Field(
        default=100, ge=2, description="Jitter samples before a provisional verdict"
    )

    @model_validator(mode="after")
    def _lag_fits_in_run_length(self) -> "StreamingConfig":
        """``lag`` is only used by the fixed-delay rules, and must be a kept run length."""
        if (
            self.decision != "map"
            and self.max_run_length is not None
            and self.lag > self.max_run_length
        ):
            raise ValueError(
                f"lag ({self.lag}) must not exceed max_run_length ({self.max_run_length}): "
                "the detector cannot report a run length it does not keep"
            )
        return self

    model_config = ConfigDict(validate_assignment=True)


class JitterbugConfig(BaseSettings):
    """
    Main configuration class for Jitterbug.

    Attributes
    ----------
    analysis_mode : Literal['sequential', 'clustering']
        ``sequential`` (default) is the PAM 2022 method: change points split the series
        and each period is compared with the previous one. ``clustering`` groups
        minimum-RTT intervals by (latency, jitter) regardless of when they occur and
        compares each cluster with the baseline one; see ``ClusteringConfig``.
    change_point_detection : ChangePointDetectionConfig
        Configuration for change point detection.
    jitter_analysis : JitterAnalysisConfig
        Configuration for jitter analysis.
    latency_jump : LatencyJumpConfig
        Configuration for latency jump detection.
    data_processing : DataProcessingConfig
        Configuration for data processing.
    clustering : ClusteringConfig
        Configuration for the clustering mode.
    streaming : StreamingConfig
        Configuration for the online (streaming) mode.
    output_format : Literal['json', 'csv', 'parquet']
        Output format for results.
    verbose : bool
        Whether to enable verbose logging.
    """

    analysis_mode: Literal["sequential", "clustering"] = Field(
        default="sequential", description="Sequential (change points) or clustering mode"
    )
    change_point_detection: ChangePointDetectionConfig = Field(
        default_factory=ChangePointDetectionConfig
    )
    jitter_analysis: JitterAnalysisConfig = Field(default_factory=JitterAnalysisConfig)
    latency_jump: LatencyJumpConfig = Field(default_factory=LatencyJumpConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)

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

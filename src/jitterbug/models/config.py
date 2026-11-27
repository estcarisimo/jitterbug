"""
Configuration models using Pydantic for validation and serialization.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
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
    revise_change_points : bool
        ``bocpd`` back end with the ``map`` rule only. When the MAP rule moves the last
        change point by less than ``min_time_elapsed`` (it only moves later), emit a
        ``revision`` event instead of dropping the refinement: the open period starts at
        the revised epoch, its provisional verdict is judged again, and its final verdict
        and ``StateChangeTracker`` use it. The previous period's final verdict, already
        emitted, keeps its ``end_epoch``. Off by default: on the paper dataset it puts 13
        instead of 8 of the 30 reference boundaries in the right bin but one revision
        moves a boundary away from the reference (19 instead of 20 within 30 min), and
        revisions arrive hours after the first report (see ``docs/ONLINE_MODE.md``).
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
    state_change_bins : int
        Closed minimum-RTT bins of a new period the early state-change rule looks at
        (``jitterbug.streaming.StateChangeTracker``): a change is *suspected* when every one
        of them is above (or, from ``congested``, none is above) the previous period's mean
        by more than ``latency_jump.threshold``. More bins, fewer retractions, later
        signal.
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
    revise_change_points: bool = Field(
        default=False,
        description="Report MAP refinements within min_time_elapsed as revisions (bocpd, map)",
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
    state_change_bins: int = Field(
        default=1, ge=1, description="Closed bins of a new period the early state-change rule uses"
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


RTT_COLUMN_SYNONYMS = ("values", "rtt_value", "rtt", "latency")
"""RTT column names searched, in this order, when ``InputConfig.rtt_column`` is ``None``."""


class InputConfig(BaseModel):
    """
    How raw input rows map to RTT samples: column names, units and row policies.

    Every file reader and ``DataLoader.load_from_dataframe`` hand their rows to one
    normalizer (``jitterbug.io.normalize``) that applies these settings. The defaults
    are the historical contract: an ``epoch`` column in Unix seconds, an RTT column in
    milliseconds named ``values``, ``rtt_value``, ``rtt`` or ``latency``, duplicate
    timestamps kept, out-of-range RTTs dropped and unsorted rows sorted. Jitterbug never
    imputes or resamples.

    Attributes
    ----------
    time_column : str
        Column holding the sample time.
    rtt_column : str | None
        Column holding the RTT. ``None`` takes the first of ``values``, ``rtt_value``,
        ``rtt``, ``latency`` that is present.
    source_column, destination_column : str
        Optional columns kept as per-sample metadata; absent columns give ``None``.
    time_unit : Literal['s', 'ms', 'us', 'ns', 'iso', 'auto']
        Unit of numeric times (Unix epoch in seconds, milliseconds, microseconds or
        nanoseconds), or ``iso`` for datetime strings (naive ones are UTC). ``auto``
        reads datetime strings as ``iso`` and numbers by the magnitude of their median:
        below 1e11 seconds, below 1e14 milliseconds, below 1e17 microseconds, otherwise
        nanoseconds. The inferred unit is logged and recorded in the dataset metadata.
    rtt_unit : Literal['ms', 'us', 's', 'ns']
        Unit of the RTT column; values are converted to milliseconds.
    delimiter : str | None
        Field separator of CSV-like files. ``None`` detects it from the header line
        (comma, tab or semicolon); TSV files always default to tab.
    duplicates : Literal['keep', 'first', 'min', 'error']
        Rows sharing a timestamp: keep them all, keep the first in file order, keep the
        one with the lowest RTT, or raise.
    out_of_range : Literal['drop', 'error']
        RTTs that are not positive or exceed ``MAX_RTT_MS``: drop them (counted in the
        metadata) or raise. Missing values are always dropped.
    unsorted : Literal['sort', 'error']
        Rows not in time order: sort them (stable) or raise.
    group_by : list[str]
        Columns whose values identify a series; each distinct combination is loaded and
        analyzed on its own, with the row policies applied per series. ``source`` and
        ``destination`` name the mapped ``source_column`` and ``destination_column``; any
        other name is a raw input column (a probe id, a device). Empty (the default)
        reads the input as one series.
    series : dict[str, str]
        Keep only the rows whose column equals the value (``{"destination":
        "8.8.8.8"}``), compared as text (numbers in a file or in Python become text:
        ``1234`` means ``"1234"``); names resolve like ``group_by``. Applied before
        grouping and before the row policies. A filter that matches no row is an error
        naming the values that exist.
    """

    time_column: str = Field(default="epoch", min_length=1, description="Time column")
    rtt_column: str | None = Field(
        default=None,
        min_length=1,
        description="RTT column; None searches values, rtt_value, rtt, latency",
    )
    source_column: str = Field(default="source", min_length=1, description="Source column")
    destination_column: str = Field(
        default="destination", min_length=1, description="Destination column"
    )
    time_unit: Literal["s", "ms", "us", "ns", "iso", "auto"] = Field(
        default="s", description="Unit of the time column (s, ms, us, ns, iso, auto)"
    )
    rtt_unit: Literal["ms", "us", "s", "ns"] = Field(
        default="ms", description="Unit of the RTT column, converted to milliseconds"
    )
    delimiter: str | None = Field(
        default=None,
        min_length=1,
        max_length=1,
        description="CSV field separator; None detects comma, tab or semicolon",
    )
    duplicates: Literal["keep", "first", "min", "error"] = Field(
        default="keep", description="Rows sharing a timestamp: keep, first, min or error"
    )
    out_of_range: Literal["drop", "error"] = Field(
        default="drop", description="Non-positive or too large RTTs: drop or error"
    )
    unsorted: Literal["sort", "error"] = Field(
        default="sort", description="Rows out of time order: sort or error"
    )
    group_by: list[str] = Field(
        default_factory=list,
        description="Columns identifying a series (source, destination or raw column names)",
    )
    series: dict[str, str] = Field(
        default_factory=dict, description="Keep only rows whose column equals the value"
    )

    @field_validator("group_by")
    @classmethod
    def _validate_group_by(cls, v: list[str]) -> list[str]:
        names = [name.strip() for name in v]
        if any(not name for name in names):
            raise ValueError("group_by names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"group_by names must be distinct: {names}")
        return names

    @field_validator("series", mode="before")
    @classmethod
    def _series_values_as_text(cls, v: Any) -> Any:
        """Filter values are compared as text, so ``probe: 1234`` in a YAML file means
        ``"1234"`` as on the command line; integral floats lose their ``.0``."""
        if not isinstance(v, dict):
            return v

        def text(value: Any) -> Any:
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            if isinstance(value, int | float):
                return str(value)
            return value

        return {name: text(value) for name, value in v.items()}

    @field_validator("series")
    @classmethod
    def _validate_series(cls, v: dict[str, str]) -> dict[str, str]:
        cleaned = {name.strip(): value for name, value in v.items()}
        if any(not name for name in cleaned):
            raise ValueError("series filter names must not be empty")
        return cleaned

    @model_serializer(mode="wrap")
    def _omit_empty_series_settings(self, handler: SerializerFunctionWrapHandler) -> Any:
        """Leave ``group_by`` and ``series`` out of dumps while they are empty, so a
        single-series run writes the same configuration (and result files) as before
        they existed. Loading fills them back with their defaults."""
        data = handler(self)
        if isinstance(data, dict):
            for name in ("group_by", "series"):
                if not data.get(name, True):
                    del data[name]
        return data

    model_config = ConfigDict(validate_assignment=True)


SOURCE_KINDS = ("clickhouse", "influxdb2", "influxdb3", "duckdb")
"""Database back ends a ``SourceConfig`` can name."""

SECRET_ENV_DEFAULTS = {
    "clickhouse": "CLICKHOUSE_PASSWORD",
    "influxdb2": "INFLUXDB_V2_TOKEN",
    "influxdb3": "INFLUX_TOKEN",
}
"""Environment variable each back end reads its password or token from by default:
the names ``clickhouse-client``, ``influxdb-client`` and ``influxdb3-python`` use."""

SECRET_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "token",
        "access_token",
        "api_key",
        "apikey",
        "api_token",
        "secret",
        "auth",
        "authorization",
        "credentials",
    }
)
"""Setting names that would hold a secret; they are rejected in a source profile or URI."""

_SOURCE_FIELDS: dict[str, frozenset[str]] = {
    "clickhouse": frozenset({"host", "port", "database", "user", "secure", "password_env"}),
    "influxdb2": frozenset({"host", "port", "org", "secure", "token_env"}),
    "influxdb3": frozenset({"host", "port", "database", "secure", "token_env", "language"}),
    "duckdb": frozenset({"path", "read_only"}),
}
"""Connection settings that apply to each back end (``query`` and ``query_file`` apply to
all of them)."""

_ENV_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"


def secret_rejection(key: str, kind: str | None) -> str:
    """
    The error for a secret found under ``key``: where the secret should come from instead.

    Parameters
    ----------
    key : str
        The offending setting or URI part (``password``, ``token``, ``userinfo``...).
    kind : str | None
        The back end, to name its environment variable; ``None`` lists them all.

    Returns
    -------
    str
        Message that never contains the secret itself.
    """
    if kind in SECRET_ENV_DEFAULTS:
        variable = SECRET_ENV_DEFAULTS[kind]
        option = "password_env" if kind == "clickhouse" else "token_env"
        where = f"the environment variable {variable} (or name another one with {option})"
    else:
        names = ", ".join(f"{v} ({k})" for k, v in SECRET_ENV_DEFAULTS.items())
        where = f"an environment variable: {names}"
    return (
        f"Credentials are not accepted in {key}: they would end up in configuration "
        f"files, shell history or logs. Put the secret in {where}"
    )


class SourceConfig(BaseModel):
    """
    A database query to read RTT samples from, instead of a file.

    The query result goes through the same column mapping, units, row policies,
    ``series`` filters and ``group_by`` as a file (``InputConfig``). Credentials are never
    part of this model: passwords and tokens are read from environment variables when
    the query runs, and a setting that looks like a secret is rejected.

    Attributes
    ----------
    kind : Literal['clickhouse', 'influxdb2', 'influxdb3', 'duckdb']
        Back end: ClickHouse (SQL, ``clickhouse`` extra), InfluxDB 2 (Flux, ``influx``
        extra), InfluxDB 3 (SQL or InfluxQL, ``influxdb3`` extra) or DuckDB (SQL over
        files, attached databases and object storage, ``duckdb`` extra).
    host : str | None
        Server host name or address (ClickHouse, InfluxDB); ``localhost`` when ``None``.
    port : int | None
        Server port; ``None`` uses the client's default for ClickHouse, 8086 for
        InfluxDB 2 and 8181 for InfluxDB 3 (443 with ``secure``).
    secure : bool
        Connect over HTTPS (ClickHouse, InfluxDB).
    database : str | None
        ClickHouse database, or InfluxDB 3 database (required there).
    user : str | None
        ClickHouse user; ``None`` reads ``CLICKHOUSE_USER``, then uses the client's
        default (``default``).
    org : str | None
        InfluxDB 2 organization.
    language : Literal['sql', 'influxql']
        Query language for InfluxDB 3.
    path : str | None
        DuckDB database file; ``None`` is an in-memory database.
    read_only : bool
        Open the DuckDB database file read-only (the default); never applies to an
        in-memory database.
    password_env : str | None
        Environment variable holding the ClickHouse password; ``None`` means
        ``CLICKHOUSE_PASSWORD``. An unset variable means no password.
    token_env : str | None
        Environment variable holding the InfluxDB token; ``None`` means
        ``INFLUXDB_V2_TOKEN`` (InfluxDB 2) or ``INFLUX_TOKEN`` (InfluxDB 3).
    query : str | None
        Query text. A ``--query`` or ``--query-file`` on the command line replaces it.
    query_file : Path | None
        File holding the query text, read when the query runs; exclusive with ``query``.
    """

    kind: Literal["clickhouse", "influxdb2", "influxdb3", "duckdb"] = Field(
        description="Database back end"
    )
    host: str | None = Field(default=None, description="Server host (no scheme, no user)")
    port: int | None = Field(default=None, ge=1, le=65535, description="Server port")
    secure: bool = Field(default=False, description="Connect over HTTPS")
    database: str | None = Field(default=None, description="ClickHouse or InfluxDB 3 database")
    user: str | None = Field(default=None, description="ClickHouse user")
    org: str | None = Field(default=None, description="InfluxDB 2 organization")
    language: Literal["sql", "influxql"] = Field(
        default="sql", description="InfluxDB 3 query language"
    )
    path: str | None = Field(default=None, description="DuckDB database file (None: in memory)")
    read_only: bool = Field(default=True, description="Open the DuckDB file read-only")
    password_env: str | None = Field(
        default=None, pattern=_ENV_NAME, description="Variable holding the ClickHouse password"
    )
    token_env: str | None = Field(
        default=None, pattern=_ENV_NAME, description="Variable holding the InfluxDB token"
    )
    query: str | None = Field(default=None, description="Query text")
    query_file: Path | None = Field(default=None, description="File holding the query text")

    @model_validator(mode="before")
    @classmethod
    def _reject_secrets(cls, data: Any) -> Any:
        """Refuse settings that would hold a password or token."""
        if isinstance(data, dict):
            kind = data.get("kind") if isinstance(data.get("kind"), str) else None
            for key in data:
                if isinstance(key, str) and key.lower() in SECRET_KEYS:
                    raise ValueError(secret_rejection(f"a source setting ('{key}')", kind))
        return data

    @field_validator("host")
    @classmethod
    def _validate_host(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if "@" in v:
            raise ValueError(secret_rejection("the host (user information before '@')", None))
        if not v.strip() or any(c in v for c in "/ \t\n") or "://" in v:
            raise ValueError(
                f"host must be a host name or address, without scheme, path or spaces: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _check_fields_for_kind(self) -> "SourceConfig":
        """Settings of another back end are an error, not silently ignored."""
        allowed = _SOURCE_FIELDS[self.kind] | {"kind", "query", "query_file"}
        defaults = type(self).model_fields
        stray = sorted(
            name
            for name in type(self).model_fields
            if name not in allowed and getattr(self, name) != defaults[name].default
        )
        if stray:
            raise ValueError(
                f"{', '.join(stray)} do(es) not apply to a {self.kind} source; it takes "
                f"{', '.join(sorted(_SOURCE_FIELDS[self.kind]))}, query and query_file"
            )
        if self.query is not None and self.query_file is not None:
            raise ValueError("give either query or query_file, not both")
        if self.kind == "influxdb3" and not self.database:
            raise ValueError("an influxdb3 source needs a database")
        return self

    def secret_env(self) -> str | None:
        """
        The environment variable this source reads its password or token from.

        Returns
        -------
        str | None
            ``password_env``/``token_env`` or the back end's default; ``None`` for DuckDB.
        """
        if self.kind == "clickhouse":
            return self.password_env or SECRET_ENV_DEFAULTS["clickhouse"]
        if self.kind in ("influxdb2", "influxdb3"):
            return self.token_env or SECRET_ENV_DEFAULTS[self.kind]
        return None

    # Validation errors never echo the input: it may be a rejected secret.
    model_config = ConfigDict(validate_assignment=True, extra="forbid", hide_input_in_errors=True)


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
    input : InputConfig
        Column mapping, units and row policies for loading RTT data.
    sources : dict[str, SourceConfig]
        Named database sources (profiles), selected with ``--source NAME``. Left out of
        dumps while empty, so configurations without sources serialize as before.
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
    input: InputConfig = Field(default_factory=InputConfig)
    sources: dict[str, SourceConfig] = Field(
        default_factory=dict, description="Named database sources (--source NAME)"
    )

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

    @model_serializer(mode="wrap")
    def _omit_empty_sources(self, handler: SerializerFunctionWrapHandler) -> Any:
        """Leave ``sources`` out of dumps while empty: result files and templates of runs
        without database sources stay as they were."""
        data = handler(self)
        if isinstance(data, dict) and not data.get("sources", True):
            del data["sources"]
        return data

    # Validation errors do not echo the input: a rejected source setting may be a secret.
    model_config = SettingsConfigDict(
        validate_assignment=True,
        env_prefix="JITTERBUG_",
        case_sensitive=False,
        hide_input_in_errors=True,
    )

"""
Main analyzer class for Jitterbug framework.
"""

import logging
from pathlib import Path

import pandas as pd

from .analysis import (
    ClusteringCongestionAnalyzer,
    CongestionInferenceAnalyzer,
    JitterAnalyzer,
    LatencyJumpAnalyzer,
)
from .detection import ChangePointDetector
from .io import DataLoader
from .io.compression import open_text
from .io.exporters import reject_compressed_parquet
from .io.series import SeriesKey, format_series_key, series_key_dict
from .models import (
    ChangePoint,
    CongestionInferenceResult,
    JitterbugConfig,
    MinimumRTTDataset,
    MultiSeriesResult,
    RTTDataset,
    SeriesResult,
    SourceConfig,
)

logger = logging.getLogger(__name__)


class JitterbugAnalyzer:
    """
    Main analyzer class that orchestrates the entire Jitterbug analysis pipeline.

    This class coordinates data loading, change point detection, jitter analysis,
    latency jump detection, and final congestion inference.

    Parameters
    ----------
    config : JitterbugConfig
        Configuration object containing all analysis parameters.

    Attributes
    ----------
    config : JitterbugConfig
        Configuration object.
    data_loader : DataLoader
        Data loading component.
    change_point_detector : ChangePointDetector
        Change point detection component.
    jitter_analyzer : JitterAnalyzer
        Jitter analysis component.
    latency_jump_analyzer : LatencyJumpAnalyzer
        Latency jump analysis component.
    congestion_inference_analyzer : CongestionInferenceAnalyzer
        Congestion inference component.
    """

    def __init__(self, config: JitterbugConfig):
        """
        Initialize the Jitterbug analyzer.

        Parameters
        ----------
        config : JitterbugConfig
            Configuration object containing all analysis parameters.
        """
        self.config = config
        self._setup_logging()
        self._initialize_components()

        # Store analysis results for access after analysis
        self.raw_data: RTTDataset | None = None
        self.min_rtt_data: MinimumRTTDataset | None = None
        self.change_points: list[ChangePoint] | None = None

    def _setup_logging(self) -> None:
        """Set up logging based on configuration."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # basicConfig is a no-op once a handler is installed (the CLI installs one before
        # it reads the configuration file), so ``verbose: true`` raises the package logger
        # explicitly. The non-verbose case leaves whatever the caller configured alone.
        if self.config.verbose:
            logging.getLogger("jitterbug").setLevel(logging.DEBUG)

    def _initialize_components(self) -> None:
        """Initialize all analysis components."""
        self.data_loader = DataLoader(self.config.input)
        self.change_point_detector = ChangePointDetector(self.config.change_point_detection)
        self.jitter_analyzer = JitterAnalyzer(self.config.jitter_analysis)
        self.latency_jump_analyzer = LatencyJumpAnalyzer(self.config.latency_jump)
        self.congestion_inference_analyzer = CongestionInferenceAnalyzer()
        self.clustering_analyzer = ClusteringCongestionAnalyzer(
            self.config.clustering,
            latency_threshold=(
                self.config.clustering.latency_threshold
                if self.config.clustering.latency_threshold is not None
                else self.config.latency_jump.threshold
            ),
            significance_level=self.config.jitter_analysis.significance_level,
            interval_minutes=self.config.data_processing.minimum_interval_minutes,
        )

    def analyze_from_file(
        self, file_path: str | Path, file_format: str | None = None
    ) -> CongestionInferenceResult:
        """
        Analyze RTT data from a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the RTT data file.
        file_format : Optional[str]
            Input format, one of ``jitterbug.io.readers.available_formats()``. If None,
            it is inferred from the extension, then from the content.

        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info(f"Loading RTT data from {file_path}")

        # Load RTT data
        rtt_data = self.data_loader.load_from_file(file_path, file_format)

        # Store raw data for visualization
        self.raw_data = rtt_data

        # Perform analysis
        return self.analyze(rtt_data)

    def analyze_from_source(self, source: SourceConfig) -> CongestionInferenceResult:
        """
        Analyze RTT data read from a database query.

        Parameters
        ----------
        source : SourceConfig
            Back end, connection settings and query; the result rows are mapped through
            ``config.input`` like a file's. Credentials come from environment variables.

        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info(f"Loading RTT data from a {source.kind} query")
        rtt_data = self.data_loader.load_from_source(source)
        self.raw_data = rtt_data
        return self.analyze(rtt_data)

    def analyze_series_from_source(self, source: SourceConfig) -> MultiSeriesResult:
        """
        Analyze each series of a database query result on its own; see
        :meth:`analyze_series_from_file`.

        Parameters
        ----------
        source : SourceConfig
            Back end, connection settings and query.

        Returns
        -------
        MultiSeriesResult
            One entry per series.
        """
        logger.info(f"Loading RTT series from a {source.kind} query")
        skipped: dict[SeriesKey, str] = {}
        datasets = self.data_loader.load_series_from_source(source, skipped)
        return self._analyze_each(datasets, skipped)

    def analyze_from_dataframe(self, df: pd.DataFrame) -> CongestionInferenceResult:
        """
        Analyze RTT data from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing RTT data, mapped through ``config.input`` (by default
            an ``epoch`` column in seconds and a ``values`` column in milliseconds).

        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info("Loading RTT data from DataFrame")

        # Convert DataFrame to RTTDataset
        rtt_data = self.data_loader.load_from_dataframe(df)

        # Store raw data for visualization
        self.raw_data = rtt_data

        # Perform analysis
        return self.analyze(rtt_data)

    def analyze_series_from_file(
        self, file_path: str | Path, file_format: str | None = None
    ) -> MultiSeriesResult:
        """
        Analyze each series of a file on its own (``config.input.group_by``).

        Rows are filtered by ``input.series``, split by ``input.group_by`` and loaded per
        series (``DataLoader.load_series``); each series is then analyzed as
        :meth:`analyze` would analyze it alone. A series that cannot be analyzed (all
        rows dropped, fewer than two minimum-RTT intervals, an analysis error) is
        reported as skipped with its reason; the others are unaffected. After the call,
        ``raw_data``, ``min_rtt_data`` and ``change_points`` belong to the last series
        analyzed.

        Parameters
        ----------
        file_path : str | Path
            Path to the RTT data file.
        file_format : str | None
            Input format; inferred when ``None``.

        Returns
        -------
        MultiSeriesResult
            One entry per series: series analyzed, in order of first appearance, then
            series whose rows were all dropped while loading.

        Raises
        ------
        ValueError
            If the file cannot be loaded, a key column is missing or a ``series`` filter
            matches no row.
        """
        logger.info(f"Loading RTT series from {file_path}")
        skipped: dict[SeriesKey, str] = {}
        datasets = self.data_loader.load_series(file_path, file_format, skipped)
        return self._analyze_each(datasets, skipped)

    def analyze_series_from_dataframe(self, df: pd.DataFrame) -> MultiSeriesResult:
        """
        Analyze each series of a DataFrame on its own; see
        :meth:`analyze_series_from_file`.

        Parameters
        ----------
        df : pd.DataFrame
            Raw rows, mapped through ``config.input``.

        Returns
        -------
        MultiSeriesResult
            One entry per series.
        """
        skipped: dict[SeriesKey, str] = {}
        datasets = self.data_loader.load_series_from_dataframe(df, skipped)
        return self._analyze_each(datasets, skipped)

    def _analyze_each(
        self, datasets: dict[SeriesKey, RTTDataset], skipped: dict[SeriesKey, str]
    ) -> MultiSeriesResult:
        """Analyze loaded series one by one; series that fail are skipped with a reason."""
        entries: list[SeriesResult] = []
        for key, dataset in datasets.items():
            label = format_series_key(key)
            logger.info(f"Analyzing series {label} ({len(dataset)} samples)")
            self.raw_data = dataset
            try:
                result = self.analyze(dataset)
            except ValueError as e:
                reason = f"analysis failed: {e}"
                logger.warning(f"Skipping series {label}: {reason}")
                entries.append(SeriesResult(key=series_key_dict(key), skipped_reason=reason))
                continue
            if "error" in result.metadata:
                reason = (
                    f"{result.metadata['min_intervals']} minimum-RTT interval(s) of "
                    f"{self.config.data_processing.minimum_interval_minutes} min, "
                    "need at least 2"
                )
                logger.warning(f"Skipping series {label}: {reason}")
                entries.append(SeriesResult(key=series_key_dict(key), skipped_reason=reason))
                continue
            entries.append(SeriesResult(key=series_key_dict(key), result=result))
        for key, reason in skipped.items():
            entries.append(SeriesResult(key=series_key_dict(key), skipped_reason=reason))
        analyzed = sum(entry.result is not None for entry in entries)
        return MultiSeriesResult(
            group_by=list(self.config.input.group_by),
            series=entries,
            metadata={
                "series_count": len(entries),
                "analyzed": analyzed,
                "skipped": len(entries) - analyzed,
                "series_filter": dict(self.config.input.series),
            },
        )

    def analyze(self, rtt_data: RTTDataset) -> CongestionInferenceResult:
        """
        Perform complete Jitterbug analysis on RTT data.

        Parameters
        ----------
        rtt_data : RTTDataset
            RTT measurement dataset.

        Returns
        -------
        CongestionInferenceResult
            Complete analysis results including congestion inferences.
        """
        logger.info(f"Starting Jitterbug analysis on {len(rtt_data)} RTT measurements")

        # Step 1: Compute minimum RTT intervals
        logger.info("Computing minimum RTT intervals")
        min_rtt_data = rtt_data.compute_minimum_intervals(
            self.config.data_processing.minimum_interval_minutes
        )

        # Store min_rtt_data for visualization
        self.min_rtt_data = min_rtt_data

        if len(min_rtt_data) < 2:
            logger.warning("Insufficient data for analysis (need at least 2 intervals)")
            return CongestionInferenceResult(
                inferences=[],
                metadata={
                    "error": "Insufficient data for analysis",
                    "total_measurements": len(rtt_data),
                    "min_intervals": len(min_rtt_data),
                    "change_points": 0,
                    "config": self.config.model_dump(),
                },
            )

        if self.config.analysis_mode == "clustering":
            return self._analyze_clustering(rtt_data, min_rtt_data)

        # Step 2: Detect change points
        logger.info("Detecting change points")
        change_points = self.change_point_detector.detect(min_rtt_data)

        # Store change points for visualization
        self.change_points = change_points

        if not change_points:
            logger.info("No change points detected")
            return CongestionInferenceResult(
                inferences=[],
                metadata={
                    "total_measurements": len(rtt_data),
                    "min_intervals": len(min_rtt_data),
                    "change_points": 0,
                    "config": self.config.model_dump(),
                },
            )

        logger.info(f"Detected {len(change_points)} change points")

        # Step 3: Analyze latency jumps
        logger.info("Analyzing latency jumps")
        latency_jumps = self.latency_jump_analyzer.analyze(min_rtt_data, change_points)

        # Step 4: Analyze jitter
        logger.info("Analyzing jitter")
        if self.config.jitter_analysis.method == "jitter_dispersion":
            jitter_results = self.jitter_analyzer.analyze_jitter_dispersion(
                min_rtt_data, change_points
            )
        else:  # ks_test
            jitter_results = self.jitter_analyzer.analyze_ks_test(rtt_data, change_points)

        # Step 5: Infer congestion
        logger.info("Inferring congestion periods")
        congestion_inferences = self.congestion_inference_analyzer.infer(
            latency_jumps, jitter_results
        )

        # Prepare results
        result = CongestionInferenceResult(
            inferences=congestion_inferences,
            metadata={
                "total_measurements": len(rtt_data),
                "min_intervals": len(min_rtt_data),
                "change_points": len(change_points),
                "latency_jumps": len(latency_jumps),
                "jitter_analyses": len(jitter_results),
                "congestion_periods": len([ci for ci in congestion_inferences if ci.is_congested]),
                "config": self.config.model_dump(),
            },
        )

        logger.info(
            f"Analysis complete: {len(congestion_inferences)} periods analyzed, "
            f"{len(result.get_congested_periods())} congestion periods found"
        )

        return result

    def _analyze_clustering(
        self, rtt_data: RTTDataset, min_rtt_data: MinimumRTTDataset
    ) -> CongestionInferenceResult:
        """
        Non-sequential mode: cluster minimum-RTT intervals instead of detecting change points.

        Parameters
        ----------
        rtt_data : RTTDataset
            Raw RTT measurements.
        min_rtt_data : MinimumRTTDataset
            Minimum-RTT intervals of ``rtt_data`` (for the metadata and the plots).

        Returns
        -------
        CongestionInferenceResult
            Periods of consecutive intervals with the same verdict. ``metadata`` carries
            ``analysis_mode`` and a ``clustering`` summary (clusters, their statistics
            and the model-selection scores).
        """
        logger.info(f"Clustering minimum-RTT intervals ({self.config.clustering.algorithm})")
        self.change_points = []
        outcome = self.clustering_analyzer.analyze(rtt_data)
        inferences = outcome.inferences
        result = CongestionInferenceResult(
            inferences=inferences,
            metadata={
                "analysis_mode": "clustering",
                "total_measurements": len(rtt_data),
                "min_intervals": len(min_rtt_data),
                "change_points": 0,
                "congestion_periods": len([ci for ci in inferences if ci.is_congested]),
                "clustering": outcome.metadata(self.config.clustering.algorithm),
                "config": self.config.model_dump(),
            },
        )
        logger.info(
            f"Analysis complete: {len(inferences)} periods, "
            f"{len(result.get_congested_periods())} congestion periods found"
        )
        return result

    def save_results(
        self, results: CongestionInferenceResult, output_path: str | Path, format: str | None = None
    ) -> None:
        """
        Save analysis results to file.

        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to save.
        output_path : Union[str, Path]
            Path to save results to. JSON and CSV are compressed when it ends in
            ``.zst``, ``.gz``, ``.bz2`` or ``.xz`` (``results.json.zst``).
        format : Optional[str]
            Output format ('json', 'csv', 'parquet'). If None, uses config default.

        Raises
        ------
        ValueError
            If the format is unknown, or Parquet is requested with a compressed path.
        ImportError
            If the path ends in ``.zst`` and no Zstandard codec is available.
        """
        output_path = Path(output_path)
        format = format or self.config.output_format

        if format == "json":
            import json

            with open_text(output_path, "w") as f:
                json.dump(results.model_dump(), f, indent=2, default=str)

        elif format == "csv":
            df = results.to_dataframe()
            with open_text(output_path, "w", newline="") as f:
                df.to_csv(f, index=False)

        elif format == "parquet":
            reject_compressed_parquet(output_path)
            df = results.to_dataframe()
            df.to_parquet(output_path)

        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Results saved to {output_path}")

    def save_series_results(
        self, results: MultiSeriesResult, output_path: str | Path, format: str | None = None
    ) -> None:
        """
        Save the results of a grouped analysis.

        JSON is one document, ``{"group_by": [...], "series": [{"key": {...}, "result":
        {...}} | {"key": {...}, "skipped_reason": "..."}, ...], "metadata": {...}}``, each
        ``result`` as :meth:`save_results` writes it. CSV and Parquet hold every period
        of every analyzed series, with the key columns first.

        Parameters
        ----------
        results : MultiSeriesResult
            Results to save.
        output_path : str | Path
            Destination; JSON and CSV are compressed for ``.zst``, ``.gz``, ``.bz2`` or
            ``.xz``.
        format : str | None
            ``json``, ``csv`` or ``parquet``; ``None`` uses ``config.output_format``.

        Raises
        ------
        ValueError
            If the format is unknown, or Parquet is requested with a compressed path.
        """
        output_path = Path(output_path)
        format = format or self.config.output_format
        if format == "json":
            import json

            with open_text(output_path, "w") as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
        elif format == "csv":
            with open_text(output_path, "w", newline="") as f:
                results.to_dataframe().to_csv(f, index=False)
        elif format == "parquet":
            reject_compressed_parquet(output_path)
            results.to_dataframe().to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        logger.info(f"Results of {len(results.series)} series saved to {output_path}")

    def get_summary_statistics(self, results: CongestionInferenceResult) -> dict:
        """
        Get summary statistics from analysis results.

        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results.

        Returns
        -------
        dict
            Summary statistics.
        """
        congested_periods = results.get_congested_periods()
        total_duration = results.get_total_congestion_duration()

        if results.inferences:
            total_analysis_time = max(inf.end_epoch for inf in results.inferences) - min(
                inf.start_epoch for inf in results.inferences
            )
            congestion_ratio = (
                total_duration / total_analysis_time if total_analysis_time > 0 else 0
            )
        else:
            total_analysis_time = 0
            congestion_ratio = 0

        return {
            "total_periods": len(results.inferences),
            "congested_periods": len(congested_periods),
            "total_duration_seconds": total_analysis_time,
            "congestion_duration_seconds": total_duration,
            "congestion_ratio": congestion_ratio,
            "average_confidence": (
                sum(inf.confidence for inf in congested_periods) / len(congested_periods)
                if congested_periods
                else 0
            ),
            "metadata": results.metadata,
        }

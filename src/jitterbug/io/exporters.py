"""
Result export utilities.

JSON and CSV outputs are Zstandard-compressed when the path ends in ``.zst``
(``results.json.zst``); see :mod:`jitterbug.io.compression`.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from ..models import CongestionInferenceResult
from .compression import is_zstd, open_text

logger = logging.getLogger(__name__)


def reject_zstd_parquet(output_path: Path) -> None:
    """Raise ValueError for ``*.zst`` Parquet output, which would compress twice."""
    if is_zstd(output_path):
        raise ValueError(
            f"Parquet output is already compressed; drop the .zst suffix: {output_path}"
        )


class ResultExporter:
    """
    Exporter for analysis results to various formats.
    """

    def __init__(self) -> None:
        """Initialize the result exporter."""

    def export_to_json(
        self, results: CongestionInferenceResult, output_path: str | Path, pretty: bool = True
    ) -> None:
        """
        Export results to JSON format.

        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save JSON file; compressed with Zstandard if it ends in ``.zst``.
        pretty : bool
            Whether to format JSON with indentation.
        """
        output_path = Path(output_path)

        # Convert to dictionary
        data = results.model_dump()

        # Custom serialization for datetime objects
        def json_serializer(obj: object) -> str:
            isoformat = getattr(obj, "isoformat", None)
            return isoformat() if callable(isoformat) else str(obj)

        with open_text(output_path, "w") as f:
            if pretty:
                json.dump(data, f, indent=2, default=json_serializer)
            else:
                json.dump(data, f, default=json_serializer)

        logger.info(f"Results exported to JSON: {output_path}")

    def export_to_csv(self, results: CongestionInferenceResult, output_path: str | Path) -> None:
        """
        Export results to CSV format.

        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save CSV file; compressed with Zstandard if it ends in ``.zst``.
        """
        output_path = Path(output_path)

        # Convert to DataFrame
        df = results.to_dataframe()

        # Add additional columns if needed
        if results.inferences:
            df["confidence"] = [inf.confidence for inf in results.inferences]
            df["has_latency_jump"] = [
                inf.latency_jump.has_jump if inf.latency_jump else False
                for inf in results.inferences
            ]
            df["has_jitter_change"] = [
                inf.jitter_analysis.has_significant_jitter if inf.jitter_analysis else False
                for inf in results.inferences
            ]

        with open_text(output_path, "w", newline="") as f:
            df.to_csv(f, index=False)

        logger.info(f"Results exported to CSV: {output_path}")

    def export_to_parquet(
        self, results: CongestionInferenceResult, output_path: str | Path
    ) -> None:
        """
        Export results to Parquet format.

        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save Parquet file.

        Raises
        ------
        ValueError
            If ``output_path`` ends in ``.zst``: Parquet compresses its own columns.
        """
        output_path = Path(output_path)
        reject_zstd_parquet(output_path)

        # Convert to DataFrame
        df = results.to_dataframe()

        # Add additional columns if needed
        if results.inferences:
            df["confidence"] = [inf.confidence for inf in results.inferences]
            df["has_latency_jump"] = [
                inf.latency_jump.has_jump if inf.latency_jump else False
                for inf in results.inferences
            ]
            df["has_jitter_change"] = [
                inf.jitter_analysis.has_significant_jitter if inf.jitter_analysis else False
                for inf in results.inferences
            ]

        df.to_parquet(output_path)

        logger.info(f"Results exported to Parquet: {output_path}")

    def export_summary(self, results: CongestionInferenceResult, output_path: str | Path) -> None:
        """
        Export summary statistics to JSON format.

        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save summary JSON file; compressed with Zstandard if it ends in
            ``.zst``.
        """
        output_path = Path(output_path)

        # Calculate summary statistics
        summary = {
            "total_periods": len(results.inferences),
            "congested_periods": len(results.get_congested_periods()),
            "congestion_ratio": len(results.get_congested_periods()) / len(results.inferences)
            if results.inferences
            else 0,
            "total_congestion_duration": results.get_total_congestion_duration(),
            "metadata": results.metadata,
        }

        if results.inferences:
            # Time range
            summary["time_range"] = {
                "start": min(inf.start_epoch for inf in results.inferences),
                "end": max(inf.end_epoch for inf in results.inferences),
            }

            # Confidence statistics
            congested_periods = results.get_congested_periods()
            if congested_periods:
                confidences = [inf.confidence for inf in congested_periods]
                summary["confidence_stats"] = {
                    "mean": float(pd.Series(confidences).mean()),
                    "median": float(pd.Series(confidences).median()),
                    "min": float(pd.Series(confidences).min()),
                    "max": float(pd.Series(confidences).max()),
                }

        with open_text(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary exported to JSON: {output_path}")

"""
Data loading: file and DataFrame entry points.

A file goes through a reader (``jitterbug.io.readers``, chosen by format) that returns a
raw DataFrame, then through the normalizer (``jitterbug.io.normalize``) that applies the
column mapping, units and row policies of ``InputConfig``.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models import RTTDataset
from ..models.config import RTT_COLUMN_SYNONYMS, InputConfig
from .normalize import datetimes_to_epoch, normalize
from .readers import get_reader, infer_format

logger = logging.getLogger(__name__)

EPOCH_COLUMN = "epoch"
"""Default time column (``InputConfig.time_column``)."""
RTT_COLUMNS = RTT_COLUMN_SYNONYMS
"""Accepted names for the RTT column when ``InputConfig.rtt_column`` is ``None``."""


class DataLoader:
    """
    Data loader for RTT data.

    Supports loading from files (CSV, TSV, generic JSONL, scamper JSON, RIPE Atlas
    results, ping/fping output, optionally compressed, and Parquet/Arrow; see
    ``jitterbug.io.readers``), pandas DataFrames and InfluxDB query results.

    Parameters
    ----------
    input_config : InputConfig | None
        Column mapping, units and row policies; ``None`` uses the defaults.
    """

    def __init__(self, input_config: InputConfig | None = None) -> None:
        """Initialize the data loader."""
        self.input_config = input_config if input_config is not None else InputConfig()

    def load_from_file(self, file_path: str | Path, file_format: str | None = None) -> RTTDataset:
        """
        Load RTT data from a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the data file. A trailing ``.zst``, ``.gz``, ``.bz2`` or ``.xz`` means
            the file is compressed (``rtts.csv.gz``) and does not count as the extension.
        file_format : Optional[str]
            A registered format (``jitterbug.io.readers.available_formats()``:
            ``parquet``, ``arrow``, ``ping``, ``fping``, ``csv``, ``tsv``, ``atlas``,
            ``json``, ``scamper``, ``jsonl``). If None, it is inferred from the
            extension, then from the magic bytes of a binary format, then from the first
            non-empty line.

        Returns
        -------
        RTTDataset
            Loaded RTT dataset; ``metadata`` records the source, the file path, the
            format and what normalization did.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the format is unknown or the content does not satisfy the input contract.
        ImportError
            If the file is compressed with a codec that is not installed, or is Parquet
            or Arrow and pyarrow is not installed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        if file_format is None:
            file_format = self._infer_format(file_path)
        reader = get_reader(file_format)

        logger.info(f"Loading data from {file_path} (format: {file_format})")
        raw = reader.read(file_path, self.input_config)
        config = (
            self.input_config.model_copy(update=raw.schema) if raw.schema else self.input_config
        )
        try:
            dataset = normalize(raw.frame, config)
        except ValueError as e:
            raise ValueError(f"Failed to load {reader.label} file {file_path}: {e}") from e
        dataset.metadata.update({"source": reader.name, "file_path": str(file_path)})
        dataset.metadata.update(raw.metadata)
        if raw.finish is not None:
            raw.finish(dataset)
        return dataset

    def load_from_dataframe(self, df: pd.DataFrame) -> RTTDataset:
        """
        Load RTT data from a pandas DataFrame.

        The frame is mapped through this loader's ``InputConfig`` (see
        ``docs/INPUT_FORMATS.md``); with the defaults: an ``epoch`` column with Unix
        seconds (UTC) and one RTT column named ``values``, ``rtt_value``, ``rtt`` or
        ``latency`` holding milliseconds; ``source`` and ``destination`` are optional.
        Validation happens once, on the whole frame (``jitterbug.io.normalize``):

        - a missing column or a value that cannot be read in its unit raises
          ``ValueError``;
        - rows with a missing time or RTT are dropped; non-positive RTTs and RTTs above
          ``MAX_RTT_MS`` are dropped (or rejected with ``out_of_range: error``); drops
          are counted in ``metadata["dropped_rows"]``;
        - rows are sorted by time (stable) if they are not already (or rejected with
          ``unsorted: error``); rows sharing a timestamp are handled by
          ``duplicates`` (kept by default).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing RTT data.

        Returns
        -------
        RTTDataset
            Loaded RTT dataset.

        Raises
        ------
        ValueError
            If a required column is missing, a value cannot be read, a policy set to
            ``error`` is violated, or no valid rows remain.
        """
        logger.info(f"Loading data from DataFrame with {len(df)} rows")
        return normalize(df, self.input_config)

    def _infer_format(self, file_path: Path) -> str:
        """
        Infer the file format: by extension (ignoring a compression suffix), otherwise
        from the first non-empty line; see ``jitterbug.io.readers.infer_format``.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        str
            A registered format name.

        Raises
        ------
        ValueError
            If the file cannot be read as text or no reader recognizes it.
        """
        return infer_format(file_path)

    def load_from_influxdb(
        self, url: str, token: str, org: str, bucket: str, query: str
    ) -> RTTDataset:
        """
        Load RTT data directly from InfluxDB.

        Parameters
        ----------
        url : str
            InfluxDB URL.
        token : str
            InfluxDB token.
        org : str
            InfluxDB organization.
        bucket : str
            InfluxDB bucket.
        query : str
            Flux query to execute.

        Returns
        -------
        RTTDataset
            Loaded RTT dataset.
        """
        try:
            from influxdb_client import InfluxDBClient
        except ImportError as e:
            raise ImportError("influxdb-client package is required for InfluxDB support") from e

        client = InfluxDBClient(url=url, token=token, org=org)
        query_api = client.query_api()

        try:
            # Execute query
            result = query_api.query_data_frame(query)

            # Convert to RTTDataset
            # A list means multiple tables were returned
            df = pd.concat(result, ignore_index=True) if isinstance(result, list) else result

            # Map InfluxDB columns to expected format
            if "_time" in df.columns:
                # Seconds since the epoch regardless of the datetime resolution
                # (pandas >= 3 parses to microseconds, so `.astype(int) / 1e9` was
                # off by a factor of 1000).
                df["epoch"] = datetimes_to_epoch(pd.to_datetime(df["_time"], utc=True))

            # Look for RTT value column
            rtt_column = None
            for col in ["_value", "rtt", "latency", "values"]:
                if col in df.columns:
                    rtt_column = col
                    break

            if rtt_column is None:
                raise ValueError("No RTT value column found in InfluxDB result")

            df["rtt_value"] = df[rtt_column]

            # The columns built above are fixed; units of the RTT and row policies
            # still come from the user's settings.
            schema = {"time_column": "epoch", "time_unit": "s", "rtt_column": "rtt_value"}
            return normalize(df, self.input_config.model_copy(update=schema))

        finally:
            client.close()

    def validate_data(self, dataset: RTTDataset) -> dict[str, Any]:
        """
        Validate RTT dataset and return quality metrics.

        Parameters
        ----------
        dataset : RTTDataset
            Dataset to validate.

        Returns
        -------
        Dict[str, any]
            Validation results and quality metrics.
        """
        if not dataset.measurements:
            return {"valid": False, "error": "No measurements found", "metrics": {}}

        epochs, rtt_values = dataset.to_arrays()

        # Check for time ordering (plain bools: the report is meant to be JSON-serializable)
        time_ordered = bool(np.all(epochs[:-1] <= epochs[1:]))

        # Check for duplicates
        unique_epochs = int(len(np.unique(epochs)))
        has_duplicates = unique_epochs < len(epochs)

        # Check for outliers (simple z-score method)
        z_scores = np.abs((rtt_values - np.mean(rtt_values)) / np.std(rtt_values))
        outliers = np.sum(z_scores > 3)

        # Time gaps analysis
        time_diffs = np.diff(epochs)
        avg_interval = np.mean(time_diffs)
        max_gap = np.max(time_diffs)

        # RTT statistics
        rtt_stats = {
            "min": float(np.min(rtt_values)),
            "max": float(np.max(rtt_values)),
            "mean": float(np.mean(rtt_values)),
            "median": float(np.median(rtt_values)),
            "std": float(np.std(rtt_values)),
            "outliers": int(outliers),
        }

        time_range = dataset.get_time_range()
        duration = (time_range[1] - time_range[0]).total_seconds()

        return {
            "valid": True,
            "metrics": {
                "total_measurements": len(dataset),
                "unique_timestamps": unique_epochs,
                "time_ordered": time_ordered,
                "has_duplicates": has_duplicates,
                "duration_seconds": float(duration),
                "average_interval_seconds": float(avg_interval),
                "max_gap_seconds": float(max_gap),
                "rtt_statistics": rtt_stats,
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat(),
                },
            },
        }

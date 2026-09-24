"""
Data loading utilities for various input formats.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models import MAX_RTT_MS, RTTDataset, RTTMeasurement

logger = logging.getLogger(__name__)

EPOCH_COLUMN = "epoch"
RTT_COLUMNS = ("values", "rtt_value", "rtt", "latency")
"""Accepted names for the RTT column, in order of preference. Values are milliseconds."""


class DataLoader:
    """
    Data loader for various RTT data formats.

    Supports loading from:
    - CSV files with epoch timestamps and RTT values
    - JSON files from scamper's warts outputs
    - InfluxDB query results
    - Pandas DataFrames
    """

    def __init__(self) -> None:
        """Initialize the data loader."""

    def load_from_file(self, file_path: str | Path, file_format: str | None = None) -> RTTDataset:
        """
        Load RTT data from a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the data file.
        file_format : Optional[str]
            Format of the file ('csv' or 'json'). If None, it is inferred from the
            extension, then from the first line.

        Returns
        -------
        RTTDataset
            Loaded RTT dataset.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Infer format from file extension if not provided
        if file_format is None:
            file_format = self._infer_format(file_path)

        logger.info(f"Loading data from {file_path} (format: {file_format})")

        if file_format == "csv":
            return self._load_from_csv(file_path)
        elif file_format == "json":
            return self._load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def load_from_dataframe(self, df: pd.DataFrame) -> RTTDataset:
        """
        Load RTT data from a pandas DataFrame.

        The input contract (see ``docs/INPUT_FORMATS.md``): an ``epoch`` column with Unix
        seconds (UTC) and one RTT column named ``values``, ``rtt_value``, ``rtt`` or
        ``latency`` holding milliseconds; ``source`` and ``destination`` are optional.
        Validation happens here, once, on the whole frame:

        - a missing column or a non-numeric value raises ``ValueError``;
        - rows with a missing epoch or RTT, a non-positive RTT, or an RTT above
          ``MAX_RTT_MS`` are dropped with a warning and counted in
          ``metadata["dropped_rows"]``;
        - rows are sorted by epoch (stable) if they are not already; duplicate epochs
          are kept (``validate_data`` reports them).

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
            If a required column is missing, a value is not numeric, or no valid rows
            remain.
        """
        logger.info(f"Loading data from DataFrame with {len(df)} rows")

        if EPOCH_COLUMN not in df.columns:
            raise ValueError(f"DataFrame must contain an '{EPOCH_COLUMN}' column")
        rtt_column = next((c for c in RTT_COLUMNS if c in df.columns), None)
        if rtt_column is None:
            raise ValueError(
                "DataFrame must contain an RTT column named one of "
                + ", ".join(f"'{c}'" for c in RTT_COLUMNS)
            )

        epochs = self._numeric_column(df, EPOCH_COLUMN)
        rtts = self._numeric_column(df, rtt_column)

        # Row filters, applied together so the counts refer to the original frame
        missing = epochs.isna() | rtts.isna()
        non_positive = ~missing & (rtts <= 0)
        too_large = ~missing & (rtts > MAX_RTT_MS)
        dropped = {
            "missing": int(missing.sum()),
            "non_positive": int(non_positive.sum()),
            "too_large": int(too_large.sum()),
        }
        keep = ~(missing | non_positive | too_large)
        for reason, count in dropped.items():
            if count:
                logger.warning(f"Dropping {count} row(s) with {reason.replace('_', '-')} RTT")
        if not keep.any():
            raise ValueError("No valid RTT rows: every row was missing, non-positive or too large")

        frame = df.loc[keep]
        epoch_values = epochs[keep].to_numpy(dtype=float)
        rtt_values = rtts[keep].to_numpy(dtype=float)
        needs_sort = bool(np.any(epoch_values[:-1] > epoch_values[1:]))
        if needs_sort:
            logger.warning("Rows are not in time order; sorting by epoch")
            order = np.argsort(epoch_values, kind="stable")
            frame = frame.iloc[order]
            epoch_values, rtt_values = epoch_values[order], rtt_values[order]

        sources = self._optional_column(frame, "source")
        destinations = self._optional_column(frame, "destination")
        measurements = [
            RTTMeasurement(
                timestamp=datetime.fromtimestamp(epoch, tz=timezone.utc),
                epoch=epoch,
                rtt_value=rtt,
                source=src,
                destination=dst,
            )
            for epoch, rtt, src, dst in zip(
                epoch_values.tolist(), rtt_values.tolist(), sources, destinations, strict=True
            )
        ]

        return RTTDataset(
            measurements=measurements,
            metadata={
                "source": "dataframe",
                "original_columns": list(df.columns),
                "rtt_column": rtt_column,
                "total_rows": len(df),
                "dropped_rows": dropped,
                "sorted_on_load": needs_sort,
            },
        )

    @staticmethod
    def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
        """Return ``column`` as floats; NaN and empty strings become NaN, anything else
        non-numeric is an error."""
        values = pd.to_numeric(df[column], errors="coerce")
        blank = df[column].map(lambda v: isinstance(v, str) and v.strip() == "")
        bad = values.isna() & df[column].notna() & ~blank
        if bad.any():
            example = df.loc[bad, column].iloc[0]
            raise ValueError(
                f"Column '{column}' has {int(bad.sum())} non-numeric value(s), e.g. {example!r}"
            )
        return values.astype(float)

    @staticmethod
    def _optional_column(df: pd.DataFrame, column: str) -> list[str | None]:
        """Per-row values of an optional string column, ``None`` where absent or missing."""
        if column not in df.columns:
            return [None] * len(df)
        return [None if pd.isna(v) else str(v) for v in df[column].tolist()]

    def _infer_format(self, file_path: Path) -> str:
        """
        Infer the file format: ``.csv``/``.json``/``.jsonl`` by extension, otherwise
        from the first line (a JSON object or a comma-separated header).

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        str
            ``"csv"`` or ``"json"``.

        Raises
        ------
        ValueError
            If the file cannot be read as text or its first line matches neither format.
        """
        extension = file_path.suffix.lower()

        if extension == ".csv":
            return "csv"
        elif extension in [".json", ".jsonl"]:
            return "json"
        # Unknown extension: look at the first line
        try:
            with file_path.open() as f:
                first_line = f.readline().strip()
        except (OSError, UnicodeError) as e:
            raise ValueError(f"Cannot infer format for file: {file_path}") from e
        if first_line.startswith("{"):
            return "json"
        if "," in first_line:
            return "csv"
        raise ValueError(
            f"Cannot infer format for file: {file_path} (expected a CSV header or a JSON "
            "object on the first line; pass file_format explicitly)"
        )

    def _load_from_csv(self, file_path: Path) -> RTTDataset:
        """
        Load RTT data from CSV file.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.

        Returns
        -------
        RTTDataset
            Loaded RTT dataset.
        """
        try:
            df = pd.read_csv(file_path)
            dataset = self.load_from_dataframe(df)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}") from e
        dataset.metadata.update({"source": "csv", "file_path": str(file_path)})
        return dataset

    def _load_from_json(self, file_path: Path) -> RTTDataset:
        """
        Load RTT data from JSON file (scamper warts format).

        Parameters
        ----------
        file_path : Path
            Path to the JSON file.

        Returns
        -------
        RTTDataset
            Loaded RTT dataset.
        """
        measurements = []
        out_of_range = 0

        try:
            with file_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Process ping measurements from scamper
                        if data.get("type") == "ping" and "responses" in data:
                            source = data.get("src", "")
                            destination = data.get("dst", "")

                            for response in data["responses"]:
                                if "rtt" in response:
                                    # Calculate timestamp from tx time
                                    tx_time = response.get("tx", {})
                                    if "sec" in tx_time and "usec" in tx_time:
                                        epoch = tx_time["sec"] + tx_time["usec"] / 1e6
                                        timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
                                        rtt_value = float(response["rtt"])
                                        # Same bounds as the DataFrame contract
                                        if not 0 < rtt_value <= MAX_RTT_MS:
                                            out_of_range += 1
                                            continue

                                        measurements.append(
                                            RTTMeasurement(
                                                timestamp=timestamp,
                                                epoch=epoch,
                                                rtt_value=rtt_value,
                                                source=source,
                                                destination=destination,
                                            )
                                        )

                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line[:120]!r}")
                        continue

        except Exception as e:
            raise ValueError(f"Failed to load JSON file {file_path}: {e}") from e

        if out_of_range:
            logger.warning(f"Dropping {out_of_range} response(s) with a non-positive or >10 s RTT")
        if not measurements:
            raise ValueError(f"No valid RTT measurements found in JSON file {file_path}")

        # Sort measurements by timestamp
        measurements.sort(key=lambda x: x.epoch)

        return RTTDataset(
            measurements=measurements,
            metadata={
                "source": "json",
                "file_path": str(file_path),
                "format": "scamper_warts",
                "total_measurements": len(measurements),
                "dropped_responses": out_of_range,
            },
        )

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
                # off by a factor of 1000). Rounded to microseconds like
                # `pd.Timestamp.timestamp()`: a float64 epoch cannot hold nanoseconds.
                times = pd.to_datetime(df["_time"], utc=True)
                nanos = (times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)
                df["epoch"] = (nanos / 1e9).round(6)

            # Look for RTT value column
            rtt_column = None
            for col in ["_value", "rtt", "latency", "values"]:
                if col in df.columns:
                    rtt_column = col
                    break

            if rtt_column is None:
                raise ValueError("No RTT value column found in InfluxDB result")

            df["rtt_value"] = df[rtt_column]

            return self.load_from_dataframe(df)

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

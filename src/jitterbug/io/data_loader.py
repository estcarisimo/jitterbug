"""
Data loading utilities for various input formats.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ..models import RTTMeasurement, RTTDataset


logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for various RTT data formats.
    
    Supports loading from:
    - CSV files with epoch timestamps and RTT values
    - JSON files from scamper's warts outputs
    - InfluxDB query results
    - Pandas DataFrames
    """
    
    def __init__(self):
        """Initialize the data loader."""
        pass
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None
    ) -> RTTDataset:
        """
        Load RTT data from a file.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the data file.
        file_format : Optional[str]
            Format of the file ('csv', 'json', 'influx'). If None, will be inferred.
            
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
        
        if file_format == 'csv':
            return self._load_from_csv(file_path)
        elif file_format == 'json':
            return self._load_from_json(file_path)
        elif file_format == 'influx':
            return self._load_from_influx_export(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def load_from_dataframe(self, df: pd.DataFrame) -> RTTDataset:
        """
        Load RTT data from a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing RTT data. Expected columns: 'epoch', 'values' or 'rtt_value'.
            
        Returns
        -------
        RTTDataset
            Loaded RTT dataset.
        """
        logger.info(f"Loading data from DataFrame with {len(df)} rows")
        
        # Validate required columns
        if 'epoch' not in df.columns:
            raise ValueError("DataFrame must contain 'epoch' column")
        
        # Handle different RTT value column names
        rtt_column = None
        for col in ['values', 'rtt_value', 'rtt', 'latency']:
            if col in df.columns:
                rtt_column = col
                break
        
        if rtt_column is None:
            raise ValueError("DataFrame must contain RTT values column ('values', 'rtt_value', 'rtt', or 'latency')")
        
        # Convert to RTT measurements
        measurements = []
        for _, row in df.iterrows():
            epoch = float(row['epoch'])
            rtt_value = float(row[rtt_column])
            timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
            
            # Extract additional columns if available
            source = row.get('source', None)
            destination = row.get('destination', None)
            
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=epoch,
                rtt_value=rtt_value,
                source=source,
                destination=destination
            ))
        
        return RTTDataset(
            measurements=measurements,
            metadata={
                'source': 'dataframe',
                'original_columns': list(df.columns),
                'total_rows': len(df)
            }
        )
    
    def _infer_format(self, file_path: Path) -> str:
        """
        Infer file format from file extension.
        
        Parameters
        ----------
        file_path : Path
            Path to the file.
            
        Returns
        -------
        str
            Inferred format.
        """
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return 'csv'
        elif extension == '.json':
            return 'json'
        elif extension in ['.influx', '.flux']:
            return 'influx'
        else:
            # Try to infer from content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{'):
                        return 'json'
                    elif ',' in first_line:
                        return 'csv'
                    else:
                        return 'influx'
            except Exception:
                raise ValueError(f"Cannot infer format for file: {file_path}")
    
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
            return self.load_from_dataframe(df)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}")
    
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
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Process ping measurements from scamper
                        if data.get('type') == 'ping' and 'responses' in data:
                            source = data.get('src', '')
                            destination = data.get('dst', '')
                            
                            for response in data['responses']:
                                if 'rtt' in response:
                                    # Calculate timestamp from tx time
                                    tx_time = response.get('tx', {})
                                    if 'sec' in tx_time and 'usec' in tx_time:
                                        epoch = tx_time['sec'] + tx_time['usec'] / 1e6
                                        timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
                                        rtt_value = response['rtt']
                                        
                                        measurements.append(RTTMeasurement(
                                            timestamp=timestamp,
                                            epoch=epoch,
                                            rtt_value=rtt_value,
                                            source=source,
                                            destination=destination
                                        ))
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line}")
                        continue
        
        except Exception as e:
            raise ValueError(f"Failed to load JSON file {file_path}: {e}")
        
        if not measurements:
            raise ValueError(f"No valid RTT measurements found in JSON file {file_path}")
        
        # Sort measurements by timestamp
        measurements.sort(key=lambda x: x.epoch)
        
        return RTTDataset(
            measurements=measurements,
            metadata={
                'source': 'json',
                'file_path': str(file_path),
                'format': 'scamper_warts',
                'total_measurements': len(measurements)
            }
        )
    
    def _load_from_influx_export(self, file_path: Path) -> RTTDataset:
        """
        Load RTT data from InfluxDB export file.
        
        Parameters
        ----------
        file_path : Path
            Path to the InfluxDB export file.
            
        Returns
        -------
        RTTDataset
            Loaded RTT dataset.
        """
        # This is a placeholder implementation
        # In a real implementation, you would parse the InfluxDB line protocol
        # or CSV export format
        raise NotImplementedError("InfluxDB export loading not yet implemented")
    
    def load_from_influxdb(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        query: str
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
        except ImportError:
            raise ImportError("influxdb-client package is required for InfluxDB support")
        
        client = InfluxDBClient(url=url, token=token, org=org)
        query_api = client.query_api()
        
        try:
            # Execute query
            result = query_api.query_data_frame(query)
            
            # Convert to RTTDataset
            if isinstance(result, list):
                # Multiple tables returned
                df = pd.concat(result, ignore_index=True)
            else:
                df = result
            
            # Map InfluxDB columns to expected format
            if '_time' in df.columns:
                df['epoch'] = pd.to_datetime(df['_time']).astype(int) / 1e9
            
            # Look for RTT value column
            rtt_column = None
            for col in ['_value', 'rtt', 'latency', 'values']:
                if col in df.columns:
                    rtt_column = col
                    break
            
            if rtt_column is None:
                raise ValueError("No RTT value column found in InfluxDB result")
            
            df['rtt_value'] = df[rtt_column]
            
            return self.load_from_dataframe(df)
        
        finally:
            client.close()
    
    def validate_data(self, dataset: RTTDataset) -> Dict[str, any]:
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
            return {
                'valid': False,
                'error': 'No measurements found',
                'metrics': {}
            }
        
        epochs, rtt_values = dataset.to_arrays()
        
        # Check for time ordering
        time_ordered = np.all(epochs[:-1] <= epochs[1:])
        
        # Check for duplicates
        unique_epochs = len(np.unique(epochs))
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
            'min': np.min(rtt_values),
            'max': np.max(rtt_values),
            'mean': np.mean(rtt_values),
            'median': np.median(rtt_values),
            'std': np.std(rtt_values),
            'outliers': int(outliers)
        }
        
        time_range = dataset.get_time_range()
        duration = (time_range[1] - time_range[0]).total_seconds()
        
        return {
            'valid': True,
            'metrics': {
                'total_measurements': len(dataset),
                'unique_timestamps': unique_epochs,
                'time_ordered': time_ordered,
                'has_duplicates': has_duplicates,
                'duration_seconds': duration,
                'average_interval_seconds': avg_interval,
                'max_gap_seconds': max_gap,
                'rtt_statistics': rtt_stats,
                'time_range': {
                    'start': time_range[0].isoformat(),
                    'end': time_range[1].isoformat()
                }
            }
        }
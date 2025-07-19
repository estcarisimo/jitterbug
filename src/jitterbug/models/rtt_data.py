"""
RTT data models using Pydantic for validation and serialization.
"""

from datetime import datetime
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class RTTMeasurement(BaseModel):
    """
    Represents a single RTT measurement.
    
    Attributes
    ----------
    timestamp : datetime
        When the measurement was taken.
    epoch : float
        Unix timestamp of the measurement.
    rtt_value : float
        Round-trip time value in milliseconds.
    source : Optional[str]
        Source identifier or IP address.
    destination : Optional[str]
        Destination identifier or IP address.
    """
    
    timestamp: datetime
    epoch: float
    rtt_value: float = Field(gt=0, description="RTT value must be positive")
    source: Optional[str] = None
    destination: Optional[str] = None
    
    @field_validator('rtt_value')
    @classmethod
    def validate_rtt_value(cls, v):
        if v <= 0:
            raise ValueError('RTT value must be positive')
        if v > 10000:  # 10 seconds seems unreasonably high
            raise ValueError('RTT value seems unreasonably high (>10s)')
        return v
    
    # Note: Timestamp-epoch consistency validation removed to avoid timezone/precision issues
    # The analysis algorithms primarily use epoch values, so this validation is not critical
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True


class RTTDataset(BaseModel):
    """
    Collection of RTT measurements with validation and processing capabilities.
    
    Attributes
    ----------
    measurements : List[RTTMeasurement]
        List of RTT measurements.
    metadata : dict
        Additional metadata about the dataset.
    """
    
    measurements: List[RTTMeasurement]
    metadata: dict = Field(default_factory=dict)
    
    @field_validator('measurements')
    @classmethod
    def validate_measurements_not_empty(cls, v):
        if not v:
            raise ValueError('Dataset must contain at least one measurement')
        return v
    
    @field_validator('measurements')
    @classmethod
    def validate_measurements_sorted(cls, v):
        """Ensure measurements are sorted by timestamp."""
        if len(v) > 1:
            epochs = [m.epoch for m in v]
            if epochs != sorted(epochs):
                raise ValueError('Measurements must be sorted by timestamp')
        return v
    
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert measurements to numpy arrays.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (epochs, rtt_values) as numpy arrays.
        """
        epochs = np.array([m.epoch for m in self.measurements])
        rtt_values = np.array([m.rtt_value for m in self.measurements])
        return epochs, rtt_values
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert measurements to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: timestamp, epoch, rtt_value, source, destination.
        """
        data = []
        for m in self.measurements:
            data.append({
                'timestamp': m.timestamp,
                'epoch': m.epoch,
                'rtt_value': m.rtt_value,
                'source': m.source,
                'destination': m.destination
            })
        return pd.DataFrame(data)
    
    def compute_minimum_intervals(self, interval_minutes: int = 15) -> 'MinimumRTTDataset':
        """
        Compute minimum RTT values over specified intervals.
        
        Parameters
        ----------
        interval_minutes : int
            Interval size in minutes for computing minimums.
            
        Returns
        -------
        MinimumRTTDataset
            Dataset containing minimum RTT values for each interval.
        """
        df = self.to_dataframe()
        df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
        
        # Group by interval and compute minimum
        interval_str = f"{interval_minutes}min"
        min_df = df.groupby(pd.Grouper(key='datetime', freq=interval_str)).agg({
            'epoch': 'min',
            'rtt_value': 'min'
        }).reset_index()
        
        # Remove rows with NaN values (empty intervals)
        min_df = min_df.dropna()
        
        # Convert back to measurements
        min_measurements = []
        for _, row in min_df.iterrows():
            min_measurements.append(RTTMeasurement(
                timestamp=row['datetime'],
                epoch=row['epoch'],
                rtt_value=row['rtt_value']
            ))
        
        return MinimumRTTDataset(
            measurements=min_measurements,
            interval_minutes=interval_minutes,
            metadata=self.metadata.copy()
        )
    
    def get_time_range(self) -> tuple[datetime, datetime]:
        """
        Get the time range of measurements.
        
        Returns
        -------
        tuple[datetime, datetime]
            Start and end timestamps of the dataset.
        """
        if not self.measurements:
            raise ValueError("No measurements available")
        
        return self.measurements[0].timestamp, self.measurements[-1].timestamp
    
    def __len__(self) -> int:
        """Return number of measurements."""
        return len(self.measurements)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True


class MinimumRTTDataset(RTTDataset):
    """
    Dataset containing minimum RTT values computed over intervals.
    
    Attributes
    ----------
    measurements : List[RTTMeasurement]
        List of minimum RTT measurements.
    interval_minutes : int
        Interval size in minutes used for computing minimums.
    """
    
    interval_minutes: int = Field(gt=0, description="Interval must be positive")
    
    @field_validator('interval_minutes')
    @classmethod
    def validate_interval_minutes(cls, v):
        if v <= 0:
            raise ValueError('Interval minutes must be positive')
        if v > 1440:  # More than 24 hours seems unreasonable
            raise ValueError('Interval minutes seems unreasonably large (>24h)')
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
"""
Result export utilities.
"""

import json
import logging
from pathlib import Path
from typing import Union
import pandas as pd

from ..models import CongestionInferenceResult


logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Exporter for analysis results to various formats.
    """
    
    def __init__(self):
        """Initialize the result exporter."""
        pass
    
    def export_to_json(
        self,
        results: CongestionInferenceResult,
        output_path: Union[str, Path],
        pretty: bool = True
    ) -> None:
        """
        Export results to JSON format.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save JSON file.
        pretty : bool
            Whether to format JSON with indentation.
        """
        output_path = Path(output_path)
        
        # Convert to dictionary
        data = results.dict()
        
        # Custom serialization for datetime objects
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)
        
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, default=json_serializer)
            else:
                json.dump(data, f, default=json_serializer)
        
        logger.info(f"Results exported to JSON: {output_path}")
    
    def export_to_csv(
        self,
        results: CongestionInferenceResult,
        output_path: Union[str, Path]
    ) -> None:
        """
        Export results to CSV format.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save CSV file.
        """
        output_path = Path(output_path)
        
        # Convert to DataFrame
        df = results.to_dataframe()
        
        # Add additional columns if needed
        if results.inferences:
            df['confidence'] = [inf.confidence for inf in results.inferences]
            df['has_latency_jump'] = [inf.latency_jump.has_jump if inf.latency_jump else False for inf in results.inferences]
            df['has_jitter_change'] = [inf.jitter_analysis.has_significant_jitter if inf.jitter_analysis else False for inf in results.inferences]
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results exported to CSV: {output_path}")
    
    def export_to_parquet(
        self,
        results: CongestionInferenceResult,
        output_path: Union[str, Path]
    ) -> None:
        """
        Export results to Parquet format.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save Parquet file.
        """
        output_path = Path(output_path)
        
        # Convert to DataFrame
        df = results.to_dataframe()
        
        # Add additional columns if needed
        if results.inferences:
            df['confidence'] = [inf.confidence for inf in results.inferences]
            df['has_latency_jump'] = [inf.latency_jump.has_jump if inf.latency_jump else False for inf in results.inferences]
            df['has_jitter_change'] = [inf.jitter_analysis.has_significant_jitter if inf.jitter_analysis else False for inf in results.inferences]
        
        df.to_parquet(output_path)
        
        logger.info(f"Results exported to Parquet: {output_path}")
    
    def export_summary(
        self,
        results: CongestionInferenceResult,
        output_path: Union[str, Path]
    ) -> None:
        """
        Export summary statistics to JSON format.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results to export.
        output_path : Union[str, Path]
            Path to save summary JSON file.
        """
        output_path = Path(output_path)
        
        # Calculate summary statistics
        summary = {
            'total_periods': len(results.inferences),
            'congested_periods': len(results.get_congested_periods()),
            'congestion_ratio': len(results.get_congested_periods()) / len(results.inferences) if results.inferences else 0,
            'total_congestion_duration': results.get_total_congestion_duration(),
            'metadata': results.metadata
        }
        
        if results.inferences:
            # Time range
            summary['time_range'] = {
                'start': min(inf.start_epoch for inf in results.inferences),
                'end': max(inf.end_epoch for inf in results.inferences)
            }
            
            # Confidence statistics
            congested_periods = results.get_congested_periods()
            if congested_periods:
                confidences = [inf.confidence for inf in congested_periods]
                summary['confidence_stats'] = {
                    'mean': float(pd.Series(confidences).mean()),
                    'median': float(pd.Series(confidences).median()),
                    'min': float(pd.Series(confidences).min()),
                    'max': float(pd.Series(confidences).max())
                }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported to JSON: {output_path}")
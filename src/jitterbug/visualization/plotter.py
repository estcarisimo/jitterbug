"""
Core plotting functionality for Jitterbug network analysis visualization.
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Create dummy objects for type annotations
    plt = None
    mdates = None

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING, Union
from pathlib import Path

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from ..models import (
    RTTDataset,
    MinimumRTTDataset,
    CongestionInferenceResult,
    ChangePoint,
    CongestionInference
)


class JitterbugPlotter:
    """
    Main plotting class for Jitterbug network analysis visualization.
    
    This class provides comprehensive plotting capabilities for RTT data,
    change points, congestion inferences, and analysis results.
    """
    
    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (15, 8)):
        """
        Initialize the plotter with style settings.
        
        Parameters
        ----------
        style : str
            Matplotlib style to use (default, seaborn, etc.)
        figsize : tuple
            Default figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting. Install with: pip install jitterbug[visualization]")
        
        self.style = style
        self.figsize = figsize
        self.colors = {
            'rtt': '#1f77b4',
            'min_rtt': '#ff7f0e', 
            'congestion': '#d62728',
            'change_point': '#2ca02c',
            'confidence': '#9467bd',
            'jitter': '#8c564b',
            'background': '#f0f0f0',
            'grid': '#bababa'
        }
        
        # Set style
        if style != "default":
            plt.style.use(style)
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.size'] = 10
        
    def plot_rtt_timeseries(
        self,
        datasets: Dict[str, RTTDataset],
        title: str = "RTT Time Series",
        save_path: Optional[Path] = None,
        show_points: bool = False
    ) -> "plt.Figure":
        """
        Plot RTT time series data.
        
        Parameters
        ----------
        datasets : dict
            Dictionary of dataset_name -> RTTDataset
        title : str
            Plot title
        save_path : Path, optional
            Path to save the plot
        show_points : bool
            Whether to show individual data points
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for name, dataset in datasets.items():
            # Convert to timestamps and values
            timestamps = [m.timestamp for m in dataset.measurements]
            values = [m.rtt_value for m in dataset.measurements]
            
            # Plot line
            ax.plot(timestamps, values, 
                   label=name, 
                   linewidth=2, 
                   alpha=0.8,
                   marker='o' if show_points else None,
                   markersize=3 if show_points else 0)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('RTT (ms)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_congestion_analysis(
        self,
        raw_data: RTTDataset,
        min_rtt_data: MinimumRTTDataset,
        results: CongestionInferenceResult,
        title: str = "Congestion Analysis",
        save_path: Optional[Path] = None
    ) -> "plt.Figure":
        """
        Plot comprehensive congestion analysis (similar to notebooks).
        
        Parameters
        ----------
        raw_data : RTTDataset
            Raw RTT measurements
        min_rtt_data : MinimumRTTDataset
            Minimum RTT data
        results : CongestionInferenceResult
            Analysis results
        title : str
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        
        # Configure grid for all subplots
        for ax in axes:
            ax.grid(True, linestyle='-', color=self.colors['grid'], alpha=0.5)
        
        # Plot 1: Raw RTT data
        raw_timestamps = [m.timestamp for m in raw_data.measurements]
        raw_values = [m.rtt_value for m in raw_data.measurements]
        
        axes[0].plot(raw_timestamps, raw_values,
                    label="RTT", 
                    color=self.colors['rtt'],
                    alpha=0.75, 
                    linewidth=2)
        
        # Plot 2: Minimum RTT data
        min_timestamps = [m.timestamp for m in min_rtt_data.measurements]
        min_values = [m.rtt_value for m in min_rtt_data.measurements]
        
        axes[1].plot(min_timestamps, min_values,
                    label="min(RTT)", 
                    color=self.colors['min_rtt'],
                    alpha=0.75, 
                    linewidth=2)
        
        # Plot 3: Congestion inferences
        congestion_times = []
        congestion_values = []
        
        for inf in results.inferences:
            # Add start point
            congestion_times.append(inf.start_timestamp)
            congestion_values.append(1.0 if inf.is_congested else 0.0)
            
            # Add end point
            congestion_times.append(inf.end_timestamp)
            congestion_values.append(1.0 if inf.is_congested else 0.0)
        
        axes[2].plot(congestion_times, congestion_values,
                    label="Congestion Inferences", 
                    color=self.colors['congestion'],
                    alpha=0.75, 
                    linewidth=3)
        
        # Fill congestion periods
        for inf in results.inferences:
            if inf.is_congested:
                axes[2].axvspan(inf.start_timestamp, inf.end_timestamp, 
                               alpha=0.3, color=self.colors['congestion'])
        
        # Labels and formatting
        axes[0].set_ylabel('RTT (ms)', fontsize=14)
        axes[1].set_ylabel('min RTT (ms)', fontsize=14)
        axes[2].set_ylabel('Congestion', fontsize=14)
        axes[2].set_xlabel('Time', fontsize=14)
        
        # Set y-axis limits for congestion plot
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(['Normal', 'Congested'])
        
        # Add legends
        for ax in axes:
            ax.legend(loc='upper right', fontsize=12)
            ax.tick_params(labelsize=12)
        
        # Format x-axis
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_change_points(
        self,
        dataset: MinimumRTTDataset,
        change_points: List[ChangePoint],
        title: str = "Change Point Detection",
        save_path: Optional[Path] = None
    ) -> "plt.Figure":
        """
        Plot change points overlaid on RTT data.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            RTT dataset
        change_points : List[ChangePoint]
            Detected change points
        title : str
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot RTT data
        timestamps = [m.timestamp for m in dataset.measurements]
        values = [m.rtt_value for m in dataset.measurements]
        
        ax.plot(timestamps, values, 
               label="RTT", 
               color=self.colors['rtt'],
               linewidth=2, 
               alpha=0.8)
        
        # Plot change points
        for cp in change_points:
            ax.axvline(x=cp.timestamp, 
                      color=self.colors['change_point'], 
                      linestyle='--', 
                      alpha=0.8,
                      linewidth=2)
            
            # Add confidence annotation
            ax.annotate(f'CP\n{cp.confidence:.2f}',
                       xy=(cp.timestamp, max(values) * 0.9),
                       xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=9,
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=self.colors['change_point'], 
                                alpha=0.3))
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('RTT (ms)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confidence_heatmap(
        self,
        results: CongestionInferenceResult,
        title: str = "Confidence Heatmap",
        save_path: Optional[Path] = None
    ) -> "plt.Figure":
        """
        Plot confidence scores as a heatmap.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results
        title : str
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data
        times = []
        confidences = []
        congestion_status = []
        
        for inf in results.inferences:
            times.append(inf.start_timestamp)
            confidences.append(inf.confidence)
            congestion_status.append(1 if inf.is_congested else 0)
        
        # Create heatmap data
        time_indices = np.arange(len(times))
        heatmap_data = np.array([confidences, congestion_status])
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, 
                      cmap='RdYlBu_r', 
                      aspect='auto',
                      interpolation='nearest')
        
        # Set labels
        step = max(1, len(time_indices)//10)  # Ensure step is at least 1
        ax.set_xticks(time_indices[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in times[::step]])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Confidence', 'Congestion'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        ax.set_xlabel('Time')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_comparison(
        self,
        dataset: MinimumRTTDataset,
        algorithm_results: Dict[str, List[ChangePoint]],
        title: str = "Algorithm Comparison",
        save_path: Optional[Path] = None
    ) -> "plt.Figure":
        """
        Compare different algorithms' change point detection.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            RTT dataset
        algorithm_results : Dict[str, List[ChangePoint]]
            Results from different algorithms
        title : str
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(len(algorithm_results) + 1, 1, 
                                figsize=(self.figsize[0], self.figsize[1] * len(algorithm_results)))
        
        if len(algorithm_results) == 0:
            axes = [axes]
        
        # Plot RTT data on top
        timestamps = [m.timestamp for m in dataset.measurements]
        values = [m.rtt_value for m in dataset.measurements]
        
        axes[0].plot(timestamps, values, 
                    label="RTT", 
                    color=self.colors['rtt'],
                    linewidth=2, 
                    alpha=0.8)
        axes[0].set_title("RTT Data")
        axes[0].set_ylabel('RTT (ms)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot each algorithm's results
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        
        for i, (algorithm, change_points) in enumerate(algorithm_results.items()):
            ax = axes[i + 1]
            
            # Plot RTT data as background
            ax.plot(timestamps, values, 
                   color='lightgray', 
                   linewidth=1, 
                   alpha=0.5)
            
            # Plot change points
            for cp in change_points:
                ax.axvline(x=cp.timestamp, 
                          color=colors[i % len(colors)], 
                          linestyle='--', 
                          alpha=0.8,
                          linewidth=2)
            
            ax.set_title(f"{algorithm.upper()} - {len(change_points)} change points")
            ax.set_ylabel('RTT (ms)')
            ax.grid(True, alpha=0.3)
        
        # Format x-axis for bottom subplot
        axes[-1].set_xlabel('Time')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_summary_statistics(
        self,
        results: CongestionInferenceResult,
        title: str = "Analysis Summary",
        save_path: Optional[Path] = None
    ) -> "plt.Figure":
        """
        Plot summary statistics and metrics.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results
        title : str
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        congested_periods = results.get_congested_periods()
        confidences = [inf.confidence for inf in congested_periods]
        durations = [inf.end_epoch - inf.start_epoch for inf in congested_periods]
        
        # Plot 1: Congestion periods over time
        times = [inf.start_timestamp for inf in results.inferences]
        congestion_binary = [1 if inf.is_congested else 0 for inf in results.inferences]
        
        axes[0, 0].plot(times, congestion_binary, 
                       marker='o', 
                       linewidth=2,
                       color=self.colors['congestion'])
        axes[0, 0].set_title('Congestion Over Time')
        axes[0, 0].set_ylabel('Congested')
        axes[0, 0].set_ylim(-0.1, 1.1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Confidence distribution
        if confidences:
            axes[0, 1].hist(confidences, bins=20, alpha=0.7, color=self.colors['confidence'])
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Duration distribution
        if durations:
            axes[1, 0].hist(durations, bins=20, alpha=0.7, color=self.colors['min_rtt'])
            axes[1, 0].set_title('Congestion Duration Distribution')
            axes[1, 0].set_xlabel('Duration (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary metrics
        total_periods = len(results.inferences)
        congested_count = len(congested_periods)
        total_duration = sum(durations) if durations else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        metrics = {
            'Total Periods': total_periods,
            'Congested Periods': congested_count,
            'Total Duration (min)': total_duration / 60,
            'Avg Confidence': avg_confidence
        }
        
        y_pos = np.arange(len(metrics))
        axes[1, 1].barh(y_pos, list(metrics.values()), 
                       color=[self.colors['rtt'], self.colors['congestion'], 
                             self.colors['min_rtt'], self.colors['confidence']])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(list(metrics.keys()))
        axes[1, 1].set_title('Summary Metrics')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(
        self,
        raw_data: RTTDataset,
        min_rtt_data: MinimumRTTDataset,
        results: CongestionInferenceResult,
        change_points: List[ChangePoint],
        output_dir: Path,
        prefix: str = "jitterbug"
    ) -> Dict[str, Path]:
        """
        Generate and save all standard plots.
        
        Parameters
        ----------
        raw_data : RTTDataset
            Raw RTT measurements
        min_rtt_data : MinimumRTTDataset
            Minimum RTT data
        results : CongestionInferenceResult
            Analysis results
        change_points : List[ChangePoint]
            Detected change points
        output_dir : Path
            Output directory
        prefix : str
            Filename prefix
            
        Returns
        -------
        Dict[str, Path]
            Dictionary of plot_name -> file_path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # Congestion analysis plot
        path = output_dir / f"{prefix}_congestion_analysis.png"
        self.plot_congestion_analysis(raw_data, min_rtt_data, results, save_path=path)
        saved_plots['congestion_analysis'] = path
        
        # Change points plot
        path = output_dir / f"{prefix}_change_points.png"
        self.plot_change_points(min_rtt_data, change_points, save_path=path)
        saved_plots['change_points'] = path
        
        # Confidence heatmap
        path = output_dir / f"{prefix}_confidence_heatmap.png"
        self.plot_confidence_heatmap(results, save_path=path)
        saved_plots['confidence_heatmap'] = path
        
        # Summary statistics
        path = output_dir / f"{prefix}_summary_stats.png"
        self.plot_summary_statistics(results, save_path=path)
        saved_plots['summary_stats'] = path
        
        # RTT time series
        path = output_dir / f"{prefix}_rtt_timeseries.png"
        self.plot_rtt_timeseries({"Raw RTT": raw_data, "Min RTT": min_rtt_data}, save_path=path)
        saved_plots['rtt_timeseries'] = path
        
        return saved_plots
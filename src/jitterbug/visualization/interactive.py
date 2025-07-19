"""
Interactive visualization tools for Jitterbug using Plotly.
"""

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import plotly.graph_objects as go

from ..models import (
    RTTDataset,
    MinimumRTTDataset,
    CongestionInferenceResult,
    ChangePoint,
    CongestionInference
)


class InteractiveVisualizer:
    """
    Interactive visualization tools using Plotly for web-based dashboards.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the interactive visualizer.
        
        Parameters
        ----------
        theme : str
            Plotly theme to use
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for interactive visualization. Install with: pip install jitterbug[visualization]")
        
        self.theme = theme
        self.colors = {
            'rtt': '#1f77b4',
            'min_rtt': '#ff7f0e', 
            'congestion': '#d62728',
            'change_point': '#2ca02c',
            'confidence': '#9467bd',
            'jitter': '#8c564b',
            'normal': '#90EE90',
            'congested': '#FFB6C1'
        }
    
    def create_interactive_timeline(
        self,
        raw_data: RTTDataset,
        min_rtt_data: MinimumRTTDataset,
        results: CongestionInferenceResult,
        change_points: List[ChangePoint],
        title: str = "Interactive Jitterbug Analysis",
        height: int = 800
    ) -> "go.Figure":
        """
        Create an interactive timeline with all analysis results.
        
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
        title : str
            Plot title
        height : int
            Plot height in pixels
            
        Returns
        -------
        go.Figure
            Interactive plotly figure
        """
        # Create subplots with consistent x-axis ranges but no shared_xaxes to avoid annotation issues
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Raw RTT Measurements', 'Minimum RTT', 'Congestion Analysis'),
            shared_xaxes=False,  # Disabled to avoid annotation conflicts, will sync manually
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Raw RTT data - convert epoch to datetime for consistent x-axis
        raw_times = [datetime.fromtimestamp(m.epoch) for m in raw_data.measurements]
        raw_values = [m.rtt_value for m in raw_data.measurements]
        
        fig.add_trace(
            go.Scatter(
                x=raw_times,
                y=raw_values,
                mode='lines',
                name='Raw RTT',
                line=dict(color=self.colors['rtt'], width=1),
                hovertemplate='<b>Raw RTT</b><br>' +
                             'Time: %{x}<br>' +
                             'RTT: %{y:.2f} ms<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Minimum RTT data - convert epoch to datetime for consistent x-axis
        min_times = [datetime.fromtimestamp(m.epoch) for m in min_rtt_data.measurements]
        min_values = [m.rtt_value for m in min_rtt_data.measurements]
        
        fig.add_trace(
            go.Scatter(
                x=min_times,
                y=min_values,
                mode='lines+markers',
                name='Min RTT',
                line=dict(color=self.colors['min_rtt'], width=2),
                marker=dict(size=4),
                hovertemplate='<b>Min RTT</b><br>' +
                             'Time: %{x}<br>' +
                             'RTT: %{y:.2f} ms<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add change points as vertical shapes - more reliable than add_vline with datetime
        for cp in change_points:
            cp_datetime = datetime.fromtimestamp(cp.epoch)
            # Add vertical line shapes to each subplot
            for row_num in [1, 2, 3]:
                # Format yref correctly (y for first subplot, y2 for second, y3 for third)
                yref = "y domain" if row_num == 1 else f"y{row_num} domain"
                
                fig.add_shape(
                    type="line",
                    x0=cp_datetime, x1=cp_datetime,
                    y0=0, y1=1,
                    yref=yref,
                    line=dict(color=self.colors['change_point'], width=2, dash='dash'),
                    row=row_num, col=1
                )
                
                # Add annotation only to middle subplot
                if row_num == 2:
                    fig.add_annotation(
                        x=cp_datetime,
                        y=1.05,
                        yref=yref,
                        text=f'CP ({cp.confidence:.2f})',
                        showarrow=False,
                        font=dict(size=10),
                        xanchor='center',
                        row=row_num, col=1
                    )
        
        # Congestion analysis - use datetime objects directly for consistent x-axis
        congestion_times = []
        congestion_values = []
        hover_texts = []
        
        for inf in results.inferences:
            # Use datetime objects directly (they should already be datetime instances)
            congestion_times.extend([inf.start_timestamp, inf.end_timestamp])
            value = 1.0 if inf.is_congested else 0.0
            congestion_values.extend([value, value])
            
            hover_text = f'<b>Period: {inf.start_timestamp.strftime("%H:%M:%S")} - {inf.end_timestamp.strftime("%H:%M:%S")}</b><br>' + \
                        f'Status: {"Congested" if inf.is_congested else "Normal"}<br>' + \
                        f'Confidence: {inf.confidence:.3f}<br>'
            
            if inf.latency_jump:
                hover_text += f'Latency Jump: {inf.latency_jump.magnitude:.2f} ms<br>'
            
            if inf.jitter_analysis:
                hover_text += f'Jitter Analysis: {inf.jitter_analysis.method}<br>' + \
                             f'Jitter Metric: {inf.jitter_analysis.jitter_metric:.3f}<br>'
                             
            hover_texts.extend([hover_text, hover_text])
        
        fig.add_trace(
            go.Scatter(
                x=congestion_times,
                y=congestion_values,
                mode='lines',
                name='Congestion',
                line=dict(color=self.colors['congestion'], width=3),
                fill='tozeroy',
                fillcolor=f'rgba(214, 39, 40, 0.3)',
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts
            ),
            row=3, col=1
        )
        
        # Add congestion period highlights
        for inf in results.inferences:
            if inf.is_congested:
                fig.add_vrect(
                    x0=inf.start_timestamp,
                    x1=inf.end_timestamp,
                    fillcolor=self.colors['congested'],
                    opacity=0.2,
                    line_width=0,
                    annotation_text=f"Congested ({inf.confidence:.2f})",
                    annotation_position='top left',
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            showlegend=True,
            hovermode='x unified',
            template=self.theme
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="RTT (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Min RTT (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Congestion", row=3, col=1, 
                        tickvals=[0, 1], ticktext=['Normal', 'Congested'])
        
        # Calculate overall time range for consistent x-axis across all subplots
        all_times = raw_times + min_times + congestion_times
        time_range = [min(all_times), max(all_times)]
        
        # Update x-axes with consistent datetime formatting and range
        for row_num in [1, 2, 3]:
            fig.update_xaxes(
                title_text="Time" if row_num == 3 else None,
                row=row_num, col=1,
                type='date',
                tickformat='%H:%M:%S',
                range=time_range,
                matches='x'  # This ensures consistent zooming behavior
            )
        
        return fig
    
    def create_confidence_scatter(
        self,
        results: CongestionInferenceResult,
        title: str = "Confidence vs Duration Analysis"
    ) -> "go.Figure":
        """
        Create an interactive scatter plot of confidence vs duration.
        
        Parameters
        ----------
        results : CongestionInferenceResult
            Analysis results
        title : str
            Plot title
            
        Returns
        -------
        go.Figure
            Interactive plotly figure
        """
        # Extract data
        confidences = []
        durations = []
        statuses = []
        hover_texts = []
        
        for inf in results.inferences:
            confidences.append(inf.confidence)
            duration = (inf.end_epoch - inf.start_epoch) / 60  # Convert to minutes
            durations.append(duration)
            statuses.append('Congested' if inf.is_congested else 'Normal')
            
            hover_text = f'<b>Period: {inf.start_timestamp.strftime("%H:%M:%S")} - {inf.end_timestamp.strftime("%H:%M:%S")}</b><br>' + \
                        f'Status: {"Congested" if inf.is_congested else "Normal"}<br>' + \
                        f'Confidence: {inf.confidence:.3f}<br>' + \
                        f'Duration: {duration:.1f} minutes<br>'
            
            hover_texts.append(hover_text)
        
        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=durations,
            y=confidences,
            mode='markers',
            marker=dict(
                size=10,
                color=[self.colors['congestion'] if s == 'Congested' else self.colors['normal'] for s in statuses],
                opacity=0.7,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Analysis Periods'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Duration (minutes)",
            yaxis_title="Confidence Score",
            hovermode='closest',
            template=self.theme
        )
        
        return fig
    
    def create_algorithm_comparison_plot(
        self,
        dataset: MinimumRTTDataset,
        algorithm_results: Dict[str, List[ChangePoint]],
        title: str = "Algorithm Comparison"
    ) -> "go.Figure":
        """
        Create an interactive comparison of different algorithms.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            RTT dataset
        algorithm_results : Dict[str, List[ChangePoint]]
            Results from different algorithms
        title : str
            Plot title
            
        Returns
        -------
        go.Figure
            Interactive plotly figure
        """
        # Extract RTT data - convert epoch to datetime for consistent x-axis
        timestamps = [datetime.fromtimestamp(m.epoch) for m in dataset.measurements]
        values = [m.rtt_value for m in dataset.measurements]
        
        # Create subplots
        fig = make_subplots(
            rows=len(algorithm_results) + 1, cols=1,
            subplot_titles=['RTT Data'] + [f'{alg.upper()} Algorithm' for alg in algorithm_results.keys()],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Add RTT data to top plot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name='RTT',
                line=dict(color=self.colors['rtt'], width=2),
                hovertemplate='<b>RTT</b><br>' +
                             'Time: %{x}<br>' +
                             'RTT: %{y:.2f} ms<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add algorithm results
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        
        for i, (algorithm, change_points) in enumerate(algorithm_results.items()):
            # Add background RTT data
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=f'{algorithm}_background',
                    line=dict(color='lightgray', width=1),
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate='<b>RTT</b><br>' +
                                 'Time: %{x}<br>' +
                                 'RTT: %{y:.2f} ms<extra></extra>'
                ),
                row=i+2, col=1
            )
            
            # Add change points as vertical shapes - more reliable than add_vline
            for cp in change_points:
                cp_datetime = datetime.fromtimestamp(cp.epoch)
                # Format yref correctly (y for first, y2 for second, etc.)
                subplot_num = i + 2
                yref = "y domain" if subplot_num == 1 else f"y{subplot_num} domain"
                
                fig.add_shape(
                    type="line",
                    x0=cp_datetime, x1=cp_datetime,
                    y0=0, y1=1,
                    yref=yref,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    row=subplot_num, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    x=cp_datetime,
                    y=1.05,
                    yref=yref,
                    text=f'CP ({cp.confidence:.2f})',
                    showarrow=False,
                    font=dict(size=10),
                    xanchor='center',
                    row=subplot_num, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=200 * (len(algorithm_results) + 1),
            showlegend=True,
            template=self.theme
        )
        
        # Update y-axes
        for i in range(len(algorithm_results) + 1):
            fig.update_yaxes(title_text="RTT (ms)", row=i+1, col=1)
        
        # Update x-axes with consistent datetime formatting
        fig.update_xaxes(
            title_text="Time", 
            row=len(algorithm_results)+1, col=1,
            type='date',
            tickformat='%H:%M:%S'
        )
        
        # Ensure all subplots use the same datetime formatting
        fig.update_xaxes(type='date', tickformat='%H:%M:%S')
        
        return fig
    
    def create_dashboard(
        self,
        raw_data: RTTDataset,
        min_rtt_data: MinimumRTTDataset,
        results: CongestionInferenceResult,
        change_points: List[ChangePoint],
        title: str = "Jitterbug Analysis Dashboard"
    ) -> "go.Figure":
        """
        Create a comprehensive dashboard with multiple visualizations.
        
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
        title : str
            Dashboard title
            
        Returns
        -------
        go.Figure
            Interactive plotly figure
        """
        # Create subplots with custom spacing
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RTT Timeline', 'Confidence Distribution', 
                           'Congestion Periods', 'Duration vs Confidence'),
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. RTT Timeline with congestion overlay - convert epoch to datetime
        min_times = [datetime.fromtimestamp(m.epoch) for m in min_rtt_data.measurements]
        min_values = [m.rtt_value for m in min_rtt_data.measurements]
        
        fig.add_trace(
            go.Scatter(
                x=min_times,
                y=min_values,
                mode='lines',
                name='Min RTT',
                line=dict(color=self.colors['min_rtt'], width=2),
                hovertemplate='<b>Min RTT</b><br>Time: %{x}<br>RTT: %{y:.2f} ms<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add congestion periods as background
        for inf in results.inferences:
            if inf.is_congested:
                fig.add_vrect(
                    x0=inf.start_timestamp,
                    x1=inf.end_timestamp,
                    fillcolor=self.colors['congested'],
                    opacity=0.3,
                    line_width=0,
                    row=1, col=1
                )
        
        # 2. Confidence Distribution
        congested_periods = results.get_congested_periods()
        confidences = [inf.confidence for inf in congested_periods]
        
        if confidences:
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    name='Confidence',
                    marker_color=self.colors['confidence'],
                    opacity=0.7,
                    hovertemplate='Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Congestion Periods Bar Chart
        periods_by_hour = {}
        for inf in results.inferences:
            hour = inf.start_timestamp.hour
            if hour not in periods_by_hour:
                periods_by_hour[hour] = {'normal': 0, 'congested': 0}
            
            if inf.is_congested:
                periods_by_hour[hour]['congested'] += 1
            else:
                periods_by_hour[hour]['normal'] += 1
        
        hours = sorted(periods_by_hour.keys())
        normal_counts = [periods_by_hour[h]['normal'] for h in hours]
        congested_counts = [periods_by_hour[h]['congested'] for h in hours]
        
        fig.add_trace(
            go.Bar(
                x=hours,
                y=normal_counts,
                name='Normal',
                marker_color=self.colors['normal'],
                hovertemplate='Hour: %{x}<br>Normal Periods: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=hours,
                y=congested_counts,
                name='Congested',
                marker_color=self.colors['congestion'],
                hovertemplate='Hour: %{x}<br>Congested Periods: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Duration vs Confidence Scatter
        durations = [(inf.end_epoch - inf.start_epoch) / 60 for inf in congested_periods]
        confidences_scatter = [inf.confidence for inf in congested_periods]
        
        if durations and confidences_scatter:
            fig.add_trace(
                go.Scatter(
                    x=durations,
                    y=confidences_scatter,
                    mode='markers',
                    name='Congestion Analysis',
                    marker=dict(
                        size=10,
                        color=self.colors['congestion'],
                        opacity=0.7,
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    hovertemplate='Duration: %{x:.1f} min<br>Confidence: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            template=self.theme
        )
        
        # Update axes with consistent datetime formatting
        fig.update_xaxes(title_text="Time", row=1, col=1, type='date', tickformat='%H:%M:%S')
        fig.update_yaxes(title_text="RTT (ms)", row=1, col=1)
        
        fig.update_xaxes(title_text="Confidence", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Period Count", row=2, col=1)
        
        fig.update_xaxes(title_text="Duration (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Confidence", row=2, col=2)
        
        return fig
    
    def save_html(self, fig: go.Figure, filepath: Path, include_plotlyjs: bool = True) -> None:
        """
        Save interactive plot as HTML file.
        
        Parameters
        ----------
        fig : go.Figure
            Plotly figure to save
        filepath : Path
            Output file path
        include_plotlyjs : bool
            Whether to include plotly.js in the HTML
        """
        fig.write_html(
            str(filepath),
            include_plotlyjs=include_plotlyjs,
            div_id="jitterbug-plot"
        )
    
    def show(self, fig: go.Figure) -> None:
        """
        Display the interactive plot.
        
        Parameters
        ----------
        fig : go.Figure
            Plotly figure to display
        """
        fig.show()
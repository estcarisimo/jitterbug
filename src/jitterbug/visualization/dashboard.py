"""
Dashboard wrapper for comprehensive Jitterbug visualization.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

from .plotter import JitterbugPlotter
from .interactive import InteractiveVisualizer
from ..models import (
    RTTDataset,
    MinimumRTTDataset,
    CongestionInferenceResult,
    ChangePoint
)


class JitterbugDashboard:
    """
    Comprehensive dashboard for Jitterbug analysis visualization.
    
    This class provides a high-level interface for creating both static
    and interactive visualizations of network analysis results.
    """
    
    def __init__(self, 
                 static_style: str = "default",
                 interactive_theme: str = "plotly_white",
                 figsize: tuple = (15, 8)):
        """
        Initialize the dashboard.
        
        Parameters
        ----------
        static_style : str
            Matplotlib style for static plots
        interactive_theme : str
            Plotly theme for interactive plots
        figsize : tuple
            Default figure size for static plots
        """
        self.plotter = JitterbugPlotter(style=static_style, figsize=figsize)
        self.interactive = InteractiveVisualizer(theme=interactive_theme)
        
    def create_comprehensive_report(
        self,
        raw_data: RTTDataset,
        min_rtt_data: MinimumRTTDataset,
        results: CongestionInferenceResult,
        change_points: List[ChangePoint],
        output_dir: Path,
        title: str = "Jitterbug Analysis Report",
        include_interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Create a comprehensive analysis report with all visualizations.
        
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
            Output directory for report
        title : str
            Report title
        include_interactive : bool
            Whether to include interactive plots
            
        Returns
        -------
        Dict[str, Any]
            Report metadata with file paths and statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'title': title,
            'output_dir': str(output_dir),
            'static_plots': {},
            'interactive_plots': {},
            'statistics': {},
            'metadata': {}
        }
        
        # Generate static plots
        print("📊 Generating static plots...")
        static_plots = self.plotter.save_all_plots(
            raw_data=raw_data,
            min_rtt_data=min_rtt_data,
            results=results,
            change_points=change_points,
            output_dir=output_dir / "static",
            prefix="jitterbug_report"
        )
        report['static_plots'] = {k: str(v) for k, v in static_plots.items()}
        
        # Generate interactive plots
        if include_interactive:
            print("🌐 Generating interactive plots...")
            interactive_dir = output_dir / "interactive"
            interactive_dir.mkdir(exist_ok=True)
            
            # Main timeline
            timeline_fig = self.interactive.create_interactive_timeline(
                raw_data, min_rtt_data, results, change_points,
                title=f"{title} - Interactive Timeline"
            )
            timeline_path = interactive_dir / "timeline.html"
            self.interactive.save_html(timeline_fig, timeline_path)
            report['interactive_plots']['timeline'] = str(timeline_path)
            
            # Dashboard
            dashboard_fig = self.interactive.create_dashboard(
                raw_data, min_rtt_data, results, change_points,
                title=f"{title} - Dashboard"
            )
            dashboard_path = interactive_dir / "dashboard.html"
            self.interactive.save_html(dashboard_fig, dashboard_path)
            report['interactive_plots']['dashboard'] = str(dashboard_path)
            
            # Confidence scatter
            confidence_fig = self.interactive.create_confidence_scatter(
                results, title=f"{title} - Confidence Analysis"
            )
            confidence_path = interactive_dir / "confidence_scatter.html"
            self.interactive.save_html(confidence_fig, confidence_path)
            report['interactive_plots']['confidence_scatter'] = str(confidence_path)
        
        # Calculate statistics
        print("📈 Calculating statistics...")
        stats = self._calculate_statistics(results, change_points)
        report['statistics'] = stats
        
        # Add metadata
        report['metadata'] = {
            'raw_data_points': len(raw_data.measurements),
            'min_rtt_points': len(min_rtt_data.measurements),
            'total_periods': len(results.inferences),
            'congested_periods': len(results.get_congested_periods()),
            'change_points': len(change_points),
            'analysis_duration_hours': (
                max(m.timestamp for m in raw_data.measurements) -
                min(m.timestamp for m in raw_data.measurements)
            ).total_seconds() / 3600
        }
        
        # Save report metadata
        report_path = output_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML index
        self._generate_html_index(report, output_dir)
        
        print(f"✅ Report generated successfully in {output_dir}")
        return report
    
    def create_algorithm_comparison_report(
        self,
        dataset: MinimumRTTDataset,
        algorithm_results: Dict[str, List[ChangePoint]],
        output_dir: Path,
        title: str = "Algorithm Comparison Report"
    ) -> Dict[str, Any]:
        """
        Create a comparison report for different algorithms.
        
        Parameters
        ----------
        dataset : MinimumRTTDataset
            RTT dataset
        algorithm_results : Dict[str, List[ChangePoint]]
            Results from different algorithms
        output_dir : Path
            Output directory
        title : str
            Report title
            
        Returns
        -------
        Dict[str, Any]
            Report metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'title': title,
            'output_dir': str(output_dir),
            'algorithms': list(algorithm_results.keys()),
            'static_plots': {},
            'interactive_plots': {},
            'comparison_stats': {}
        }
        
        # Static comparison plot
        static_path = output_dir / "algorithm_comparison_static.png"
        self.plotter.plot_algorithm_comparison(
            dataset, algorithm_results, 
            title=title,
            save_path=static_path
        )
        report['static_plots']['comparison'] = str(static_path)
        
        # Interactive comparison plot
        interactive_fig = self.interactive.create_algorithm_comparison_plot(
            dataset, algorithm_results, title=title
        )
        interactive_path = output_dir / "algorithm_comparison_interactive.html"
        self.interactive.save_html(interactive_fig, interactive_path)
        report['interactive_plots']['comparison'] = str(interactive_path)
        
        # Calculate comparison statistics
        comparison_stats = {}
        for algo, change_points in algorithm_results.items():
            comparison_stats[algo] = {
                'total_change_points': len(change_points),
                'avg_confidence': sum(cp.confidence for cp in change_points) / len(change_points) if change_points else 0,
                'confidence_std': 0 if len(change_points) <= 1 else 
                    (sum((cp.confidence - comparison_stats[algo]['avg_confidence'])**2 for cp in change_points) / (len(change_points) - 1))**0.5
            }
        
        report['comparison_stats'] = comparison_stats
        
        # Save report
        report_path = output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def quick_visualize(
        self,
        raw_data: RTTDataset,
        min_rtt_data: MinimumRTTDataset,
        results: CongestionInferenceResult,
        change_points: List[ChangePoint],
        output_format: str = "interactive",
        save_path: Optional[Path] = None
    ) -> Union[plt.Figure, go.Figure]:
        """
        Quick visualization for immediate analysis.
        
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
        output_format : str
            'static' or 'interactive'
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        Union[plt.Figure, go.Figure]
            Generated figure
        """
        if output_format == "static":
            fig = self.plotter.plot_congestion_analysis(
                raw_data, min_rtt_data, results,
                title="Quick Congestion Analysis",
                save_path=save_path
            )
        else:
            fig = self.interactive.create_interactive_timeline(
                raw_data, min_rtt_data, results, change_points,
                title="Quick Interactive Analysis"
            )
            if save_path:
                self.interactive.save_html(fig, save_path)
        
        return fig
    
    def _calculate_statistics(
        self,
        results: CongestionInferenceResult,
        change_points: List[ChangePoint]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics."""
        congested_periods = results.get_congested_periods()
        
        if not congested_periods:
            return {
                'total_periods': len(results.inferences),
                'congested_periods': 0,
                'congestion_ratio': 0,
                'total_congestion_duration': 0,
                'avg_congestion_duration': 0,
                'avg_confidence': 0,
                'change_points': len(change_points)
            }
        
        # Duration statistics
        durations = [inf.end_epoch - inf.start_epoch for inf in congested_periods]
        confidences = [inf.confidence for inf in congested_periods]
        
        return {
            'total_periods': len(results.inferences),
            'congested_periods': len(congested_periods),
            'congestion_ratio': len(congested_periods) / len(results.inferences),
            'total_congestion_duration': sum(durations),
            'avg_congestion_duration': sum(durations) / len(durations),
            'min_congestion_duration': min(durations),
            'max_congestion_duration': max(durations),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'change_points': len(change_points),
            'change_points_per_hour': len(change_points) / (
                max(m.timestamp for m in results.inferences) - 
                min(m.timestamp for m in results.inferences)
            ).total_seconds() * 3600 if results.inferences else 0
        }
    
    def _generate_html_index(self, report: Dict[str, Any], output_dir: Path) -> None:
        """Generate HTML index file for the report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .label {{ font-size: 14px; color: #7f8c8d; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report['title']}</h1>
                <p>Generated on: {report.get('generated_at', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Key Statistics</h2>
                <div class="grid">
                    <div class="card">
                        <div class="metric">{report['statistics'].get('total_periods', 0)}</div>
                        <div class="label">Total Analysis Periods</div>
                    </div>
                    <div class="card">
                        <div class="metric">{report['statistics'].get('congested_periods', 0)}</div>
                        <div class="label">Congested Periods</div>
                    </div>
                    <div class="card">
                        <div class="metric">{report['statistics'].get('congestion_ratio', 0):.1%}</div>
                        <div class="label">Congestion Ratio</div>
                    </div>
                    <div class="card">
                        <div class="metric">{report['statistics'].get('change_points', 0)}</div>
                        <div class="label">Change Points Detected</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🖼️ Static Plots</h2>
                <ul>
        """
        
        for name, path in report['static_plots'].items():
            rel_path = Path(path).relative_to(output_dir)
            html_content += f'<li><a href="{rel_path}">{name.replace("_", " ").title()}</a></li>'
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>🌐 Interactive Plots</h2>
                <ul>
        """
        
        for name, path in report['interactive_plots'].items():
            rel_path = Path(path).relative_to(output_dir)
            html_content += f'<li><a href="{rel_path}">{name.replace("_", " ").title()}</a></li>'
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>📄 Report Files</h2>
                <ul>
                    <li><a href="report.json">Raw Report Data (JSON)</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / "index.html", 'w') as f:
            f.write(html_content)
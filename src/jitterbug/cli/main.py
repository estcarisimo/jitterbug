"""
Main CLI application using Typer.
"""

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..analyzer import JitterbugAnalyzer
from ..io import DataLoader
from ..models import JitterbugConfig

# matplotlib is an optional dependency; the visualize command checks for it at run time.
try:
    from ..visualization.plotter import MATPLOTLIB_AVAILABLE
except ImportError:  # pragma: no cover - only if the package itself is broken
    MATPLOTLIB_AVAILABLE = False


# Initialize Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="jitterbug",
    help="Jitterbug: Framework for Jitter-Based Congestion Inference",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command()
def analyze(
    input_file: Path = typer.Argument(
        ..., help="Path to RTT data file (CSV, JSON, or InfluxDB export)", exists=True
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path (default: stdout)"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path (YAML or JSON)"
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Input file format (csv, json, influx). Auto-detected if not specified.",
    ),
    output_format: str | None = typer.Option(
        "json", "--output-format", help="Output format (json, csv, parquet)"
    ),
    method: str | None = typer.Option(
        "jitter_dispersion",
        "--method",
        "-m",
        help="Jitter analysis method (jitter_dispersion, ks_test)",
    ),
    algorithm: str | None = typer.Option(
        "ruptures", "--algorithm", "-a", help="Change point detection algorithm (ruptures, bcp)"
    ),
    threshold: float | None = typer.Option(
        0.25, "--threshold", "-t", help="Change point detection threshold"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    summary_only: bool = typer.Option(
        False, "--summary-only", help="Show only summary statistics (no detailed periods)"
    ),
):
    """
    Analyze RTT data for network congestion inference.

    [bold]Examples:[/bold]

    • Basic analysis:
      [cyan]jitterbug analyze rtts.csv[/cyan]

    • With custom configuration:
      [cyan]jitterbug analyze rtts.csv --config config.yaml --output results.json[/cyan]

    • Using KS-test method:
      [cyan]jitterbug analyze rtts.csv --method ks_test --algorithm bcp[/cyan]
    """
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()

        # Override config with command-line arguments
        jitterbug_config.jitter_analysis.method = method
        jitterbug_config.change_point_detection.algorithm = algorithm
        jitterbug_config.change_point_detection.threshold = threshold
        jitterbug_config.verbose = verbose
        jitterbug_config.output_format = output_format

        # Create analyzer
        analyzer = JitterbugAnalyzer(jitterbug_config)

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Load data
            task = progress.add_task("Loading RTT data...", total=None)
            results = analyzer.analyze_from_file(input_file, format)

            # Save results
            if output:
                progress.update(task, description="Saving results...")
                analyzer.save_results(results, output, output_format)

            progress.update(task, description="Analysis complete!", total=1, completed=1)

        # Display results summary
        _display_results(results, analyzer, summary_only=summary_only)

        # Save results if output file specified
        if output:
            console.print(f"\n✅ Results saved to [bold]{output}[/bold]")
        else:
            # Suggest saving results
            console.print(
                f"\n💡 [dim]Tip: To save full results, use --output flag:[/dim]\n"
                f"   [cyan]jitterbug analyze {input_file} --output results.{output_format}[/cyan]"
            )

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1) from None


@app.command()
def config(
    template: bool = typer.Option(False, "--template", help="Generate a configuration template"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path for configuration template"
    ),
    format: str = typer.Option("yaml", "--format", "-f", help="Configuration format (yaml, json)"),
):
    """
    Manage Jitterbug configuration.

    [bold]Examples:[/bold]

    • Generate configuration template:
      [cyan]jitterbug config --template --output config.yaml[/cyan]

    • Generate JSON configuration:
      [cyan]jitterbug config --template --format json --output config.json[/cyan]
    """
    if template:
        # Create default configuration
        default_config = JitterbugConfig()

        if output:
            default_config.to_file(output)
            console.print(f"✅ Configuration template saved to [bold]{output}[/bold]")
        else:
            # Print to stdout
            if format == "yaml":
                import yaml

                print(yaml.dump(default_config.dict(), default_flow_style=False))
            else:
                print(json.dumps(default_config.dict(), indent=2))
    else:
        console.print("Use --template to generate a configuration template")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Path to RTT data file to validate", exists=True),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Input file format (csv, json, influx). Auto-detected if not specified.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Validate RTT data file format and quality.

    [bold]Examples:[/bold]

    • Validate CSV file:
      [cyan]jitterbug validate rtts.csv[/cyan]

    • Validate with verbose output:
      [cyan]jitterbug validate rtts.csv --verbose[/cyan]
    """
    try:
        # Load data
        data_loader = DataLoader()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading and validating data...", total=None)

            dataset = data_loader.load_from_file(input_file, format)
            validation_results = data_loader.validate_data(dataset)

            progress.update(task, description="Validation complete!", total=1, completed=1)

        # Display validation results
        _display_validation_results(validation_results, verbose)

    except Exception as e:
        console.print(f"❌ Validation failed: {e}", style="red")
        raise typer.Exit(1) from None


@app.command()
def visualize(
    input_file: Path = typer.Argument(
        ..., help="Path to RTT data file (CSV, JSON, or InfluxDB export)", exists=True
    ),
    output_dir: Path = typer.Option(
        "visualization_output",
        "--output-dir",
        "-o",
        help="Output directory for visualization files",
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path (YAML or JSON)"
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Input file format (csv, json, influx). Auto-detected if not specified.",
    ),
    method: str | None = typer.Option(
        "jitter_dispersion",
        "--method",
        "-m",
        help="Jitter analysis method (jitter_dispersion, ks_test)",
    ),
    algorithm: str | None = typer.Option(
        "ruptures",
        "--algorithm",
        "-a",
        help="Change point detection algorithm (ruptures, bcp)",
    ),
    threshold: float | None = typer.Option(
        0.25, "--threshold", "-t", help="Change point detection threshold"
    ),
    prefix: str = typer.Option(
        "jitterbug", "--prefix", help="Filename prefix for the generated PNG files"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run the analysis and save the standard set of plots as PNG files.

    Generates the congestion analysis, change points, confidence heatmap, summary
    statistics and RTT time series figures (matplotlib, 300 dpi).

    [bold]Examples:[/bold]

    • Basic visualization:
      [cyan]jitterbug visualize rtts.csv[/cyan]

    • Custom output directory and algorithm:
      [cyan]jitterbug visualize rtts.csv --output-dir my_plots --algorithm bcp[/cyan]
    """
    if not MATPLOTLIB_AVAILABLE:
        console.print(
            "❌ [bold red]matplotlib not found![/bold red]\n"
            "Install with: [cyan]uv sync --extra visualization[/cyan]",
            style="red",
        )
        raise typer.Exit(1)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()

        # Override config with command-line arguments
        jitterbug_config.jitter_analysis.method = method
        jitterbug_config.change_point_detection.algorithm = algorithm
        jitterbug_config.change_point_detection.threshold = threshold
        jitterbug_config.verbose = verbose

        # Create analyzer
        analyzer = JitterbugAnalyzer(jitterbug_config)

        from ..visualization import JitterbugPlotter

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading and analyzing RTT data...", total=None)
            results = analyzer.analyze_from_file(input_file, format)

            progress.update(task, description="Generating plots...")
            saved = JitterbugPlotter().save_all_plots(
                raw_data=analyzer.raw_data,
                min_rtt_data=analyzer.min_rtt_data,
                results=results,
                change_points=analyzer.change_points or [],
                output_dir=output_dir,
                prefix=prefix,
            )
            progress.update(task, description="Visualization complete!", total=1, completed=1)

        console.print(f"✅ {len(saved)} plots saved to [bold]{output_dir}[/bold]")
        for name, path in saved.items():
            console.print(f"   • {name}: {path.name}")

        stats = analyzer.get_summary_statistics(results)
        console.print("\n📊 [bold]Key Statistics:[/bold]")
        console.print(f"   • Total periods: {stats['total_periods']}")
        console.print(f"   • Congested periods: {stats['congested_periods']}")
        console.print(f"   • Congestion ratio: {stats['congestion_ratio']:.1%}")
        console.print(f"   • Change points: {len(analyzer.change_points or [])}")

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        if verbose:
            import traceback

            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def version():
    """Show Jitterbug version information."""
    from .. import __author__, __email__, __version__

    console.print(
        Panel(
            f"[bold]Jitterbug[/bold] v{__version__}\n"
            f"Framework for Jitter-Based Congestion Inference\n\n"
            f"Author: {__author__}\n"
            f"Email: {__email__}",
            title="Version Information",
            expand=False,
        )
    )


def _display_results(results, analyzer, summary_only=False):
    """Display analysis results in a formatted table."""
    if not results.inferences:
        console.print("🔍 No congestion periods detected", style="yellow")
        return

    # Get summary statistics
    summary_stats = analyzer.get_summary_statistics(results)

    # Display summary
    console.print("\n📊 [bold]Analysis Summary[/bold]")
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold")

    summary_table.add_row("Total Periods", str(summary_stats["total_periods"]))
    summary_table.add_row("Congested Periods", str(summary_stats["congested_periods"]))
    summary_table.add_row("Congestion Ratio", f"{summary_stats['congestion_ratio']:.2%}")
    summary_table.add_row("Total Duration", f"{summary_stats['total_duration_seconds']:.1f}s")
    summary_table.add_row(
        "Congestion Duration", f"{summary_stats['congestion_duration_seconds']:.1f}s"
    )
    summary_table.add_row("Average Confidence", f"{summary_stats['average_confidence']:.2f}")

    console.print(summary_table)

    if not summary_only:
        # Display detailed results
        console.print("\n🔍 [bold]Congestion Periods[/bold]")

        congested_periods = results.get_congested_periods()
        if congested_periods:
            detail_table = Table()
            detail_table.add_column("Start Time", style="cyan")
            detail_table.add_column("End Time", style="cyan")
            detail_table.add_column("Duration", style="yellow")
            detail_table.add_column("Confidence", style="green")
            detail_table.add_column("Latency Jump", style="red")
            detail_table.add_column("Jitter Change", style="blue")

            for period in congested_periods:
                duration = period.end_epoch - period.start_epoch
                latency_jump = "✓" if period.latency_jump and period.latency_jump.has_jump else "✗"
                jitter_change = (
                    "✓"
                    if period.jitter_analysis and period.jitter_analysis.has_significant_jitter
                    else "✗"
                )

                detail_table.add_row(
                    period.start_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    period.end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{duration:.1f}s",
                    f"{period.confidence:.2f}",
                    latency_jump,
                    jitter_change,
                )

            console.print(detail_table)
        else:
            console.print("No congestion periods found", style="yellow")


def _display_validation_results(results, verbose=False):
    """Display data validation results."""
    if not results["valid"]:
        console.print(f"❌ [bold red]Validation Failed[/bold red]: {results['error']}")
        return

    metrics = results["metrics"]

    # Basic validation info
    console.print("✅ [bold green]Data validation passed[/bold green]")

    # Summary table
    table = Table(title="Data Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Total Measurements", str(metrics["total_measurements"]))
    table.add_row("Unique Timestamps", str(metrics["unique_timestamps"]))
    table.add_row("Time Ordered", "✓" if metrics["time_ordered"] else "✗")
    table.add_row("Has Duplicates", "✓" if metrics["has_duplicates"] else "✗")
    table.add_row("Duration", f"{metrics['duration_seconds']:.1f}s")
    table.add_row("Average Interval", f"{metrics['average_interval_seconds']:.1f}s")
    table.add_row("Max Gap", f"{metrics['max_gap_seconds']:.1f}s")

    console.print(table)

    if verbose:
        # RTT statistics
        rtt_stats = metrics["rtt_statistics"]
        console.print("\n📈 [bold]RTT Statistics[/bold]")

        rtt_table = Table()
        rtt_table.add_column("Statistic", style="cyan")
        rtt_table.add_column("Value", style="bold")

        rtt_table.add_row("Minimum", f"{rtt_stats['min']:.2f}ms")
        rtt_table.add_row("Maximum", f"{rtt_stats['max']:.2f}ms")
        rtt_table.add_row("Mean", f"{rtt_stats['mean']:.2f}ms")
        rtt_table.add_row("Median", f"{rtt_stats['median']:.2f}ms")
        rtt_table.add_row("Std Dev", f"{rtt_stats['std']:.2f}ms")
        rtt_table.add_row("Outliers", str(rtt_stats["outliers"]))

        console.print(rtt_table)

        # Time range
        console.print(
            f"\n⏰ [bold]Time Range[/bold]: {metrics['time_range']['start']} "
            f"to {metrics['time_range']['end']}"
        )


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()

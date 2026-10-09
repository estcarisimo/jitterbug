"""
Main CLI application using Typer.
"""

import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..analyzer import JitterbugAnalyzer
from ..io import DataLoader
from ..io.compression import open_text
from ..models import CongestionInferenceResult, JitterbugConfig

# matplotlib is an optional dependency; the visualize command checks for it at run time.
try:
    from ..visualization.plotter import MATPLOTLIB_AVAILABLE
except ImportError:  # pragma: no cover - only if the package itself is broken
    MATPLOTLIB_AVAILABLE = False


# Initialize Rich console
console = Console()
# `jitterbug stream` writes JSON lines to stdout, so its messages go to stderr.
err_console = Console(stderr=True)


def _apply_overrides(
    base: JitterbugConfig,
    *,
    method: str | None = None,
    algorithm: str | None = None,
    threshold: float | None = None,
    output_format: str | None = None,
    verbose: bool = False,
    mode: str | None = None,
    clustering_algorithm: str | None = None,
) -> JitterbugConfig:
    """
    Return a copy of ``base`` with the command-line overrides that were actually given.

    Values are re-validated by Pydantic, so an unknown method or algorithm name is
    rejected with the same message as in a configuration file.
    """
    data = base.model_dump()
    if method is not None:
        data["jitter_analysis"]["method"] = method
    if algorithm is not None:
        data["change_point_detection"]["algorithm"] = algorithm
    if threshold is not None:
        data["change_point_detection"]["threshold"] = threshold
    if output_format is not None:
        data["output_format"] = output_format
    if verbose:
        data["verbose"] = True
    if mode is not None:
        data["analysis_mode"] = mode
    if clustering_algorithm is not None:
        data["clustering"]["algorithm"] = clustering_algorithm
    return JitterbugConfig.model_validate(data)


def _apply_streaming_overrides(
    base: JitterbugConfig,
    *,
    backend: str | None = None,
    decision: str | None = None,
    hazard_lambda: float | None = None,
    min_period_samples: int | None = None,
    min_time_elapsed: int | None = None,
    verbose: bool = False,
) -> JitterbugConfig:
    """Return a copy of ``base`` with the online-mode flags that were actually given."""
    data = base.model_dump()
    if backend is not None:
        data["streaming"]["backend"] = backend
    if decision is not None:
        data["streaming"]["decision"] = decision
    if hazard_lambda is not None:
        data["streaming"]["hazard_lambda"] = hazard_lambda
    if min_period_samples is not None:
        data["streaming"]["min_period_samples"] = min_period_samples
    if min_time_elapsed is not None:
        data["streaming"]["min_time_elapsed"] = min_time_elapsed
    if verbose:
        data["verbose"] = True
    return JitterbugConfig.model_validate(data)


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
        ..., help="Path to RTT data file (CSV or scamper JSON)", exists=True
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
        help="Input file format (csv, json). Auto-detected if not specified.",
    ),
    output_format: str | None = typer.Option(
        None, "--output-format", help="Output format (json, csv, parquet) [default: json]"
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        "-m",
        help="Jitter analysis method (jitter_dispersion, ks_test) [default: jitter_dispersion]",
    ),
    algorithm: str | None = typer.Option(
        None,
        "--algorithm",
        "-a",
        help="Change point detection algorithm (ruptures, bcp) [default: ruptures]",
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", "-t", help="Change point detection threshold [default: 0.25]"
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Analysis mode (sequential, clustering) [default: sequential]",
    ),
    clustering_algorithm: str | None = typer.Option(
        None,
        "--clustering-algorithm",
        help="Clustering algorithm for --mode clustering (gmm, kmeans, kmeans_silhouette) "
        "[default: gmm]",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    summary_only: bool = typer.Option(
        False, "--summary-only", help="Show only summary statistics (no detailed periods)"
    ),
) -> None:
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

        # Command-line flags override the file only when given explicitly
        jitterbug_config = _apply_overrides(
            jitterbug_config,
            method=method,
            algorithm=algorithm,
            threshold=threshold,
            output_format=output_format,
            verbose=verbose,
            mode=mode,
            clustering_algorithm=clustering_algorithm,
        )

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
                analyzer.save_results(results, output, jitterbug_config.output_format)

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
                f"   [cyan]jitterbug analyze {input_file} "
                f"--output results.{jitterbug_config.output_format}[/cyan]"
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
) -> None:
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

                print(yaml.dump(default_config.model_dump(), default_flow_style=False))
            else:
                print(json.dumps(default_config.model_dump(), indent=2))
    else:
        console.print("Use --template to generate a configuration template")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Path to RTT data file to validate", exists=True),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Input file format (csv, json). Auto-detected if not specified.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
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
        ..., help="Path to RTT data file (CSV or scamper JSON)", exists=True
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
        help="Input file format (csv, json). Auto-detected if not specified.",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        "-m",
        help="Jitter analysis method (jitter_dispersion, ks_test) [default: jitter_dispersion]",
    ),
    algorithm: str | None = typer.Option(
        None,
        "--algorithm",
        "-a",
        help="Change point detection algorithm (ruptures, bcp) [default: ruptures]",
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", "-t", help="Change point detection threshold [default: 0.25]"
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Analysis mode (sequential, clustering) [default: sequential]",
    ),
    clustering_algorithm: str | None = typer.Option(
        None,
        "--clustering-algorithm",
        help="Clustering algorithm for --mode clustering (gmm, kmeans, kmeans_silhouette) "
        "[default: gmm]",
    ),
    prefix: str = typer.Option(
        "jitterbug", "--prefix", help="Filename prefix for the generated PNG files"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
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

        # Command-line flags override the file only when given explicitly
        jitterbug_config = _apply_overrides(
            jitterbug_config,
            method=method,
            algorithm=algorithm,
            threshold=threshold,
            verbose=verbose,
            mode=mode,
            clustering_algorithm=clustering_algorithm,
        )

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
            if analyzer.raw_data is None or analyzer.min_rtt_data is None:
                raise RuntimeError("analysis produced no data to plot")
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


def _iter_lines(source: Path | None, follow: bool) -> Iterator[str]:
    """
    Complete lines of ``source`` (stdin when None); with ``follow``, keep waiting for more.

    A writer that is still appending may have flushed half a line; it is held back until
    its newline arrives. Without ``follow`` a trailing unterminated line is yielded at EOF.
    """
    if source is None:
        yield from sys.stdin
        return
    with open_text(source) as f:
        partial = ""
        while True:
            chunk = f.readline()
            if chunk:
                partial += chunk
                if partial.endswith("\n"):
                    yield partial
                    partial = ""
            elif follow:
                time.sleep(0.5)
            else:
                if partial:
                    yield partial
                return


def _parse_sample(line: str) -> tuple[float, float] | None:
    """``epoch,rtt`` (extra columns ignored) or None for blank, header and junk lines."""
    parts = line.strip().split(",")
    if len(parts) < 2:
        return None
    try:
        epoch, rtt = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (math.isfinite(epoch) and math.isfinite(rtt)):
        return None
    return epoch, rtt


@app.command()
def stream(
    source: str = typer.Argument(
        "-", help="File with one 'epoch,rtt' sample per line, or '-' for standard input"
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Keep reading the file as it grows (like tail -f)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write events here as JSON lines instead of stdout"
    ),
    events: str = typer.Option(
        "all", "--events", help="Which events to emit: all, verdicts, change-points"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file (YAML or JSON)", exists=True
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Online back end: bocpd (incremental) or window (offline rerun)"
    ),
    decision: str | None = typer.Option(
        None, "--decision", help="Change point rule: map, lag, window (bocpd back end only)"
    ),
    hazard_lambda: float | None = typer.Option(
        None, "--hazard-lambda", help="Expected run length in bins (bocpd back end only)"
    ),
    min_period_samples: int | None = typer.Option(
        None, "--min-period-samples", help="Jitter samples before a provisional verdict"
    ),
    min_time_elapsed: int | None = typer.Option(
        None,
        "--min-time-elapsed",
        help="Minimum seconds between change points (bocpd; the window back end uses "
        "change_point_detection.min_time_elapsed)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Infer congestion online from a stream of RTT samples.

    The default back end (bocpd) needs the bcp extra; --backend window reruns the offline
    pipeline on a trailing window instead (its cadence and window come from the streaming
    section of --config; --decision and --hazard-lambda do not apply to it).

    Reads 'epoch,rtt' lines (seconds, milliseconds; a header line is skipped) and prints one
    JSON object per event: change points, provisional verdicts and final verdicts. Samples
    must arrive in time order; a sample older than the last one is dropped. The last bin is
    closed when the input ends (or on Ctrl-C with --follow), as 'jitterbug replay' does.

    [bold]Examples:[/bold]

    • From a probe writing to standard output:
      [cyan]my-probe | jitterbug stream[/cyan]

    • Following a file that another process appends to:
      [cyan]jitterbug stream rtts.csv --follow --events verdicts[/cyan]
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    if events not in ("all", "verdicts", "change-points"):
        err_console.print(
            f"❌ Error: --events must be all, verdicts or change-points, not {events}"
        )
        raise typer.Exit(2)
    if follow and source == "-":
        err_console.print("--follow has no effect on standard input", style="yellow")
    try:
        from ..streaming import create_online_analyzer

        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()
        jitterbug_config = _apply_streaming_overrides(
            jitterbug_config,
            backend=backend,
            decision=decision,
            hazard_lambda=hazard_lambda,
            min_period_samples=min_period_samples,
            min_time_elapsed=min_time_elapsed,
            verbose=verbose,
        )
        online = create_online_analyzer(jitterbug_config)
        path = None if source == "-" else Path(source)
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        sink = output.open("w") if output else sys.stdout
        n_samples = n_skipped = n_events = 0
        wanted = {"all": None, "verdicts": "verdict", "change-points": "change_point"}[events]

        def emit(batch: list[Any]) -> None:
            nonlocal n_events
            for event in batch:
                if wanted is not None and event.kind != wanted:
                    continue
                sink.write(json.dumps(event.to_dict()) + "\n")
                sink.flush()
                n_events += 1

        try:
            try:
                for line in _iter_lines(path, follow):
                    sample = _parse_sample(line)
                    if sample is None:
                        n_skipped += bool(line.strip())
                        continue
                    n_samples += 1
                    emit(online.push(*sample))
            except KeyboardInterrupt:
                # Ctrl-C is how a --follow stream ends; close it like an EOF.
                err_console.print("interrupted", style="yellow")
            # The input ended (stdin EOF, a file without --follow, or Ctrl-C): close the
            # open bin so the result matches `jitterbug replay` on the same data.
            emit(online.flush())
        finally:
            if output:
                sink.close()
        err_console.print(
            f"{n_samples} samples read, {n_skipped} lines skipped, {n_events} events emitted"
        )
    except Exception as e:
        err_console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1) from None


@app.command()
def replay(
    input_file: Path = typer.Argument(
        ..., help="RTT data file to replay in time order", exists=True
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write every event as a JSON array to this file"
    ),
    reference: Path | None = typer.Option(
        None,
        "--reference",
        help="CSV with starts,ends,congestion columns to score the final verdicts against",
        exists=True,
    ),
    format: str | None = typer.Option(
        None, "--format", help="Input format (csv, json, scamper); inferred from the file"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file (YAML or JSON)", exists=True
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Online back end: bocpd (incremental) or window (offline rerun)"
    ),
    decision: str | None = typer.Option(
        None, "--decision", help="Change point rule: map, lag, window (bocpd back end only)"
    ),
    hazard_lambda: float | None = typer.Option(
        None, "--hazard-lambda", help="Expected run length in bins (bocpd back end only)"
    ),
    min_period_samples: int | None = typer.Option(
        None, "--min-period-samples", help="Jitter samples before a provisional verdict"
    ),
    min_time_elapsed: int | None = typer.Option(
        None,
        "--min-time-elapsed",
        help="Minimum seconds between change points (bocpd; the window back end uses "
        "change_point_detection.min_time_elapsed)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Replay a recorded dataset through the online pipeline and summarize the result.

    [bold]Examples:[/bold]

    • Paper dataset against the paper's KS reference:
      [cyan]jitterbug replay examples/network_analysis/data/raw.csv \\
        --reference examples/network_analysis/expected_results/kstest_inferences.csv[/cyan]

    • Keep every event:
      [cyan]jitterbug replay rtts.csv --output events.json[/cyan]
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    try:
        from ..streaming import events_to_json, score
        from ..streaming import replay as replay_dataset

        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()
        jitterbug_config = _apply_streaming_overrides(
            jitterbug_config,
            backend=backend,
            decision=decision,
            hazard_lambda=hazard_lambda,
            min_period_samples=min_period_samples,
            min_time_elapsed=min_time_elapsed,
            verbose=verbose,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Replaying...", total=None)
            dataset = DataLoader().load_from_file(input_file, format)
            events = replay_dataset(dataset, jitterbug_config)
            progress.update(task, total=1, completed=1)
        summary = score(events, reference)
        if output:
            events_to_json(events, output)

        table = Table(title="Online replay", show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        labels = {
            "change_points": "Change points",
            "periods": "Periods with a final verdict",
            "congested": "Congested periods",
            "onset_delay_min_median": "Onset delay, median (min)",
            "onset_delay_min_max": "Onset delay, max (min)",
            "return_delay_min_median": "Return delay, median (min)",
            "return_delay_min_max": "Return delay, max (min)",
            "provisional_flips": "Provisional verdicts that flipped",
            "provisional_pairs": "Provisional/final pairs",
            "provisional_lead_h_mean": "Provisional lead, mean (h)",
            "recovered": "Reference congested periods recovered",
            "reference_congested": "Reference congested periods",
            "spurious": "Spurious congested periods",
            "boundaries_within_30min": "Reference boundaries with a change point within 30 min",
            "reference_boundaries": "Reference boundaries",
        }
        for key, label in labels.items():
            if key in summary:
                value = summary[key]
                if value is None:
                    text = "-"
                elif isinstance(value, float):
                    text = f"{value:.1f}"
                else:
                    text = str(value)
                table.add_row(label, text)
        console.print(table)
        if output:
            console.print(f"\n✅ Events saved to [bold]{output}[/bold]")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1) from None


@app.command()
def version() -> None:
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


def _display_results(
    results: CongestionInferenceResult, analyzer: JitterbugAnalyzer, summary_only: bool = False
) -> None:
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
            detail_table.add_column("Start (UTC)", style="cyan")
            detail_table.add_column("End (UTC)", style="cyan")
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


def _display_validation_results(results: dict[str, Any], verbose: bool = False) -> None:
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


def main() -> None:
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

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
from ..io.readers import available_formats
from ..io.readers.ping import LINE_PARSERS, NO_TIMESTAMP_HINT, LineParser
from ..models import CongestionInferenceResult, JitterbugConfig, MultiSeriesResult

# matplotlib is an optional dependency; the visualize command checks for it at run time.
try:
    from ..visualization.plotter import MATPLOTLIB_AVAILABLE
except ImportError:  # pragma: no cover - only if the package itself is broken
    MATPLOTLIB_AVAILABLE = False


# Initialize Rich console
console = Console()
# `jitterbug stream` writes JSON lines to stdout, so its messages go to stderr.
err_console = Console(stderr=True)


INPUT_FORMAT_HELP = (
    f"Input file format ({', '.join(available_formats())}). Auto-detected from the "
    "extension, then the content, if not specified."
)

# Input mapping options shared by analyze, validate, visualize and replay; each overrides
# the `input` section of the configuration file when given.
TIME_COLUMN_OPTION = typer.Option(
    None, "--time-column", help="Name of the time column [default: epoch]"
)
RTT_COLUMN_OPTION = typer.Option(
    None,
    "--rtt-column",
    help="Name of the RTT column [default: first of values, rtt_value, rtt, latency]",
)
TIME_UNIT_OPTION = typer.Option(
    None, "--time-unit", help="Unit of the time column (s, ms, us, ns, iso, auto) [default: s]"
)
RTT_UNIT_OPTION = typer.Option(
    None, "--rtt-unit", help="Unit of the RTT column (ms, us, s, ns) [default: ms]"
)
GROUP_BY_OPTION = typer.Option(
    None,
    "--group-by",
    help="Analyze each series separately, keyed by these columns (repeatable or "
    "comma-separated; source and destination name the mapped columns)",
)
SERIES_OPTION = typer.Option(
    None,
    "--series",
    help="Keep only rows where COLUMN equals VALUE (COLUMN=VALUE, repeatable; source and "
    "destination name the mapped columns)",
)


def _parse_group_by(values: list[str] | None) -> list[str] | None:
    """``--group-by a,b --group-by c`` as ``["a", "b", "c"]``; ``None`` when not given."""
    if not values:
        return None
    return [name.strip() for value in values for name in value.split(",") if name.strip()]


def _parse_series(values: list[str] | None) -> dict[str, str] | None:
    """``--series name=value`` pairs as a mapping; ``None`` when not given."""
    if not values:
        return None
    filters: dict[str, str] = {}
    for value in values:
        name, sep, wanted = value.partition("=")
        if not sep or not name.strip():
            raise typer.BadParameter(
                f"expected COLUMN=VALUE, got {value!r}", param_hint="'--series'"
            )
        filters[name.strip()] = wanted
    return filters


def _apply_input_overrides(
    base: JitterbugConfig,
    *,
    time_column: str | None = None,
    rtt_column: str | None = None,
    time_unit: str | None = None,
    rtt_unit: str | None = None,
    group_by: list[str] | None = None,
    series: list[str] | None = None,
) -> JitterbugConfig:
    """
    Return a copy of ``base`` with the input mapping flags that were actually given.

    Values are re-validated by Pydantic, so an unknown unit is rejected with the same
    message as in a configuration file. ``--group-by`` replaces ``input.group_by``;
    each ``--series`` filter replaces the one for the same column in ``input.series``.
    """
    data = base.model_dump()
    given: dict[str, Any] = {
        "time_column": time_column,
        "rtt_column": rtt_column,
        "time_unit": time_unit,
        "rtt_unit": rtt_unit,
        "group_by": _parse_group_by(group_by),
    }
    data["input"].update({k: v for k, v in given.items() if v is not None})
    filters = _parse_series(series)
    if filters is not None:
        data["input"]["series"] = {**data["input"].get("series", {}), **filters}
    return JitterbugConfig.model_validate(data)


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
    method: str | None = None,
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
    if method is not None:
        data["jitter_analysis"]["method"] = method
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
        ...,
        help="Path to RTT data file (CSV, TSV, JSONL or scamper JSON; may be compressed)",
        exists=True,
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
        help=INPUT_FORMAT_HELP,
    ),
    time_column: str | None = TIME_COLUMN_OPTION,
    rtt_column: str | None = RTT_COLUMN_OPTION,
    time_unit: str | None = TIME_UNIT_OPTION,
    rtt_unit: str | None = RTT_UNIT_OPTION,
    group_by: list[str] | None = GROUP_BY_OPTION,
    series: list[str] | None = SERIES_OPTION,
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

    • One analysis per destination, or one destination only:
      [cyan]jitterbug analyze pings.json --group-by destination -o results.json[/cyan]
      [cyan]jitterbug analyze pings.json --series destination=8.8.8.8[/cyan]
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
        jitterbug_config = _apply_input_overrides(
            jitterbug_config,
            time_column=time_column,
            rtt_column=rtt_column,
            time_unit=time_unit,
            rtt_unit=rtt_unit,
            group_by=group_by,
            series=series,
        )

        # Create analyzer
        analyzer = JitterbugAnalyzer(jitterbug_config)

        if jitterbug_config.input.group_by:
            _analyze_grouped(analyzer, input_file, format, output)
            return

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

    except typer.BadParameter:
        raise
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1) from None


def _analyze_grouped(
    analyzer: JitterbugAnalyzer, input_file: Path, file_format: str | None, output: Path | None
) -> None:
    """``analyze --group-by``: one analysis per series, one summary row per series."""
    output_format = analyzer.config.output_format
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading and analyzing series...", total=None)
        results = analyzer.analyze_series_from_file(input_file, file_format)
        if output:
            progress.update(task, description="Saving results...")
            analyzer.save_series_results(results, output, output_format)
        progress.update(task, description="Analysis complete!", total=1, completed=1)

    _display_series_results(results, analyzer)
    if output:
        console.print(
            f"\n✅ Results of {len(results.series)} series saved to [bold]{output}[/bold]"
        )
    else:
        console.print(
            f"\n💡 [dim]Tip: To save full results, use --output flag:[/dim]\n"
            f"   [cyan]jitterbug analyze {input_file} --group-by "
            f"{','.join(analyzer.config.input.group_by)} --output results.{output_format}[/cyan]"
        )


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
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path (YAML or JSON); only `input` is used"
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help=INPUT_FORMAT_HELP,
    ),
    time_column: str | None = TIME_COLUMN_OPTION,
    rtt_column: str | None = RTT_COLUMN_OPTION,
    time_unit: str | None = TIME_UNIT_OPTION,
    rtt_unit: str | None = RTT_UNIT_OPTION,
    series: list[str] | None = SERIES_OPTION,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Validate RTT data file format and quality.

    [bold]Examples:[/bold]

    • Validate CSV file:
      [cyan]jitterbug validate rtts.csv[/cyan]

    • Validate with verbose output:
      [cyan]jitterbug validate rtts.csv --verbose[/cyan]

    • Millisecond timestamps in a column named ts:
      [cyan]jitterbug validate rtts.csv --time-column ts --time-unit ms[/cyan]

    • One destination of a file that holds several:
      [cyan]jitterbug validate pings.json --series destination=8.8.8.8[/cyan]
    """
    try:
        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()
        jitterbug_config = _apply_input_overrides(
            jitterbug_config,
            time_column=time_column,
            rtt_column=rtt_column,
            time_unit=time_unit,
            rtt_unit=rtt_unit,
            series=series,
        )
        data_loader = DataLoader(jitterbug_config.input)

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

    except typer.BadParameter:
        raise
    except Exception as e:
        console.print(f"❌ Validation failed: {e}", style="red")
        raise typer.Exit(1) from None


@app.command()
def inspect(
    input_file: Path = typer.Argument(..., help="Path to RTT data file to inspect", exists=True),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path (YAML or JSON); only `input` is used"
    ),
    format: str | None = typer.Option(None, "--format", "-f", help=INPUT_FORMAT_HELP),
    time_column: str | None = TIME_COLUMN_OPTION,
    rtt_column: str | None = RTT_COLUMN_OPTION,
    time_unit: str | None = TIME_UNIT_OPTION,
    rtt_unit: str | None = RTT_UNIT_OPTION,
    group_by: list[str] | None = GROUP_BY_OPTION,
    series: list[str] | None = SERIES_OPTION,
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON"),
) -> None:
    """
    Show what loading a file would do, without analyzing it.

    Reports the detected format and compression, the column mapping and units in effect
    (including what --time-unit auto inferred), how many rows each row policy would drop
    or reorder, and the series found: under --group-by, or the distinct source and
    destination pairs otherwise, each with its sample count, time range, median sampling
    interval, largest gap and duplicate timestamps. Exits with status 1 when loading
    would fail (the report says why).

    [bold]Examples:[/bold]

    • What is in this file?
      [cyan]jitterbug inspect pings.json[/cyan]

    • The series `analyze --group-by` would analyze, as JSON:
      [cyan]jitterbug inspect rtts.csv --group-by source,destination --json[/cyan]
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    try:
        from ..io.inspection import inspect_file

        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()
        jitterbug_config = _apply_input_overrides(
            jitterbug_config,
            time_column=time_column,
            rtt_column=rtt_column,
            time_unit=time_unit,
            rtt_unit=rtt_unit,
            group_by=group_by,
            series=series,
        )
        report = inspect_file(input_file, format, jitterbug_config.input)
    except typer.BadParameter:
        raise
    except Exception as e:
        err_console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1) from None

    if as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _display_inspection(report)
    if report["errors"]:
        raise typer.Exit(1)


@app.command()
def visualize(
    input_file: Path = typer.Argument(
        ...,
        help="Path to RTT data file (CSV, TSV, JSONL or scamper JSON; may be compressed)",
        exists=True,
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
        help=INPUT_FORMAT_HELP,
    ),
    time_column: str | None = TIME_COLUMN_OPTION,
    rtt_column: str | None = RTT_COLUMN_OPTION,
    time_unit: str | None = TIME_UNIT_OPTION,
    rtt_unit: str | None = RTT_UNIT_OPTION,
    series: list[str] | None = SERIES_OPTION,
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
        jitterbug_config = _apply_input_overrides(
            jitterbug_config,
            time_column=time_column,
            rtt_column=rtt_column,
            time_unit=time_unit,
            rtt_unit=rtt_unit,
            series=series,
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

    except typer.BadParameter:
        raise
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


STREAM_FORMATS = ("csv", "ping", "fping")
"""Line formats ``jitterbug stream --format`` reads."""


def _parse_probe_sample(
    line: str, parse: LineParser
) -> tuple[tuple[float, float] | None, bool, str | None]:
    """
    A sample from one ``ping -D`` / ``fping -D`` line.

    Returns the ``(epoch, rtt)`` of a timestamped reply (``None`` for any other line),
    whether the line was a reply without a timestamp, and the reply's destination.
    """
    probe = parse(line)
    if probe is None or probe.kind != "reply" or probe.rtt is None:
        return None, False, None
    if probe.epoch is None:
        return None, True, probe.destination
    return (probe.epoch, probe.rtt), False, probe.destination


@app.command()
def stream(
    source: str = typer.Argument(
        "-", help="File with one sample per line (see --format), or '-' for standard input"
    ),
    input_format: str = typer.Option(
        "csv",
        "--format",
        help="Line format: csv ('epoch,rtt'), ping (ping -D output) or fping (fping -D output)",
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Keep reading the file as it grows (like tail -f)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write events here as JSON lines instead of stdout"
    ),
    events: str = typer.Option(
        "all", "--events", help="Which events to emit: all, verdicts, change-points, none"
    ),
    state_changes: bool = typer.Option(
        False,
        "--state-changes",
        help="Also emit state changes of the path (suspected, confirmed, retracted)",
    ),
    on_state_change: str | None = typer.Option(
        None,
        "--on-state-change",
        help="Command run per state change with the event as JSON on stdin (no shell)",
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file (YAML or JSON)", exists=True
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Online back end: bocpd (incremental) or window (offline rerun)"
    ),
    method: str | None = typer.Option(
        None, "--method", "-m", help="Jitter method: jitter_dispersion (default) or ks_test"
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
    JSON object per event: change points, provisional verdicts and final verdicts. With
    --format ping or fping it reads the output of 'ping -D' or 'fping -D' instead: each
    reply with a [epoch] timestamp is a sample; replies without one are skipped (with one
    warning), as are headers, timeouts and summaries. Samples
    must arrive in time order; a sample older than the last one is dropped. The last bin is
    closed when the input ends (or on Ctrl-C with --follow), as 'jitterbug replay' does.

    --state-changes adds state-change events ("kind": "state_change") after the events of
    the sample that caused them; with --events none only those are printed.
    --on-state-change runs a command per state change in the background, the event as JSON
    on its standard input; a failing or slow command (killed after 60 s) is reported on
    standard error and does not stop the stream.

    [bold]Examples:[/bold]

    • From a probe writing to standard output:
      [cyan]my-probe | jitterbug stream[/cyan]

    • Following a file that another process appends to:
      [cyan]jitterbug stream rtts.csv --follow --events verdicts[/cyan]

    • Straight from ping (Linux iputils; -D prints the timestamps):
      [cyan]ping -D example.net | jitterbug stream --format ping[/cyan]

    • State changes handed to a script, nothing printed:
      [cyan]jitterbug stream rtts.csv -f --events none --on-state-change ./hook.sh[/cyan]
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    if events not in ("all", "verdicts", "change-points", "none"):
        err_console.print(
            f"❌ Error: --events must be all, verdicts, change-points or none, not {events}"
        )
        raise typer.Exit(2)
    if input_format not in STREAM_FORMATS:
        err_console.print(
            f"❌ Error: --format must be {', '.join(STREAM_FORMATS)}, not {input_format}"
        )
        raise typer.Exit(2)
    if follow and source == "-":
        err_console.print("--follow has no effect on standard input", style="yellow")
    try:
        from ..streaming import CommandHook, StateChangeTracker, create_online_analyzer

        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()
        jitterbug_config = _apply_streaming_overrides(
            jitterbug_config,
            backend=backend,
            method=method,
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
        wanted = {
            "all": None,
            "verdicts": "verdict",
            "change-points": "change_point",
            "none": "",
        }[events]
        tracker = StateChangeTracker(online) if state_changes or on_state_change else None
        hook = CommandHook(on_state_change) if on_state_change else None

        def emit(batch: list[Any]) -> None:
            nonlocal n_events
            for event in batch:
                if event.kind == "state_change":
                    if hook is not None:
                        hook(event)
                    if not state_changes:
                        continue
                elif wanted is not None and event.kind != wanted:
                    continue
                sink.write(json.dumps(event.to_dict()) + "\n")
                sink.flush()
                n_events += 1

        def feed(sample: tuple[float, float] | None) -> None:
            if hook is not None:
                hook.poll()  # reap finished commands, kill the ones past the timeout
            if tracker is None:
                emit(online.push(*sample) if sample else online.flush())
                return
            batch, changes = tracker.push(*sample) if sample else tracker.flush()
            emit(batch)
            emit(changes)

        parse_probe = LINE_PARSERS.get(input_format)
        warned_untimestamped = False
        first_destination: str | None = None
        warned_destinations = False

        try:
            try:
                for line in _iter_lines(path, follow):
                    if parse_probe is None:
                        sample = _parse_sample(line)
                    else:
                        sample, untimestamped, destination = _parse_probe_sample(line, parse_probe)
                        if sample is not None and destination is not None:
                            if first_destination is None:
                                first_destination = destination
                            elif destination != first_destination and not warned_destinations:
                                err_console.print(
                                    f"replies from {first_destination} and {destination} are "
                                    "pooled into one RTT series; stream one destination per "
                                    "jitterbug process",
                                    style="yellow",
                                )
                                warned_destinations = True
                        if untimestamped and not warned_untimestamped:
                            err_console.print(
                                f"{input_format} reply lines without a timestamp are "
                                f"skipped; {NO_TIMESTAMP_HINT[input_format]}",
                                style="yellow",
                            )
                            warned_untimestamped = True
                    if sample is None:
                        n_skipped += bool(line.strip())
                        continue
                    n_samples += 1
                    feed(sample)
            except KeyboardInterrupt:
                # Ctrl-C is how a --follow stream ends; close it like an EOF.
                err_console.print("interrupted", style="yellow")
            # The input ended (stdin EOF, a file without --follow, or Ctrl-C): close the
            # open bin so the result matches `jitterbug replay` on the same data.
            feed(None)
        finally:
            if hook is not None:
                hook.close()
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
    format: str | None = typer.Option(None, "--format", help=INPUT_FORMAT_HELP),
    time_column: str | None = TIME_COLUMN_OPTION,
    rtt_column: str | None = RTT_COLUMN_OPTION,
    time_unit: str | None = TIME_UNIT_OPTION,
    rtt_unit: str | None = RTT_UNIT_OPTION,
    series: list[str] | None = SERIES_OPTION,
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file (YAML or JSON)", exists=True
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Online back end: bocpd (incremental) or window (offline rerun)"
    ),
    method: str | None = typer.Option(
        None, "--method", "-m", help="Jitter method: jitter_dispersion (default) or ks_test"
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
      [cyan]jitterbug replay examples/network_analysis/data/raw.csv --method ks_test \\
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
        from ..streaming import events_to_json, replay_state_changes, score, score_state_changes

        jitterbug_config = JitterbugConfig.from_file(config) if config else JitterbugConfig()
        jitterbug_config = _apply_streaming_overrides(
            jitterbug_config,
            backend=backend,
            method=method,
            decision=decision,
            hazard_lambda=hazard_lambda,
            min_period_samples=min_period_samples,
            min_time_elapsed=min_time_elapsed,
            verbose=verbose,
        )
        jitterbug_config = _apply_input_overrides(
            jitterbug_config,
            time_column=time_column,
            rtt_column=rtt_column,
            time_unit=time_unit,
            rtt_unit=rtt_unit,
            series=series,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Replaying...", total=None)
            dataset = DataLoader(jitterbug_config.input).load_from_file(input_file, format)
            events, changes = replay_state_changes(dataset, jitterbug_config)
            progress.update(task, total=1, completed=1)
        summary = score(events, reference)
        changes_summary = score_state_changes(events, changes, reference)
        summary.update({k: v for k, v in changes_summary.items() if k not in summary})
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
            "state_changes": "State changes",
            "suspected": "Suspected state changes",
            "suspected_confirmed": "Suspected, then confirmed",
            "suspected_retracted": "Suspected, then retracted",
            "confirmed_without_suspicion": "Confirmed without suspicion",
            "corrected_by_final": "Corrected by a final verdict",
            "suspected_delay_min_median": "Suspicion delay, median (min)",
            "suspected_lead_min_median": "Suspicion lead over first verdict, median (min)",
            "suspected_lead_min_max": "Suspicion lead over first verdict, max (min)",
            "reference_congested_suspected": "Reference congested periods suspected",
            "suspected_onsets_outside_reference": "Suspected onsets outside the reference",
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
    except typer.BadParameter:
        raise
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


def _display_series_results(results: MultiSeriesResult, analyzer: JitterbugAnalyzer) -> None:
    """One summary row per series of a grouped analysis."""
    meta = results.metadata
    console.print(
        f"\n📊 [bold]Analysis Summary[/bold]: {meta['series_count']} series, "
        f"{meta['analyzed']} analyzed, {meta['skipped']} skipped"
    )
    table = Table()
    for name in results.group_by:
        table.add_column(name, style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Periods", justify="right")
    table.add_column("Congested", justify="right")
    table.add_column("Congestion Ratio", justify="right")
    table.add_column("Status")
    for entry in results.series:
        key = ["-" if entry.key.get(n) is None else str(entry.key[n]) for n in results.group_by]
        if entry.result is None:
            table.add_row(*key, "-", "-", "-", "-", f"[yellow]skipped: {entry.skipped_reason}")
            continue
        stats = analyzer.get_summary_statistics(entry.result)
        table.add_row(
            *key,
            str(entry.result.metadata.get("total_measurements", "-")),
            str(stats["total_periods"]),
            str(stats["congested_periods"]),
            f"{stats['congestion_ratio']:.2%}",
            "[green]analyzed",
        )
    console.print(table)


def _display_inspection(report: dict[str, Any]) -> None:
    """Human-readable form of an ``inspect_file`` report."""
    compression = report["compression"] or "none"
    detail = f" ({report['format_detail']})" if report.get("format_detail") else ""
    console.print(f"📄 [bold]{report['file']}[/bold]")
    console.print(f"   Format: {report['format']}{detail}, compression: {compression}")
    for key, value in report["reader"].items():
        console.print(f"   {key.replace('_', ' ').capitalize()}: {value!r}")
    console.print(f"   Columns: {', '.join(report['columns'])}")

    mapping = report.get("mapping")
    if mapping is not None:
        table = Table(title="Mapping in effect", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="bold")
        unit = mapping["time_unit_applied"]
        if mapping["time_unit"] == "auto":
            unit = f"{unit} (inferred by auto)"
        rtt_column = mapping["rtt_column"]
        if mapping["rtt_column_configured"] is None:
            rtt_column = f"{rtt_column} (first synonym found)"
        table.add_row("Time column", f"{mapping['time_column']} [{unit}]")
        table.add_row("RTT column", f"{rtt_column} [{mapping['rtt_unit']} -> ms]")
        table.add_row("Source column", mapping["source_column"] or "(absent)")
        table.add_row("Destination column", mapping["destination_column"] or "(absent)")
        if mapping["series"]:
            filters = ", ".join(f"{k}={v}" for k, v in mapping["series"].items())
            table.add_row("Series filter", filters)
        table.add_row("Group by", ", ".join(mapping["group_by"]) or "(none: one series)")
        if mapping["fixed_by_format"]:
            table.add_row("Fixed by format", ", ".join(mapping["fixed_by_format"]))
        console.print(table)

    rows = report.get("rows")
    if rows is not None:
        policies = report["policies"]
        table = Table(title="Rows and row policies")
        table.add_column("Step", style="cyan")
        table.add_column("Setting")
        table.add_column("Rows", justify="right")
        table.add_column("Effect")
        table.add_row("Read", "", str(rows["read"]), "")
        if rows["selected"] != rows["read"]:
            table.add_row("Series filter", "", str(rows["selected"]), "kept")
        table.add_row("Missing time or RTT", "", str(rows["missing"]), "dropped")
        out_of_range = policies["out_of_range"]
        table.add_row(
            "RTT out of range",
            out_of_range["setting"],
            str(out_of_range["rows"]),
            out_of_range["effect"],
        )
        unsorted = policies["unsorted"]
        table.add_row(
            "Out of time order",
            unsorted["setting"],
            f"{unsorted['rows']} in {unsorted['series_affected']} series",
            unsorted["effect"],
        )
        duplicates = policies["duplicates"]
        table.add_row(
            "Duplicate timestamps",
            duplicates["setting"],
            str(duplicates["rows"]),
            duplicates["effect"],
        )
        table.add_row("Valid", "", str(rows["valid"]), "")
        console.print(table)

    entries = report.get("series")
    if entries is not None:
        names = report["series_by"]
        title = (
            f"Series by {', '.join(names)}"
            if report["grouped"]
            else "Source/destination pairs (analyzed as one series)"
            if names
            else "Series (no source or destination column)"
        )
        table = Table(title=f"{title}: {len(entries)}")
        for name in names:
            table.add_column(name, style="cyan")
        for column in ("Samples", "Start (UTC)", "End (UTC)", "Median step", "Max gap", "Dup."):
            table.add_column(column, justify="right" if column != "Start (UTC)" else "left")

        def seconds(value: float | None) -> str:
            return "-" if value is None else f"{value:g}s"

        for entry in entries:
            key = ["-" if entry["key"].get(n) is None else str(entry["key"][n]) for n in names]
            table.add_row(
                *key,
                str(entry["samples"]),
                (entry["start"] or "-")[:19],
                (entry["end"] or "-")[:19],
                seconds(entry["median_interval_seconds"]),
                seconds(entry["max_gap_seconds"]),
                str(entry["duplicate_timestamps"]),
            )
        console.print(table)
        if not report["grouped"] and len(entries) > 1:
            console.print(
                "⚠️  Several pairs would be analyzed as one mixed series; separate them with "
                "--group-by source,destination or keep one with --series destination=VALUE",
                style="yellow",
            )

    for error in report["errors"]:
        console.print(f"❌ Loading would fail: {error}", style="red")
    if not report["errors"]:
        console.print("✅ Loading would succeed", style="green")


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

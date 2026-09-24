# Visualization

Jitterbug ships a small set of matplotlib figures for looking at an analysis: the raw
and minimum RTT series, the detected change points, the congestion periods, and summary
statistics. They are available from the CLI (`jitterbug visualize`) and from Python
(`JitterbugPlotter`).

## Installation

```bash
uv sync --extra visualization    # adds matplotlib
```

## CLI: `jitterbug visualize`

Runs the same analysis as `jitterbug analyze` and writes five PNG files (300 dpi):

```bash
jitterbug visualize examples/network_analysis/data/raw.csv --output-dir plots
```

```text
✅ 5 plots saved to plots
   • congestion_analysis: jitterbug_congestion_analysis.png
   • change_points: jitterbug_change_points.png
   • confidence_heatmap: jitterbug_confidence_heatmap.png
   • summary_stats: jitterbug_summary_stats.png
   • rtt_timeseries: jitterbug_rtt_timeseries.png

📊 Key Statistics:
   • Total periods: 22
   • Congested periods: 11
   • Congestion ratio: 24.4%
   • Change points: 24
```

Options:

| Option | Meaning |
| --- | --- |
| `--output-dir, -o` | Directory for the PNG files (default `visualization_output`) |
| `--prefix` | Filename prefix (default `jitterbug`) |
| `--algorithm, -a` | `ruptures` (default) or `bcp` |
| `--method, -m` | `jitter_dispersion` (default) or `ks_test` |
| `--threshold, -t` | Change point detection threshold |
| `--config, -c` | YAML/JSON configuration file, as for `analyze` |
| `--format, -f` | Input format (`csv`, `json`); auto-detected if omitted |

Example with the paper's configuration:

```bash
jitterbug visualize examples/network_analysis/data/raw.csv \
  --algorithm bcp --method ks_test --output-dir plots_bcp --prefix bcp
```

## The figures

| File | Content |
| --- | --- |
| `*_congestion_analysis.png` | Three stacked panels sharing the time axis: raw RTT, minimum RTT per interval, and the congestion verdict per period with congested spans shaded. The figure in the README is this one. |
| `*_change_points.png` | Minimum RTT series with a vertical line per change point, colored by confidence. |
| `*_confidence_heatmap.png` | Confidence of each period's verdict over time. |
| `*_summary_stats.png` | Counts, congestion ratio, confidence and duration distributions. |
| `*_rtt_timeseries.png` | Raw and minimum RTT series overlaid. |

Time axes pick their own tick spacing from the span of the data (hours to weeks).

## Python: `JitterbugPlotter`

```python
from pathlib import Path

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.visualization import JitterbugPlotter

analyzer = JitterbugAnalyzer(JitterbugConfig())
results = analyzer.analyze_from_file("examples/network_analysis/data/raw.csv")

plotter = JitterbugPlotter()  # JitterbugPlotter(style="ggplot", figsize=(14, 8)) also works

# Everything the CLI writes, in one call
paths = plotter.save_all_plots(
    raw_data=analyzer.raw_data,
    min_rtt_data=analyzer.min_rtt_data,
    results=results,
    change_points=analyzer.change_points or [],
    output_dir=Path("plots"),
    prefix="run1",
)

# Or one figure at a time; each method returns the matplotlib Figure
fig = plotter.plot_congestion_analysis(
    analyzer.raw_data, analyzer.min_rtt_data, results, title="Path A, December 2017"
)
fig.savefig("path_a.png", dpi=300, bbox_inches="tight")
```

Methods, all returning a `matplotlib.figure.Figure` and accepting `title` and `save_path`:

| Method | Inputs |
| --- | --- |
| `plot_rtt_timeseries(datasets)` | `{"label": RTTDataset, ...}`; `show_points=True` draws markers |
| `plot_congestion_analysis(raw_data, min_rtt_data, results)` | the three-panel figure |
| `plot_change_points(dataset, change_points)` | `MinimumRTTDataset` and `list[ChangePoint]` |
| `plot_confidence_heatmap(results)` | `CongestionInferenceResult` |
| `plot_summary_statistics(results)` | `CongestionInferenceResult` |
| `plot_algorithm_comparison(dataset, algorithm_results)` | `{"ruptures": [...], "bcp": [...]}` change points per detector |

`analyzer.raw_data`, `analyzer.min_rtt_data` and `analyzer.change_points` are populated by
the last `analyze*` call.

### In notebooks

The figures are ordinary matplotlib objects, so `%matplotlib inline` shows them. The two
notebooks in `examples/` (`jitter_dispersion_analysis.ipynb`,
`kolmogorov_smirnov_analysis.ipynb`) build their own figures with matplotlib directly,
which is a good starting point for custom layouts.

### Headless use

Set `MPLBACKEND=Agg` (or `matplotlib.use("Agg")` before importing the plotter) on servers
and in tests. The test suite does exactly this in `tests/conftest.py`.

## Comparing detectors

`tools/generate_visualizations.py` runs both detectors on the bundled dataset, writes
one congestion analysis figure per detector plus a comparison chart, and regenerates
`examples/network_analysis/plots/README.md`. It needs the `bcp` extra and takes a few
seconds on CPU (see `bcp_device` in the configuration).

## Troubleshooting

- `matplotlib not found`: `uv sync --extra visualization`.
- Blank or tiny figures in a notebook: call the plotting method after `%matplotlib inline`
  and do not call `plt.close()` before display.
- Slow on very long series: the plots draw every raw sample; downsample the raw data
  before plotting or plot only `min_rtt_data`.

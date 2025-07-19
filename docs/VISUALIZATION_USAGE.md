# Jitterbug Visualization Usage Guide

This guide covers all visualization capabilities in Jitterbug, including static plots, interactive dashboards, and programmatic usage.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Visualization Commands](#cli-visualization-commands)
4. [Visualization Types](#visualization-types)
5. [Output Formats](#output-formats)
6. [Programmatic Usage](#programmatic-usage)
7. [Advanced Options](#advanced-options)
8. [Troubleshooting](#troubleshooting)

## Installation

### Using uv (Recommended)

```bash
# Install visualization dependencies
uv pip install matplotlib plotly

# Or install all visualization dependencies with extras
uv pip install -e ".[visualization]"
```

### Using pip

```bash
# Install visualization dependencies
pip install matplotlib plotly

# Or with extras
pip install jitterbug[visualization]
```

## Quick Start

Generate a comprehensive visualization report with a single command:

```bash
# Basic visualization
jitterbug visualize examples/network_analysis/data/raw.csv

# Custom output directory
jitterbug visualize examples/network_analysis/data/raw.csv --output-dir my_report

# With specific algorithm
jitterbug visualize examples/network_analysis/data/raw.csv --algorithm bcp --output-dir bcp_report
```

## CLI Visualization Commands

### Basic Syntax

```bash
jitterbug visualize <input_file> [OPTIONS]
```

### Available Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-dir` | `-o` | Output directory for visualization files | `visualization_output` |
| `--config` | `-c` | Configuration file path (YAML or JSON) | None |
| `--format` | `-f` | Input file format (csv, json, influx) | Auto-detected |
| `--method` | `-m` | Jitter analysis method | `jitter_dispersion` |
| `--algorithm` | `-a` | Change point detection algorithm | `ruptures` |
| `--threshold` | `-t` | Change point detection threshold | `0.25` |
| `--title` | | Report title | Based on filename |
| `--static-only` | | Generate only static plots (PNG) | False |
| `--interactive-only` | | Generate only interactive plots (HTML) | False |
| `--verbose` | `-v` | Enable verbose logging | False |

### Examples

#### 1. Basic Visualization Report

```bash
# Generate full report with both static and interactive plots
jitterbug visualize examples/network_analysis/data/raw.csv
```

This creates:
```
visualization_output/
├── index.html              # Main report page
├── static/                 # Static PNG plots
│   ├── congestion_analysis.png
│   ├── change_points.png
│   └── summary_stats.png
└── interactive/            # Interactive HTML plots
    ├── timeline.html
    ├── dashboard.html
    └── confidence_scatter.html
```

#### 2. Static Plots Only

```bash
# Generate only PNG images (no HTML)
jitterbug visualize examples/network_analysis/data/raw.csv \
  --static-only \
  --output-dir static_plots
```

Output:
```
static_plots/
├── congestion_analysis.png
├── change_points.png
├── summary_stats.png
└── algorithm_comparison.png
```

#### 3. Interactive Plots Only

```bash
# Generate only interactive HTML visualizations
jitterbug visualize examples/network_analysis/data/raw.csv \
  --interactive-only \
  --output-dir interactive_plots
```

Output:
```
interactive_plots/
├── timeline.html         # Zoomable timeline
├── dashboard.html        # Multi-panel dashboard
└── confidence_scatter.html
```

#### 4. Custom Algorithm and Title

```bash
# Use Bayesian algorithm with custom title
jitterbug visualize examples/network_analysis/data/raw.csv \
  --algorithm bcp \
  --title "Network Congestion Analysis - Q4 2024" \
  --output-dir quarterly_report
```

#### 5. Different Change Point Detection Algorithms

```bash
# Use PyTorch neural network algorithm
jitterbug visualize examples/network_analysis/data/raw.csv \
  --algorithm torch \
  --title "PyTorch Deep Learning Analysis" \
  --output-dir pytorch_analysis

# Use Rbeast seasonal pattern detection
jitterbug visualize examples/network_analysis/data/raw.csv \
  --algorithm rbeast \
  --title "Seasonal Pattern Analysis" \
  --output-dir rbeast_analysis

# Use ADTK anomaly detection
jitterbug visualize examples/network_analysis/data/raw.csv \
  --algorithm adtk \
  --title "Anomaly Detection Analysis" \
  --output-dir adtk_analysis

# Use Ruptures (default, fast)
jitterbug visualize examples/network_analysis/data/raw.csv \
  --algorithm ruptures \
  --title "Fast Change Point Detection" \
  --output-dir ruptures_analysis
```

#### 6. With Configuration File

```bash
# Use custom configuration
jitterbug visualize examples/network_analysis/data/raw.csv \
  --config config.yaml \
  --output-dir configured_report
```

## Visualization Types

### 1. RTT Time Series Plot
- **File**: `congestion_analysis.png` / `timeline.html`
- **Shows**: Raw RTT measurements with congestion periods highlighted
- **Features**:
  - Color-coded congestion periods
  - Minimum RTT overlay
  - Interactive zoom/pan (HTML version)

### 2. Change Point Detection Visualization
- **File**: `change_points.png` / Part of `dashboard.html`
- **Shows**: Detected change points with confidence scores
- **Features**:
  - Change point markers
  - Confidence score heatmap
  - Algorithm comparison (if multiple algorithms used)

### 3. Summary Statistics
- **File**: `summary_stats.png` / Part of `dashboard.html`
- **Shows**: Statistical summaries and distributions
- **Features**:
  - RTT distribution histogram
  - Congestion duration pie chart
  - Key metrics table

### 4. Interactive Dashboard
- **File**: `dashboard.html`
- **Shows**: Multi-panel view with all analyses
- **Features**:
  - Synchronized time axes
  - Hover tooltips
  - Downloadable data

### 5. Algorithm Comparison
- **File**: `algorithm_comparison.png` (when multiple algorithms tested)
- **Shows**: Side-by-side comparison of different algorithms
- **Features**:
  - Precision/recall metrics
  - Change point alignment
  - Performance statistics

## Output Formats

### HTML Report Structure

The main `index.html` provides:
- Navigation menu
- Embedded static images
- Links to interactive plots
- Summary statistics
- Analysis configuration details

### Static Plot Formats
- **PNG**: Default format, high resolution (300 DPI)
- **PDF**: Available programmatically (vector format)
- **SVG**: Available programmatically (scalable)

### Interactive Plot Features
- **Zoom**: Click and drag to zoom
- **Pan**: Shift + click to pan
- **Reset**: Double-click to reset view
- **Export**: Download as PNG/SVG
- **Data**: Hover for detailed information

## Programmatic Usage

### Basic Visualization

```python
from jitterbug import JitterbugAnalyzer
from jitterbug.visualization import JitterbugDashboard

# Analyze data
analyzer = JitterbugAnalyzer()
results = analyzer.analyze_from_file('rtts.csv')

# Create visualizations
dashboard = JitterbugDashboard()
report = dashboard.create_comprehensive_report(
    raw_data=analyzer.raw_data,
    min_rtt_data=analyzer.min_rtt_data,
    results=results,
    change_points=analyzer.change_points,
    output_dir="./my_report",
    title="My Analysis"
)
```

### Custom Static Plots

```python
from jitterbug.visualization import StaticVisualizer

# Create static visualizer
visualizer = StaticVisualizer()

# Generate individual plots
fig = visualizer.plot_congestion_analysis(
    raw_data, min_rtt_data, results
)
fig.savefig("congestion.png", dpi=300, bbox_inches='tight')

# Change point visualization
fig = visualizer.plot_change_points(
    min_rtt_data, change_points
)
fig.savefig("change_points.png")
```

### Custom Interactive Plots

```python
from jitterbug.visualization import InteractiveVisualizer

# Create interactive visualizer
visualizer = InteractiveVisualizer()

# Create interactive timeline
fig = visualizer.create_interactive_timeline(
    raw_data, min_rtt_data, results, change_points,
    title="Network Timeline"
)

# Save as HTML
visualizer.save_html(fig, "timeline.html")

# Or show in Jupyter notebook
fig.show()
```

### Jupyter Notebook Usage

```python
# Enable inline plotting
%matplotlib inline

from jitterbug import JitterbugAnalyzer
from jitterbug.visualization import JitterbugDashboard

# Analyze and visualize
analyzer = JitterbugAnalyzer()
results = analyzer.analyze_from_file('rtts.csv')

# Create dashboard
dashboard = JitterbugDashboard()

# Show static plot inline
fig = dashboard.static.plot_congestion_analysis(
    analyzer.raw_data, 
    analyzer.min_rtt_data, 
    results
)
plt.show()

# Show interactive plot inline
interactive_fig = dashboard.interactive.create_interactive_timeline(
    analyzer.raw_data,
    analyzer.min_rtt_data,
    results,
    analyzer.change_points
)
interactive_fig.show()
```

## Advanced Options

### 1. Custom Color Schemes

```python
# Use custom colors for congestion periods
dashboard = JitterbugDashboard(
    congestion_color="#FF6B6B",
    normal_color="#4ECDC4"
)
```

### 2. Multiple Algorithm Comparison

```bash
# Compare different algorithms
for algo in ruptures bcp torch; do
    jitterbug visualize data.csv \
      --algorithm $algo \
      --output-dir comparison/$algo
done
```

### 3. Batch Processing

```bash
# Process multiple files
for file in data/*.csv; do
    jitterbug visualize "$file" \
      --output-dir "reports/$(basename $file .csv)"
done
```

### 4. High-Resolution Export

```python
# Export high-resolution static plots
fig = dashboard.static.plot_congestion_analysis(
    raw_data, min_rtt_data, results
)
fig.savefig("high_res.png", dpi=600, bbox_inches='tight')
```

### 5. Custom Time Ranges

```python
# Focus on specific time range
from datetime import datetime

start_time = datetime(2024, 1, 1)
end_time = datetime(2024, 1, 7)

# Filter data
filtered_data = raw_data[
    (raw_data.epoch >= start_time.timestamp()) & 
    (raw_data.epoch <= end_time.timestamp())
]

# Visualize filtered data
dashboard.create_comprehensive_report(
    raw_data=filtered_data,
    # ... other parameters
)
```

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'matplotlib'

```bash
# Install visualization dependencies
uv pip install matplotlib plotly

# Or with pip
pip install jitterbug[visualization]
```

#### 2. Large Datasets (Slow Performance)

For datasets with >100k points:

```bash
# Use static-only mode (faster)
jitterbug visualize large_data.csv --static-only

# Or downsample in Python
import pandas as pd
df = pd.read_csv('large_data.csv')
df_sampled = df.sample(n=50000)  # Sample 50k points
df_sampled.to_csv('sampled_data.csv', index=False)
```

#### 3. Interactive Plots Not Opening

```bash
# Check if browser is set
echo $BROWSER

# Set default browser (Linux/Mac)
export BROWSER=firefox

# Or open manually
cd visualization_output
python -m http.server 8000
# Open http://localhost:8000 in browser
```

#### 4. Memory Issues

```python
# Process in chunks for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # Process chunk
    results = analyzer.analyze_from_dataframe(chunk)
    # Visualize chunk
```

### Debugging Visualization Issues

```bash
# Enable verbose mode
jitterbug visualize data.csv --verbose

# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Test basic plotting
python -c "import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.savefig('test.png')"
```

### Performance Tips

1. **Use appropriate format**:
   - Static for reports/documents
   - Interactive for exploration

2. **Optimize file sizes**:
   - PNG: Good compression, suitable for web
   - PDF: Vector format, good for printing
   - HTML: Can be large with many data points

3. **Consider data size**:
   - <10k points: Use interactive
   - 10k-100k: Both work well
   - >100k: Consider static or sampling

## Examples Gallery

### Example 1: Basic Network Analysis

```bash
# Analyze network RTT data
jitterbug visualize examples/network_analysis/data/raw.csv \
  --title "Network Performance Analysis" \
  --output-dir network_report
```

### Example 2: Multi-Algorithm Comparison

```bash
# Compare all algorithms
for algo in ruptures bcp torch; do
    echo "Processing with $algo..."
    jitterbug visualize examples/network_analysis/data/raw.csv \
      --algorithm $algo \
      --title "$algo Algorithm Analysis" \
      --output-dir "comparison/$algo"
done

# Create comparison summary
echo "Reports generated in comparison/ directory"
```

### Example 3: Production Monitoring Dashboard

```bash
# Generate daily reports
DATE=$(date +%Y-%m-%d)
jitterbug visualize /var/log/rtt_data_$DATE.csv \
  --title "Daily Network Report - $DATE" \
  --output-dir "/var/www/reports/$DATE" \
  --static-only  # For web serving
```

### Example 4: Research Paper Figures

```python
# Generate publication-quality figures
from jitterbug import JitterbugAnalyzer
from jitterbug.visualization import StaticVisualizer
import matplotlib.pyplot as plt

# Set publication style
plt.style.use('seaborn-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Analyze data
analyzer = JitterbugAnalyzer()
results = analyzer.analyze_from_file('experiment_data.csv')

# Create visualizer
viz = StaticVisualizer()

# Generate figure
fig = viz.plot_congestion_analysis(
    analyzer.raw_data,
    analyzer.min_rtt_data,
    results
)

# Save for publication
fig.savefig('figure_1.pdf', format='pdf', bbox_inches='tight')
```

## See Also

- [ALGORITHM_USAGE.md](ALGORITHM_USAGE.md) - Detailed algorithm usage
- [API.md](API.md) - REST API documentation
- [Examples README](../examples/README.md) - More example workflows
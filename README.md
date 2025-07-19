# Jitterbug 2.0: Framework for Jitter-Based Congestion Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Jitterbug 2.0** is a modern, completely rewritten Python framework for detecting network congestion through jitter analysis and change point detection in Round-Trip Time (RTT) measurements.

This framework is the result of research presented in the paper *"Jitterbug: A new framework for jitter-based congestion inference"*, published in the Proceedings of the Passive and Active Measurement Conference (PAM) 2022.

## 📚 Table of Contents

- [What's New in Version 2.0](#-whats-new-in-version-20)
- [Installation](#-installation)
  - [Using pip](#using-pip-recommended)
  - [Using uv](#using-uv-fast-python-package-manager)
  - [From Source](#from-source)
  - [Using Docker](#using-docker-containerized)
- [Quick Start](#-quick-start)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [REST API Server (Optional)](#-rest-api-server-optional)
- [Visualization (Optional)](#-visualization-optional)
- [Website (Future Feature)](#-website-future-feature)
- [Input Data Formats](#-input-data-formats)
- [Configuration](#%EF%B8%8F-configuration)
- [Analysis Pipeline](#-analysis-pipeline)
- [Algorithms](#-algorithms)
- [Output Formats](#-output-formats)
- [Development](#%EF%B8%8F-development)
- [Examples](#-examples)
- [Research & Citations](#-research--citations)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)

## 🚀 What's New in Version 2.0

- **Complete Rewrite**: Modern Python architecture with type safety
- **Pydantic Models**: Robust data validation and serialization
- **Multiple Algorithms**: Support for 5 change point detection algorithms (BCP, Ruptures, PyTorch, Rbeast, ADTK)
- **Flexible Input Formats**: CSV, JSON (scamper), and InfluxDB support
- **Rich CLI Interface**: Beautiful command-line interface with progress bars and tables
- **Configuration Management**: YAML/JSON configuration files with validation
- **Better Performance**: Optimized algorithms and memory usage
- **Comprehensive Documentation**: Type hints, docstrings, and examples

## 📦 Installation

### Using uv (Recommended)

⚠️ **Note**: The PyPI version may be outdated. For the latest features and algorithms, install from source (see below).

```bash
# Install with uv (faster)
uv pip install jitterbug

# Or with traditional pip
pip install jitterbug
```

### Using uv (Fast Python Package Manager)

First install uv if you haven't already:

```bash
# Install uv (recommended for faster dependency resolution)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with homebrew (macOS)
brew install uv
```

Then install Jitterbug:

```bash
# Create a new virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### From Source (Development)

```bash
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug

# Create a new virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with core dependencies
uv pip install -e .

# Install optional dependencies
uv pip install -e ".[bayesian]"  # For Bayesian algorithm
uv pip install -e ".[torch]"      # For PyTorch algorithm
uv pip install -e ".[visualization]"  # For visualization
uv pip install -e ".[all]"        # For all optional dependencies

# OR install the bayesian dependency directly:
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git
```

### Optional Dependencies

```bash
# For PyTorch change point detection
uv pip install jitterbug[torch]

# For Bayesian change point detection
uv pip install jitterbug[bayesian]
# OR install directly from GitHub:
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git

# For InfluxDB support
uv pip install jitterbug[influx]

# For visualization (matplotlib, plotly)
uv pip install jitterbug[visualization]

# For REST API server (fastapi, uvicorn)
uv pip install jitterbug[api]

# For Jupyter notebooks
uv pip install jitterbug[jupyter]

# Install everything
uv pip install jitterbug[all]
```

### Using Docker (Containerized)

Docker provides an easy way to run Jitterbug without installing Python dependencies:

```bash
# Quick start with Docker Compose
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug
docker-compose up -d

# The API server will be available at http://localhost:8000
curl http://localhost:8000/api/v1/health
```

**Docker Usage Examples:**

```bash
# Run CLI analysis
docker-compose run --rm jitterbug-cli analyze /app/examples/network_analysis/data/raw.csv

# Run visualization
docker-compose run --rm jitterbug-cli visualize /app/examples/network_analysis/data/raw.csv --output-dir /app/output

# Run validation
docker-compose run --rm jitterbug-cli validate /app/examples/network_analysis/data/raw.csv

# Start API server only
docker-compose up jitterbug-api
```

**Using Docker directly:**

```bash
# Build the image
docker build -t jitterbug:latest .

# Run CLI commands
docker run --rm -v $(pwd)/examples:/app/examples jitterbug:latest cli analyze /app/examples/network_analysis/data/raw.csv

# Run API server
docker run -d -p 8000:8000 --name jitterbug-api jitterbug:latest

# Interactive shell
docker run -it --rm jitterbug:latest bash
```

**Data Processing with Docker:**

```bash
# Create data directory
mkdir -p data output

# Use the example data (or copy your own RTT data)
# cp your_rtt_data.csv examples/network_analysis/data/

# Start services
docker-compose up -d

# Process data
docker-compose run --rm jitterbug-cli analyze /app/examples/network_analysis/data/raw.csv --output /app/output/results.json

# Generate visualizations
docker-compose run --rm jitterbug-cli visualize /app/examples/network_analysis/data/raw.csv --output-dir /app/output/plots

# Check results
ls output/
```

See the [Docker documentation](docs/DOCKER.md) for detailed deployment instructions.

## 🔧 Quick Start

### Command Line Interface

```bash
# Basic analysis (shows summary on screen)
jitterbug analyze examples/network_analysis/data/raw.csv

# Save results to file
jitterbug analyze examples/network_analysis/data/raw.csv --output results.json

# With custom configuration
jitterbug analyze examples/network_analysis/data/raw.csv --config config.yaml --output results.json

# Using different algorithms
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures --method ks_test

# Show only summary statistics
jitterbug analyze examples/network_analysis/data/raw.csv --summary-only

# Generate configuration template
jitterbug config --template --output config.yaml

# Validate data quality
jitterbug validate examples/network_analysis/data/raw.csv --verbose
```

### Python API

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig

# Load configuration
config = JitterbugConfig()

# Create analyzer
analyzer = JitterbugAnalyzer(config)

# Analyze RTT data
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')

# Get congestion periods
congested_periods = results.get_congested_periods()

# Display summary
summary = analyzer.get_summary_statistics(results)
print(f"Found {len(congested_periods)} congestion periods")
print(f"Total congestion duration: {summary['congestion_duration_seconds']:.1f}s")
```

## 🌐 REST API Server (Optional)

For integration with other systems or web applications, Jitterbug provides an optional REST API server:

### Starting the API Server

```bash
# Install API dependencies
uv pip install jitterbug[api]  # or uv pip install fastapi uvicorn

# Start the server (using Python module)
python -m jitterbug.api.server --host 0.0.0.0 --port 8000

# Or with default settings
python -m jitterbug.api.server
```

### API Documentation

Once the server is running, you can access:
- **API Documentation**: http://localhost:8000/docs
- **API Explorer**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

### Using the API

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Analyze data via API
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "measurements": [
        {
          "timestamp": "2024-01-01T10:00:00Z",
          "epoch": 1704110400.0,
          "rtt_value": 25.6,
          "source": "192.168.1.1",
          "destination": "8.8.8.8"
        }
      ]
    },
    "algorithm": "ruptures",
    "method": "jitter_dispersion"
  }'
```

### Python API Client

```python
import requests

# Analyze data programmatically
response = requests.post('http://localhost:8000/api/v1/analyze', json={
    "data": {"measurements": [...]},
    "algorithm": "ruptures"
})

if response.status_code == 200:
    results = response.json()
    print(f"Analysis completed in {results['execution_time']:.2f}s")
```

## 📊 Visualization (Optional)

Jitterbug provides comprehensive visualization capabilities for analyzing network congestion patterns:

### Installing Visualization Dependencies

```bash
# Install visualization dependencies
uv pip install jitterbug[visualization]  # or uv pip install matplotlib plotly
```

### Generating Visualizations

```bash
# Basic visualization report
jitterbug visualize examples/network_analysis/data/raw.csv

# Custom output directory
jitterbug visualize examples/network_analysis/data/raw.csv --output-dir my_plots

# Static plots only (PNG/PDF)
jitterbug visualize examples/network_analysis/data/raw.csv --static-only

# Interactive plots only (HTML)
jitterbug visualize examples/network_analysis/data/raw.csv --interactive-only

# Custom title and algorithm
jitterbug visualize examples/network_analysis/data/raw.csv --title "Network Analysis Report" --algorithm bcp
```

### Visualization Output

The visualization command generates:

**Static Plots** (PNG format):
- RTT time series with congestion periods highlighted
- Change point detection visualization
- Confidence score heatmaps
- Summary statistics and distributions

**Interactive Plots** (HTML format):
- Interactive timeline with zoom/pan capabilities
- Comprehensive dashboard with multiple views
- Algorithm comparison charts
- Hover tooltips with detailed information

### Example Visualization Usage

```bash
# Generate comprehensive report
jitterbug visualize examples/network_analysis/data/raw.csv \
  --title "Network Congestion Analysis" \
  --output-dir network_analysis_report

# This creates:
# network_analysis_report/
# ├── index.html              # Main report page
# ├── static/                 # Static PNG plots
# │   ├── congestion_analysis.png
# │   ├── change_points.png
# │   └── summary_stats.png
# └── interactive/            # Interactive HTML plots
#     ├── timeline.html
#     ├── dashboard.html
#     └── confidence_scatter.html
```

### Programmatic Visualization

```python
from jitterbug.visualization import JitterbugDashboard
from jitterbug import JitterbugAnalyzer

# Analyze data
analyzer = JitterbugAnalyzer()
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')

# Create visualizations
dashboard = JitterbugDashboard()
report = dashboard.create_comprehensive_report(
    raw_data=analyzer.raw_data,
    min_rtt_data=analyzer.min_rtt_data,
    results=results,
    change_points=analyzer.change_points,
    output_dir="./analysis_report",
    title="My Network Analysis"
)

print(f"Report generated: {report['output_dir']}/index.html")
```

## 🌍 Website (Future Feature)

**Note**: The website feature is planned for future development and is not yet implemented.

**Planned Purpose**: The website will provide:
- **Web-based Interface**: Upload RTT data files through a web browser
- **Real-time Analysis**: Interactive analysis without installing Python
- **Visualization Gallery**: Browse and share network analysis results
- **Educational Resources**: Tutorials and documentation about network congestion analysis
- **Community Features**: Share datasets and analysis results with researchers

**Current Status**: This feature is in the planning phase. For now, users can:
- Use the CLI for local analysis
- Use the REST API for programmatic access
- Use the visualization tools for creating reports
- Deploy the API server for web integration

## 📊 Input Data Formats

### CSV Format

```csv
epoch,values
1512144010.0,63.86
1512144010.0,66.52
1512144020.0,85.2
1512144110.0,50.79
```

*Example data available in `examples/network_analysis/data/raw.csv` (47,164 measurements)*

### JSON Format (Scamper)

```json
{"type":"ping", "src":"192.168.1.1", "dst":"8.8.8.8", "responses":[{"rtt":1.712, "tx":{"sec":1752855461, "usec":719258}}]}
```

### InfluxDB Query Results

Direct integration with InfluxDB for real-time analysis:

```python
from jitterbug.io import DataLoader

loader = DataLoader()
dataset = loader.load_from_influxdb(
    url="http://localhost:8086",
    token="your-token",
    org="your-org",
    bucket="network-metrics",
    query='from(bucket:"network-metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "rtt")'
)
```

## ⚙️ Configuration

Create a configuration file to customize analysis parameters:

```yaml
# config.yaml
change_point_detection:
  algorithm: "ruptures"  # or "bcp", "torch"
  threshold: 0.25
  min_time_elapsed: 1800  # seconds
  ruptures_model: "rbf"
  ruptures_penalty: 10.0

jitter_analysis:
  method: "jitter_dispersion"  # or "ks_test"
  threshold: 0.25
  moving_average_order: 6
  moving_iqr_order: 4
  significance_level: 0.05

latency_jump:
  threshold: 0.5

data_processing:
  minimum_interval_minutes: 15
  outlier_detection: true
  outlier_threshold: 3.0

output_format: "json"  # or "csv", "parquet"
verbose: false
```

## 🧪 Analysis Pipeline

Jitterbug follows a sophisticated analysis pipeline:

1. **Data Loading & Validation**: Load RTT measurements with quality checks
2. **Minimum RTT Computation**: Aggregate data into time intervals
3. **Change Point Detection**: Identify significant changes in RTT patterns
4. **Latency Jump Analysis**: Detect baseline latency increases
5. **Jitter Analysis**: Analyze jitter dispersion or distribution changes
6. **Congestion Inference**: Combine evidence to infer congestion periods

## 🔍 Algorithms

Jitterbug v2.0 supports multiple change point detection algorithms with proven performance on real network data.

### Change Point Detection Algorithms

| Algorithm | Performance | Rating | Description | Use Case |
|-----------|-------------|---------|-------------|----------|
| **BCP (Bayesian)** | 14/15 (93.3%) | ⭐⭐⭐⭐⭐ | Classical Bayesian approach with statistical rigor | Gold standard, research applications |
| **PyTorch Neural** | 14/15 (93.3%) | ⭐⭐⭐⭐⭐ | Deep learning-based pattern recognition | Complex patterns, advanced analysis |
| **Ruptures** | 11/15 (73.3%) | ⭐⭐⭐⭐ | Fast and reliable using multiple models | Production environments, quick analysis |
| **Rbeast** | 10/15 (66.7%) | ⭐⭐⭐ | Seasonal pattern detection and trends | Time series with seasonal components |
| **ADTK** | 9/15 (60.0%) | ⭐⭐ | Anomaly detection with level shifts | Basic anomaly detection, simple patterns |

*Performance tested against 47,164 RTT measurements with 15 expected congestion periods*

### Algorithm Selection Guide

**Choose BCP when:**
- You need the highest accuracy (93.3%)
- Statistical rigor is important
- Research or academic applications
- Uncertainty quantification required

**Choose PyTorch when:**
- Complex pattern recognition needed (93.3% accuracy)
- Advanced machine learning capabilities desired
- Non-standard network behavior expected

**Choose Ruptures when:**
- Fast processing is priority (10-20s runtime)
- Good balance of speed and accuracy (73.3%)
- Production environments with time constraints

**Choose Rbeast when:**
- Seasonal patterns are present in data
- Trend analysis is important
- Moderate accuracy acceptable (66.7%)

**Choose ADTK when:**
- Simple anomaly detection sufficient
- Minimal computational resources
- Basic pattern detection acceptable (60.0%)

### Jitter Analysis Methods

- **Jitter Dispersion**: Analyzes changes in jitter variability using moving IQR and averaging
- **Kolmogorov-Smirnov Test**: Statistical test for distribution changes between periods

### Example Results (BCP + KS-Test)

Using the gold-standard BCP algorithm with KS-test jitter analysis on example dataset:

```
📊 Analysis Summary
┌─────────────────────┬────────────┐
│ Total Periods       │ 34         │
│ Congested Periods   │ 14         │
│ Congestion Ratio    │ 41.18%     │
│ Average Confidence  │ 0.90       │
│ Detection Accuracy  │ 93.3%      │
└─────────────────────┴────────────┘
```

*Complete visualization examples available in `examples/network_analysis/plots/`*

![BCP + KS-Test](/examples/network_analysis/plots/bcp_congestion_analysis.png)

## 📈 Output Formats

### JSON Output

```json
{
  "inferences": [
    {
      "start_timestamp": "2024-01-01T10:00:00",
      "end_timestamp": "2024-01-01T10:15:00",
      "is_congested": true,
      "confidence": 0.85,
      "latency_jump": {
        "has_jump": true,
        "magnitude": 12.5
      },
      "jitter_analysis": {
        "has_significant_jitter": true,
        "method": "jitter_dispersion"
      }
    }
  ],
  "metadata": {
    "total_measurements": 1000,
    "change_points": 5,
    "congestion_periods": 2
  }
}
```

### CSV Output

```csv
starts,ends,congestion,confidence,has_latency_jump,has_jitter_change
1641024000,1641024900,true,0.85,true,true
1641024900,1641025800,false,0.0,false,false
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/jitterbug/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=jitterbug

# Run specific test file
pytest tests/test_analyzer.py
```

## 📚 Examples

**🚀 Ready to Try?** Use the comprehensive example dataset with 47,164 RTT measurements:

```bash
# IMPORTANT: Install from the repository directory (not from PyPI)
cd jitterbug  # Make sure you're in the cloned repository

# Option 1: Use the installation script
./install_dev.sh

# Option 2: Manual installation with uv
uv pip install -e .  # Install jitterbug in editable mode
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git  # For Bayesian

# Test installation
python test_algorithms.py

# Quick analysis (uses ruptures algorithm - no extra dependencies needed)
jitterbug analyze examples/network_analysis/data/raw.csv

# Test all available algorithms
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm bcp      # Bayesian (93.3% accuracy)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm torch    # PyTorch Neural (93.3% accuracy)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures # Ruptures (73.3% accuracy)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm rbeast   # Rbeast (66.7% accuracy)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm adtk     # ADTK (60.0% accuracy)

# Generate visualization examples (requires matplotlib/plotly)
uv pip install matplotlib plotly  # Install visualization dependencies
jitterbug visualize examples/network_analysis/data/raw.csv --algorithm bcp --output-dir bcp_report
python3 generate_visualizations.py  # Generate all algorithm comparison plots

# View generated visualizations
ls examples/network_analysis/plots/  # Individual algorithm plots and comparison charts
```

See `examples/README.md` for detailed documentation and more examples.

### Basic Analysis

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig

# Create analyzer with default configuration
analyzer = JitterbugAnalyzer(JitterbugConfig())

# Analyze data
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')

# Print summary
for period in results.get_congested_periods():
    print(f"Congestion from {period.start_timestamp} to {period.end_timestamp}")
    print(f"  Confidence: {period.confidence:.2f}")
    print(f"  Latency jump: {period.latency_jump.magnitude:.2f}ms")
```

### Custom Configuration

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig, ChangePointDetectionConfig

# Custom configuration
config = JitterbugConfig(
    change_point_detection=ChangePointDetectionConfig(
        algorithm="ruptures",
        threshold=0.15,
        ruptures_model="l2"
    )
)

analyzer = JitterbugAnalyzer(config)
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')
```

### Real-time Analysis

```python
import pandas as pd
from jitterbug import JitterbugAnalyzer, JitterbugConfig

# Simulate real-time data
def analyze_realtime_data():
    analyzer = JitterbugAnalyzer(JitterbugConfig())
    
    # Load data in chunks
    for chunk in pd.read_csv('examples/network_analysis/data/raw.csv', chunksize=1000):
        results = analyzer.analyze_from_dataframe(chunk)
        
        # Process results
        if results.get_congested_periods():
            print(f"Alert: Congestion detected at {chunk.iloc[-1]['timestamp']}")
```

## 🔬 Research & Citations

If you use Jitterbug in your research, please cite:

```bibtex
@InProceedings{carisimo2022jitterbug,
  author="Carisimo, Esteban and Mok, Ricky K. P. and Clark, David D. and Claffy, K. C.",
  title="Jitterbug: A New Framework for Jitter-Based Congestion Inference",
  booktitle="Passive and Active Measurement",
  year="2022",
  publisher="Springer International Publishing",
  address="Cham",
  pages="155--179",
  isbn="978-3-030-98785-5"
}
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the [docs](https://github.com/estcarisimo/jitterbug/tree/main/docs)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/estcarisimo/jitterbug/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/estcarisimo/jitterbug/discussions)

## 🙏 Acknowledgments

- Northwestern University, CAIDA/UC San Diego and MIT for supporting this research
- The Passive and Active Measurement Conference (PAM) community
- All contributors and users of Jitterbug

---

**Jitterbug 2.0** - Making network congestion analysis accessible, accurate, and efficient.
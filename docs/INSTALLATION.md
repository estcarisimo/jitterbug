# Jitterbug Installation Guide

This guide covers installing Jitterbug and its optional dependencies for different algorithms. We recommend using `uv` for faster dependency resolution and installation.

## Installing uv (Recommended)

```bash
# Install uv - a fast Python package installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with homebrew (macOS)
brew install uv
```

## Basic Installation

### Core Package (Ruptures algorithm only)

```bash
# Install base package with ruptures algorithm (using uv for faster installation)
uv pip install jitterbug

# Or with traditional pip
pip install jitterbug
```

This installs:
- Core analysis framework
- Ruptures change point detection algorithm
- Jitter dispersion and KS-test methods
- Command line interface

## Optional Dependencies

### Bayesian Change Point Detection

```bash
# Method 1: Install via extra
uv pip install jitterbug[bayesian]

# Method 2: Install dependency directly (if Method 1 fails)
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git
```

**What this enables:**
- Bayesian change point detection algorithm (`--algorithm bcp`)
- Classical statistical approach with uncertainty quantification
- Better handling of noise and uncertainty

### PyTorch Neural Network Detection

```bash
# Install PyTorch dependencies
uv pip install jitterbug[torch]
```

**What this enables:**
- PyTorch-based neural network detection (`--algorithm torch`)
- Deep learning approach for complex pattern detection
- Better performance on subtle changes

### Visualization Support

```bash
# Install visualization dependencies
uv pip install jitterbug[visualization]
```

**What this enables:**
- `jitterbug visualize` command
- Static plots with matplotlib
- Interactive plots with plotly
- Comprehensive analysis reports

### API Server

```bash
# Install API server dependencies
uv pip install jitterbug[api]
```

**What this enables:**
- REST API server (`python -m jitterbug.api.server`)
- Web interface for analysis
- JSON API endpoints

### InfluxDB Support

```bash
# Install InfluxDB client
uv pip install jitterbug[influx]
```

**What this enables:**
- Direct InfluxDB data loading
- Time series database integration
- Real-time analysis capabilities

### Jupyter Notebook Support

```bash
# Install Jupyter dependencies
uv pip install jitterbug[jupyter]
```

**What this enables:**
- Jupyter notebook integration
- Interactive analysis environments
- Rich display capabilities

## Complete Installation

### Install Everything

```bash
# Install all optional dependencies
pip install jitterbug[all]
```

This includes all algorithms, visualization, API, and data source support.

### Development Installation

```bash
# Clone repository
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug

# Install in development mode
pip install -e ".[all]"

# Install development tools
pip install -e ".[dev]"
```

## Installation Verification

### Test Basic Installation

```bash
# Test CLI is working
jitterbug --help

# Test basic analysis (uses ruptures)
jitterbug analyze examples/network_analysis/data/raw.csv
```

### Test Algorithm Availability

```bash
# Test ruptures (should always work)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures

# Test Bayesian (requires bayesian extra)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm bcp

# Test PyTorch (requires torch extra)
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm torch
```

### Test Visualization

```bash
# Test visualization (requires visualization extra)
jitterbug visualize examples/network_analysis/data/raw.csv --output-dir test_viz
```

### Test API Server

```bash
# Test API server (requires api extra)
python -m jitterbug.api.server --help
```

## Troubleshooting

### Common Issues

#### 1. Bayesian Installation Fails

**Problem:**
```
ERROR: Could not find a version that satisfies the requirement bayesian_changepoint_detection
```

**Solution:**
```bash
# Install directly from GitHub
pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git
```

#### 2. PyTorch Installation Issues

**Problem:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution:**
```bash
# Install PyTorch separately first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install jitterbug
pip install jitterbug[torch]
```

#### 3. Visualization Dependencies Fail

**Problem:**
```
ERROR: Microsoft Visual C++ 14.0 is required (Windows)
```

**Solution:**
```bash
# On Windows, install Visual C++ Build Tools
# Or use conda instead:
conda install matplotlib plotly
pip install jitterbug
```

#### 4. Permission Errors

**Problem:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Install in user directory
pip install --user jitterbug[all]

# Or use virtual environment
python -m venv jitterbug_env
source jitterbug_env/bin/activate  # On Windows: jitterbug_env\Scripts\activate
pip install jitterbug[all]
```

### Environment-Specific Instructions

#### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv git build-essential

# Install jitterbug
uv pip install jitterbug[all]
```

#### macOS

```bash
# Install via Homebrew
brew install python git

# Install jitterbug
uv pip install jitterbug[all]
```

#### Windows

```bash
# Install Python from python.org
# Install Git from git-scm.com
# Install Visual C++ Build Tools

# Install jitterbug
pip install jitterbug[all]
```

## Docker Installation

### Using Docker Compose

```bash
# Clone repository
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug

# Start services
docker-compose up -d

# Use CLI
docker-compose run --rm jitterbug-cli analyze examples/network_analysis/data/raw.csv
```

### Using Docker Directly

```bash
# Build image
docker build -t jitterbug .

# Run analysis
docker run --rm -v $(pwd)/examples:/app/examples jitterbug \
  analyze /app/examples/network_analysis/data/raw.csv
```

## Version Information

```bash
# Check installed version
jitterbug version

# Check algorithm availability
python -c "
from jitterbug.detection import get_available_algorithms
print('Available algorithms:', get_available_algorithms())
"
```

## Next Steps

After installation:

1. **Read the examples**: Check `examples/README.md`
2. **Try the algorithm guide**: See `docs/ALGORITHM_USAGE.md`
3. **Analyze your data**: Use your own RTT measurements
4. **Explore visualization**: Generate analysis reports
5. **Set up API server**: For programmatic access

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Examples**: See `examples/` for working code
- **Issues**: Report problems on GitHub
- **CLI Help**: Run `jitterbug --help` for commands
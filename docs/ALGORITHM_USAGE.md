# Jitterbug Algorithm Usage Guide

This guide demonstrates all available ways to run change point detection algorithms with Jitterbug using the example dataset.

## Dataset

All examples use the comprehensive network analysis dataset:
- **File**: `examples/network_analysis/data/raw.csv`
- **Size**: 47,163 RTT measurements
- **Format**: CSV with epoch timestamps and RTT values

## Available Algorithms

Jitterbug supports multiple change point detection algorithms:

1. **Ruptures** (`ruptures`) - Fast and accurate using various models *(included by default)*
2. **Bayesian Change Point** (`bcp`) - The Bayesian detector evaluated in the PAM 2022 paper *(requires the `bcp` extra)*

### Installing Algorithm Dependencies

```bash
# From a clone (see docs/INSTALLATION.md)
uv sync                    # ruptures only
uv sync --extra bcp        # + the Bayesian detector (bayesian-changepoint, pulls in torch)
uv sync --extra all        # every optional back end
```

## Available Jitter Analysis Methods

1. **Jitter Dispersion** (`jitter_dispersion`) - Analyzes jitter variability changes
2. **Kolmogorov-Smirnov Test** (`ks_test`) - Statistical distribution change detection

---

## Command Line Usage

### Basic Usage (Default Settings)

```bash
# Uses default algorithm (ruptures) and method (jitter_dispersion)
# No additional dependencies required
jitterbug analyze examples/network_analysis/data/raw.csv
```

### Ruptures Algorithm *(No additional dependencies required)*

```bash
# Basic ruptures with default settings
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures

# Ruptures with jitter dispersion (default)
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures \
    --method jitter_dispersion

# Ruptures with Kolmogorov-Smirnov test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures \
    --method ks_test

# Ruptures with custom threshold
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures \
    --threshold 0.15

# Ruptures with high sensitivity
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures \
    --threshold 0.1 \
    --method jitter_dispersion

# Ruptures with low sensitivity
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures \
    --threshold 0.5 \
    --method ks_test
```

### Bayesian Change Point Algorithm *(requires the `bcp` extra)*

```bash
# First install the extra (from a clone):
uv sync --extra bcp

# Basic Bayesian change point detection
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm bcp

# Bayesian with jitter dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp \
    --method jitter_dispersion

# Bayesian with Kolmogorov-Smirnov test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp \
    --method ks_test

# Bayesian with custom threshold
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp \
    --threshold 0.2

# Bayesian with high sensitivity
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp \
    --threshold 0.1 \
    --method jitter_dispersion
```

### All Combinations

```bash
# Ruptures + Jitter Dispersion (default)
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures --method jitter_dispersion

# Ruptures + KS Test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures --method ks_test

# Bayesian + Jitter Dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp --method jitter_dispersion

# Bayesian + KS Test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp --method ks_test

```

---

## Configuration File Usage

### Create Configuration Template

```bash
# Generate configuration template
jitterbug config --template --output algorithm_config.yaml
```

### Example Configuration Files

#### Ruptures Configuration
```yaml
# ruptures_config.yaml
change_point_detection:
  algorithm: "ruptures"
  threshold: 0.25
  min_time_elapsed: 1800
  ruptures_model: "rbf"
  ruptures_penalty: 10.0

jitter_analysis:
  method: "jitter_dispersion"
  threshold: 0.25
  moving_average_order: 6
  moving_iqr_order: 4

output_format: "json"
verbose: true
```

#### Bayesian Configuration
```yaml
# bayesian_config.yaml
change_point_detection:
  algorithm: "bcp"
  threshold: 0.2
  min_time_elapsed: 1800

jitter_analysis:
  method: "ks_test"
  threshold: 0.25
  significance_level: 0.05

output_format: "json"
verbose: true
```

### Using Configuration Files

```bash
# Use ruptures configuration
jitterbug analyze examples/network_analysis/data/raw.csv \
    --config ruptures_config.yaml

# Use bayesian configuration
jitterbug analyze examples/network_analysis/data/raw.csv \
    --config bayesian_config.yaml

```

---

## Python API Usage

### Basic Usage

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import ChangePointDetectionConfig, JitterAnalysisConfig

# Default configuration (ruptures + jitter_dispersion)
analyzer = JitterbugAnalyzer(JitterbugConfig())
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')
```

### Ruptures Algorithm

```python
# Ruptures with jitter dispersion
config = JitterbugConfig(
    change_point_detection=ChangePointDetectionConfig(
        algorithm="ruptures",
        threshold=0.25,
        ruptures_model="rbf",
        ruptures_penalty=10.0
    ),
    jitter_analysis=JitterAnalysisConfig(
        method="jitter_dispersion",
        threshold=0.25
    )
)

analyzer = JitterbugAnalyzer(config)
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')
```

### Bayesian Algorithm

```python
# Bayesian with KS test
config = JitterbugConfig(
    change_point_detection=ChangePointDetectionConfig(
        algorithm="bcp",
        threshold=0.2
    ),
    jitter_analysis=JitterAnalysisConfig(
        method="ks_test",
        significance_level=0.05
    )
)

analyzer = JitterbugAnalyzer(config)
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')
```

### Algorithm Comparison

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import ChangePointDetectionConfig, JitterAnalysisConfig

# Test all algorithms
algorithms = ['ruptures', 'bcp']
methods = ['jitter_dispersion', 'ks_test']

results = {}

for algorithm in algorithms:
    for method in methods:
        print(f"Testing {algorithm} + {method}...")

        config = JitterbugConfig(
            change_point_detection=ChangePointDetectionConfig(
                algorithm=algorithm,
                threshold=0.25
            ),
            jitter_analysis=JitterAnalysisConfig(
                method=method,
                threshold=0.25
            )
        )

        analyzer = JitterbugAnalyzer(config)
        result = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')

        results[f"{algorithm}_{method}"] = result

        # Print summary
        summary = analyzer.get_summary_statistics(result)
        print(f"  Congested periods: {summary['congested_periods']}")
        print(f"  Congestion ratio: {summary['congestion_ratio']:.2%}")
        print()
```

---

## Output and Saving Results

### Save Results to Different Formats

```bash
# Save as JSON
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures --output results.json

# Save as CSV
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp --output results.csv

# Save as Parquet
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp --output results.parquet
```

### Verbose Output

```bash
# Detailed logging
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm ruptures --verbose

# Quiet output
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm bcp --quiet
```

---

## Performance Expectations

### Expected Performance for Example Dataset

| Detector | Method | Runtime (CLI, median) | Peak memory | Periods / congested |
|----------|--------|----------------------:|------------:|--------------------:|
| ruptures | KS test | 1.32 s | ~255 MiB | 22 / 11 |
| ruptures | Jitter dispersion | 1.21 s | ~255 MiB | 22 / 11 |
| BCP | KS test | 2.95 s | ~470 MiB | 28 / 14 |
| BCP | Jitter dispersion | 2.64 s | ~470 MiB | 28 / 14 |

Jitterbug 2.3.0 on an Apple M1, Python 3.12, whole `jitterbug analyze` command including
start-up, 5 runs. The KS test itself takes about 0.13 s and jitter dispersion 0.03 s;
about 1 s of each run is Python start-up and imports (PyTorch for BCP). See
[PERFORMANCE.md](PERFORMANCE.md) for the method and for earlier releases.

---

## Algorithm Selection Guide

### When to Use Each Algorithm

#### Ruptures
- **Best for**: Fast, accurate detection with good performance
- **Pros**: Fast execution, well-tested, multiple models available
- **Cons**: May miss subtle changes
- **Use when**: You need quick results with good accuracy

#### Bayesian Change Point (BCP)
- **Best for**: Classical statistical approach with uncertainty quantification
- **Pros**: Provides uncertainty estimates, theoretically grounded
- **Cons**: Slower execution, requires more memory
- **Use when**: You need statistical rigor and uncertainty quantification

### Method Selection Guide

#### Jitter Dispersion
- **Best for**: Network congestion detection
- **Pros**: Domain-specific, designed for network measurements
- **Cons**: Less general than statistical tests
- **Use when**: Analyzing network RTT data

#### Kolmogorov-Smirnov Test
- **Best for**: General distribution change detection
- **Pros**: General statistical test, well-established
- **Cons**: May be less sensitive to network-specific patterns
- **Use when**: You want general change detection

---

## Troubleshooting

### Common Issues

1. **Algorithm not found**: Install required dependencies
   ```bash
   uv sync --extra bcp   # the Bayesian detector (bayesian-changepoint + torch)
   ```

2. **Memory issues**: Reduce dataset size or use different algorithm
   ```bash
   # Use ruptures for lower memory usage
   jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures
   ```

3. **Slow performance**: Use ruptures algorithm for fastest results
   ```bash
   jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures
   ```

### Getting Help

```bash
# Show help for analyze command
jitterbug analyze --help

# Show all available options
jitterbug --help
```

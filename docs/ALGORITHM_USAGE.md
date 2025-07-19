# Jitterbug Algorithm Usage Guide

This guide demonstrates all available ways to run change point detection algorithms with Jitterbug using the example dataset.

## Dataset

All examples use the comprehensive network analysis dataset:
- **File**: `examples/network_analysis/data/raw.csv`
- **Size**: 47,164 RTT measurements
- **Format**: CSV with epoch timestamps and RTT values

## Available Algorithms

Jitterbug supports multiple change point detection algorithms:

1. **Ruptures** (`ruptures`) - Fast and accurate using various models *(included by default)*
2. **Bayesian Change Point** (`bcp`) - Classical Bayesian approach *(requires optional dependency)*
3. **PyTorch Neural Network** (`torch`) - Deep learning-based detection *(requires optional dependency)*
4. **Rbeast** (`rbeast`) - Seasonal pattern detection and change point analysis *(requires optional dependency)*
5. **ADTK** (`adtk`) - Anomaly Detection Toolkit with level shift detection *(requires optional dependency)*

### Installing Algorithm Dependencies

```bash
# For basic analysis (ruptures only) - no additional dependencies needed
uv pip install jitterbug

# For Bayesian change point detection
uv pip install jitterbug[bayesian]
# OR install directly:
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git

# For PyTorch neural network detection
uv pip install jitterbug[torch]

# For Rbeast seasonal pattern detection
uv pip install jitterbug[rbeast]
# OR install directly:
uv pip install Rbeast

# For ADTK anomaly detection
uv pip install jitterbug[adtk]
# OR install directly:
uv pip install adtk

# For all algorithms
uv pip install jitterbug[all]
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

### Bayesian Change Point Algorithm *(Requires: uv pip install jitterbug[bayesian])*

```bash
# First install the dependency:
uv pip install jitterbug[bayesian]

# OR install the bayesian dependency directly:
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git

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

### PyTorch Neural Network Algorithm *(Requires: uv pip install jitterbug[torch])*

```bash
# First install the dependency:
uv pip install jitterbug[torch]

# Basic PyTorch detection
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm torch

# PyTorch with jitter dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm torch \
    --method jitter_dispersion

# PyTorch with Kolmogorov-Smirnov test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm torch \
    --method ks_test

# PyTorch with custom threshold
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm torch \
    --threshold 0.3
```

### Rbeast Algorithm *(Requires: uv pip install jitterbug[rbeast])*

```bash
# First install the dependency:
uv pip install jitterbug[rbeast]

# OR install the rbeast dependency directly:
uv pip install Rbeast

# Basic Rbeast seasonal pattern detection
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm rbeast

# Rbeast with jitter dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm rbeast \
    --method jitter_dispersion

# Rbeast with Kolmogorov-Smirnov test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm rbeast \
    --method ks_test

# Rbeast with custom threshold
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm rbeast \
    --threshold 0.2
```

### ADTK Algorithm *(Requires: uv pip install jitterbug[adtk])*

```bash
# First install the dependency:
uv pip install jitterbug[adtk]

# OR install the ADTK dependency directly:
uv pip install adtk

# Basic ADTK anomaly detection
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm adtk

# ADTK with jitter dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm adtk \
    --method jitter_dispersion

# ADTK with Kolmogorov-Smirnov test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm adtk \
    --method ks_test

# ADTK with custom threshold
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm adtk \
    --threshold 0.3
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

# PyTorch + Jitter Dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm torch --method jitter_dispersion

# PyTorch + KS Test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm torch --method ks_test

# Rbeast + Jitter Dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm rbeast --method jitter_dispersion

# Rbeast + KS Test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm rbeast --method ks_test

# ADTK + Jitter Dispersion
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm adtk --method jitter_dispersion

# ADTK + KS Test
jitterbug analyze examples/network_analysis/data/raw.csv \
    --algorithm adtk --method ks_test
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

#### PyTorch Configuration
```yaml
# torch_config.yaml
change_point_detection:
  algorithm: "torch"
  threshold: 0.3
  min_time_elapsed: 1800

jitter_analysis:
  method: "jitter_dispersion"
  threshold: 0.25

output_format: "json"
verbose: true
```

#### Rbeast Configuration
```yaml
# rbeast_config.yaml
change_point_detection:
  algorithm: "rbeast"
  threshold: 0.2
  min_time_elapsed: 1800

jitter_analysis:
  method: "ks_test"
  threshold: 0.25
  significance_level: 0.05

output_format: "json"
verbose: true
```

#### ADTK Configuration
```yaml
# adtk_config.yaml
change_point_detection:
  algorithm: "adtk"
  threshold: 0.3
  min_time_elapsed: 1800

jitter_analysis:
  method: "jitter_dispersion"
  threshold: 0.25

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

# Use PyTorch configuration
jitterbug analyze examples/network_analysis/data/raw.csv \
    --config torch_config.yaml

# Use Rbeast configuration
jitterbug analyze examples/network_analysis/data/raw.csv \
    --config rbeast_config.yaml

# Use ADTK configuration
jitterbug analyze examples/network_analysis/data/raw.csv \
    --config adtk_config.yaml
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

### PyTorch Algorithm

```python
# PyTorch with jitter dispersion
config = JitterbugConfig(
    change_point_detection=ChangePointDetectionConfig(
        algorithm="torch",
        threshold=0.3
    ),
    jitter_analysis=JitterAnalysisConfig(
        method="jitter_dispersion",
        threshold=0.25
    )
)

analyzer = JitterbugAnalyzer(config)
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')
```

### Rbeast Algorithm

```python
# Rbeast with KS test
config = JitterbugConfig(
    change_point_detection=ChangePointDetectionConfig(
        algorithm="rbeast",
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

### ADTK Algorithm

```python
# ADTK with jitter dispersion
config = JitterbugConfig(
    change_point_detection=ChangePointDetectionConfig(
        algorithm="adtk",
        threshold=0.3
    ),
    jitter_analysis=JitterAnalysisConfig(
        method="jitter_dispersion",
        threshold=0.25
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
algorithms = ['ruptures', 'bcp', 'torch', 'rbeast', 'adtk']
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
    --algorithm torch --output results.parquet
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

| Algorithm | Method | Typical Runtime | Memory Usage | Change Points |
|-----------|---------|----------------|--------------|---------------|
| Ruptures | Jitter Dispersion | 10-20s | ~100MB | 15-25 |
| Ruptures | KS Test | 15-25s | ~100MB | 10-20 |
| Bayesian | Jitter Dispersion | 20-40s | ~150MB | 5-15 |
| Bayesian | KS Test | 25-45s | ~150MB | 8-18 |
| PyTorch | Jitter Dispersion | 30-60s | ~200MB | 10-30 |
| PyTorch | KS Test | 35-65s | ~200MB | 12-25 |
| Rbeast | Jitter Dispersion | 25-40s | ~150MB | 8-15 |
| Rbeast | KS Test | 30-45s | ~150MB | 10-18 |
| ADTK | Jitter Dispersion | 15-25s | ~120MB | 5-12 |
| ADTK | KS Test | 20-30s | ~120MB | 8-15 |

*Performance may vary based on system specifications and dataset characteristics.*

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

#### PyTorch Neural Network
- **Best for**: Complex patterns and subtle changes
- **Pros**: Can detect complex patterns, learns from data
- **Cons**: Requires more computational resources, needs training
- **Use when**: You have complex time series with subtle patterns

#### Rbeast
- **Best for**: Seasonal pattern detection and trend analysis
- **Pros**: Handles seasonal data well, robust to outliers
- **Cons**: May overfit to patterns, moderate computational cost
- **Use when**: Your data has seasonal components or trends

#### ADTK (Anomaly Detection Toolkit)
- **Best for**: General anomaly detection with level shifts
- **Pros**: Fast execution, simple implementation, good for basic detection
- **Cons**: Limited sensitivity, may miss subtle changes
- **Use when**: You need fast, basic anomaly detection

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
   uv pip install jitterbug[torch]  # For PyTorch
   uv pip install jitterbug[bayesian]  # For Bayesian
   uv pip install jitterbug[rbeast]  # For Rbeast
   uv pip install jitterbug[adtk]  # For ADTK
   
   # If bayesian installation fails, try direct installation:
   uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git
   
   # If other installations fail, try direct installation:
   uv pip install Rbeast  # For Rbeast
   uv pip install adtk  # For ADTK
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
# Network Analysis Example

This directory contains a comprehensive example of using Jitterbug for network congestion analysis.

## Dataset

The dataset contains RTT (Round-Trip Time) measurements from a network monitoring experiment:

- **`data/raw.csv`** - Raw RTT measurements (47,164 data points)
- **`data/mins.csv`** - Minimum RTT values computed over 15-minute intervals
- **`expected_results/`** - Expected inference results from the original research

## Usage Examples

### Basic Analysis

```bash
# Analyze with default settings
jitterbug analyze examples/network_analysis/data/raw.csv

# Analyze with specific algorithm
jitterbug analyze examples/network_analysis/data/raw.csv --algorithm ruptures --method jitter_dispersion

# Save results to JSON
jitterbug analyze examples/network_analysis/data/raw.csv --output results.json
```

### Validation

```bash
# Validate data quality
jitterbug validate examples/network_analysis/data/raw.csv --verbose
```

### Visualization (Optional)

```bash
# Install visualization dependencies first
pip install jitterbug[visualization]

# Generate comprehensive visualization report
jitterbug visualize examples/network_analysis/data/raw.csv --output-dir analysis_report

# Generate only interactive plots
jitterbug visualize examples/network_analysis/data/raw.csv --interactive-only --output-dir interactive_plots
```

### Configuration

```bash
# Generate configuration template
jitterbug config --template --output config.yaml

# Use custom configuration
jitterbug analyze examples/network_analysis/data/raw.csv --config config.yaml
```

## Data Format

The CSV files use the following format:

```csv
epoch,values
1512144010.0,63.86
1512144010.0,66.52
1512144020.0,85.2
...
```

Where:
- `epoch` - Unix timestamp (seconds since epoch)
- `values` - RTT measurement in milliseconds

## Expected Results

The `expected_results/` directory contains reference outputs from the original research:

- **`jd_inferences.csv`** - Congestion inferences using jitter dispersion method
- **`kstest_inferences.csv`** - Congestion inferences using Kolmogorov-Smirnov test

These can be used to validate that Jitterbug 2.0 produces consistent results with the original implementation.

## Python API Example

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig

# Create analyzer
config = JitterbugConfig()
analyzer = JitterbugAnalyzer(config)

# Analyze data
results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv')

# Get congestion periods
congested_periods = results.get_congested_periods()
print(f"Found {len(congested_periods)} congestion periods")

# Get summary statistics
summary = analyzer.get_summary_statistics(results)
print(f"Congestion ratio: {summary['congestion_ratio']:.2%}")
```

## Performance Notes

- The raw dataset contains ~47K measurements spanning several hours
- Analysis typically completes in under 30 seconds
- Visualization generation may take 1-2 minutes depending on system performance
- Expected memory usage: ~100MB for this dataset size
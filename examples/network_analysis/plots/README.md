# Jitterbug Algorithm Visualization Examples

This directory contains visualization examples for both change point detection algorithms in Jitterbug.

## Algorithm Performance Summary

| Algorithm | Detected Periods | Accuracy vs Expected (15) | Rating |
|-----------|------------------|----------------------------|--------|
| BCP | 14 | 93.3% | ⭐⭐⭐⭐⭐ (the paper's detector) |
| RUPTURES | 11 | 73.3% | ⭐⭐⭐⭐ (fast default) |

## Files Generated

### Individual Algorithm Analyses
- `bcp_congestion_analysis.png` - Visualization for BCP algorithm
- `bcp_summary.txt` - Performance summary for BCP
- `ruptures_congestion_analysis.png` - Visualization for RUPTURES algorithm
- `ruptures_summary.txt` - Performance summary for RUPTURES

### Comparison Charts
- `algorithm_comparison.png` - Side-by-side performance comparison
- `README.md` - This documentation file

## Usage

These visualizations demonstrate the effectiveness of different change point detection algorithms for network congestion inference. Each algorithm has different strengths:

- **BCP**: the Bayesian detector evaluated in the PAM 2022 paper
- **Ruptures**: fast, no optional dependency, the default

All visualizations use the same example dataset for fair comparison.

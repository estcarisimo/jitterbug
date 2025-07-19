# Jitterbug Algorithm Visualization Examples

This directory contains visualization examples for all change point detection algorithms in Jitterbug v2.0.

## Algorithm Performance Summary

| Algorithm | Detected Periods | Accuracy vs Expected (15) | Rating |
|-----------|------------------|----------------------------|--------|
| BCP | 14 | 93.3% | ⭐⭐⭐⭐⭐ (Gold Standard) |
| RUPTURES | 11 | 73.3% | ⭐⭐⭐⭐ (Very Good) |
| TORCH | 14 | 93.3% | ⭐⭐⭐⭐⭐ (Excellent) |
| RBEAST | 10 | 66.7% | ⭐⭐⭐ (Good) |
| ADTK | 9 | 60.0% | ⭐⭐ (Fair) |

## Files Generated

### Individual Algorithm Analyses
- `bcp_congestion_analysis.png` - Visualization for BCP algorithm
- `bcp_summary.txt` - Performance summary for BCP
- `ruptures_congestion_analysis.png` - Visualization for RUPTURES algorithm
- `ruptures_summary.txt` - Performance summary for RUPTURES
- `torch_congestion_analysis.png` - Visualization for TORCH algorithm
- `torch_summary.txt` - Performance summary for TORCH
- `rbeast_congestion_analysis.png` - Visualization for RBEAST algorithm
- `rbeast_summary.txt` - Performance summary for RBEAST
- `adtk_congestion_analysis.png` - Visualization for ADTK algorithm
- `adtk_summary.txt` - Performance summary for ADTK

### Comparison Charts
- `algorithm_comparison.png` - Side-by-side performance comparison
- `README.md` - This documentation file

## Usage

These visualizations demonstrate the effectiveness of different change point detection algorithms for network congestion inference. Each algorithm has different strengths:

- **BCP**: Gold standard with statistical rigor
- **Ruptures**: Fast and reliable
- **PyTorch**: Advanced pattern recognition
- **Rbeast**: Seasonal pattern detection
- **ADTK**: Simple anomaly detection

All visualizations use the same example dataset for fair comparison.

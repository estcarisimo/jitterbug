#!/usr/bin/env python
"""
Basic network analysis example using Jitterbug.

This script demonstrates how to perform a complete network congestion analysis
using the Jitterbug framework with the provided example dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import ChangePointDetectionConfig, JitterAnalysisConfig


def main():
    """Run basic network analysis example."""
    print("🚀 Jitterbug Network Analysis Example")
    print("=" * 50)
    
    # Define paths
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    raw_data_path = data_dir / "raw.csv"
    
    if not raw_data_path.exists():
        print(f"❌ Data file not found: {raw_data_path}")
        print("Please ensure the example data is available.")
        return 1
    
    print(f"📊 Loading data from: {raw_data_path}")
    
    # Configure analysis
    config = JitterbugConfig(
        change_point_detection=ChangePointDetectionConfig(
            algorithm="ruptures",
            threshold=0.25,
            ruptures_model="rbf",
            ruptures_penalty=10.0
        ),
        jitter_analysis=JitterAnalysisConfig(
            method="jitter_dispersion",
            threshold=0.25,
            moving_average_order=6,
            moving_iqr_order=4
        ),
        verbose=True
    )
    
    print(f"⚙️  Configuration:")
    print(f"   • Algorithm: {config.change_point_detection.algorithm}")
    print(f"   • Method: {config.jitter_analysis.method}")
    print(f"   • Threshold: {config.change_point_detection.threshold}")
    
    # Create analyzer
    analyzer = JitterbugAnalyzer(config)
    
    # Perform analysis
    print("\n🔍 Performing analysis...")
    try:
        results = analyzer.analyze_from_file(raw_data_path)
        print("✅ Analysis completed successfully!")
        
        # Get summary statistics
        summary = analyzer.get_summary_statistics(results)
        
        # Display results
        print(f"\n📈 Results Summary:")
        print(f"   • Total periods: {summary['total_periods']}")
        print(f"   • Congested periods: {summary['congested_periods']}")
        print(f"   • Congestion ratio: {summary['congestion_ratio']:.2%}")
        print(f"   • Total duration: {summary['total_duration_seconds']:.1f} seconds")
        print(f"   • Congestion duration: {summary['congestion_duration_seconds']:.1f} seconds")
        print(f"   • Average confidence: {summary['average_confidence']:.2f}")
        
        # Get congested periods
        congested_periods = results.get_congested_periods()
        
        if congested_periods:
            print(f"\n🚨 Congested Periods ({len(congested_periods)}):")
            for i, period in enumerate(congested_periods[:5], 1):  # Show first 5
                duration = period.end_epoch - period.start_epoch
                print(f"   {i}. {period.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                      f"{period.end_timestamp.strftime('%H:%M:%S')} "
                      f"({duration:.1f}s, confidence: {period.confidence:.2f})")
            
            if len(congested_periods) > 5:
                print(f"   ... and {len(congested_periods) - 5} more periods")
        else:
            print("\n✅ No congestion periods detected")
        
        # Save results
        output_path = results_dir / "analysis_results.json"
        analyzer.save_results(results, output_path, format="json")
        print(f"\n💾 Results saved to: {output_path}")
        
        # Save summary CSV for comparison
        csv_path = results_dir / "congestion_summary.csv"
        df = results.to_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"📄 Summary CSV saved to: {csv_path}")
        
        print(f"\n🎉 Analysis complete! Check the results directory: {results_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
Basic Jitterbug analysis example.

This example demonstrates how to use the new Jitterbug 2.0 API for
analyzing RTT data and detecting network congestion.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path so we can import jitterbug
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import ChangePointDetectionConfig, JitterAnalysisConfig


def create_sample_data():
    """Create sample RTT data for demonstration."""
    print("Creating sample RTT data...")
    
    # Generate 2 hours of RTT measurements every 5 seconds
    start_time = datetime.now() - timedelta(hours=2)
    duration = 2 * 3600  # 2 hours in seconds
    interval = 5  # 5 seconds
    
    timestamps = []
    rtt_values = []
    
    for i in range(0, duration, interval):
        timestamp = start_time + timedelta(seconds=i)
        epoch = timestamp.timestamp()
        
        # Simulate different network conditions
        if i < 1800:  # First 30 minutes: normal conditions
            base_rtt = 20.0
            jitter = np.random.normal(0, 2.0)
        elif i < 3600:  # Next 30 minutes: congested conditions
            base_rtt = 35.0  # Higher baseline
            jitter = np.random.normal(0, 8.0)  # More jitter
        elif i < 5400:  # Next 30 minutes: back to normal
            base_rtt = 22.0
            jitter = np.random.normal(0, 2.5)
        else:  # Last 30 minutes: another congestion period
            base_rtt = 40.0
            jitter = np.random.normal(0, 10.0)
        
        rtt = max(1.0, base_rtt + jitter)  # Ensure positive RTT
        
        timestamps.append(epoch)
        rtt_values.append(rtt)
    
    # Create DataFrame
    df = pd.DataFrame({
        'epoch': timestamps,
        'values': rtt_values
    })
    
    return df


def main():
    """Main analysis function."""
    print("🚀 Jitterbug 2.0 Basic Analysis Example")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Generated {len(df)} RTT measurements")
    
    # Save sample data to CSV
    sample_file = Path(__file__).parent / "sample_rtts.csv"
    df.to_csv(sample_file, index=False)
    print(f"Sample data saved to: {sample_file}")
    
    # Create configuration
    config = JitterbugConfig(
        change_point_detection=ChangePointDetectionConfig(
            algorithm="ruptures",
            threshold=0.2,
            min_time_elapsed=600  # 10 minutes
        ),
        jitter_analysis=JitterAnalysisConfig(
            method="jitter_dispersion",
            threshold=0.3
        ),
        verbose=True
    )
    
    print("\n📊 Analysis Configuration:")
    print(f"  Change Point Algorithm: {config.change_point_detection.algorithm}")
    print(f"  Jitter Analysis Method: {config.jitter_analysis.method}")
    print(f"  Detection Threshold: {config.change_point_detection.threshold}")
    
    # Create analyzer
    analyzer = JitterbugAnalyzer(config)
    
    # Analyze data
    print("\n🔍 Running Analysis...")
    results = analyzer.analyze_from_dataframe(df)
    
    # Display results
    print("\n📈 Analysis Results:")
    print("=" * 30)
    
    if not results.inferences:
        print("❌ No congestion periods detected")
        return
    
    # Summary statistics
    summary = analyzer.get_summary_statistics(results)
    print(f"Total Periods Analyzed: {summary['total_periods']}")
    print(f"Congestion Periods Found: {summary['congested_periods']}")
    print(f"Congestion Ratio: {summary['congestion_ratio']:.1%}")
    print(f"Total Analysis Duration: {summary['total_duration_seconds']:.1f}s")
    print(f"Total Congestion Duration: {summary['congestion_duration_seconds']:.1f}s")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    
    # Detailed results
    congested_periods = results.get_congested_periods()
    if congested_periods:
        print(f"\n🔴 Congestion Periods ({len(congested_periods)} found):")
        print("-" * 80)
        
        for i, period in enumerate(congested_periods, 1):
            duration = period.end_epoch - period.start_epoch
            
            print(f"Period {i}:")
            print(f"  Start: {period.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End: {period.end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
            print(f"  Confidence: {period.confidence:.2f}")
            
            if period.latency_jump:
                print(f"  Latency Jump: {period.latency_jump.has_jump} "
                      f"(magnitude: {period.latency_jump.magnitude:.1f}ms)")
            
            if period.jitter_analysis:
                print(f"  Jitter Change: {period.jitter_analysis.has_significant_jitter} "
                      f"(method: {period.jitter_analysis.method})")
            
            print()
    
    # Save results
    output_file = Path(__file__).parent / "analysis_results.json"
    analyzer.save_results(results, output_file, "json")
    print(f"💾 Results saved to: {output_file}")
    
    # Save CSV format too
    csv_output_file = Path(__file__).parent / "analysis_results.csv"
    analyzer.save_results(results, csv_output_file, "csv")
    print(f"💾 CSV results saved to: {csv_output_file}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
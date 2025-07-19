#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from jitterbug.analyzer import JitterbugAnalyzer
from jitterbug.models.config import JitterbugConfig
import pandas as pd
from datetime import datetime

def analyze_algorithm(algorithm_name):
    print(f"\n=== {algorithm_name.upper()} ANALYSIS ===")
    
    config = JitterbugConfig()
    config.change_point_detection.algorithm = algorithm_name
    config.jitter_analysis.method = 'ks_test'
    config.change_point_detection.threshold = 0.25
    
    analyzer = JitterbugAnalyzer(config)
    
    try:
        results = analyzer.analyze_from_file('examples/network_analysis/data/raw.csv', 'csv')
        
        print(f"Change points: {len(analyzer.change_points)}")
        print(f"Analysis periods: {len(results.inferences)}")
        
        # Detailed analysis
        congested = [inf for inf in results.inferences if inf.is_congested]
        jitter_significant = [inf for inf in results.inferences if inf.jitter_analysis and inf.jitter_analysis.has_significant_jitter]
        latency_jumps = [inf for inf in results.inferences if inf.latency_jump and inf.latency_jump.has_jump]
        
        print(f"Congestion periods: {len(congested)}")
        print(f"Periods with significant jitter: {len(jitter_significant)}")  
        print(f"Periods with latency jumps: {len(latency_jumps)}")
        
        return len(congested)
        
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    # Compare all algorithms
    algorithms = ['bcp', 'ruptures', 'torch', 'rbeast', 'adtk']
    results = {}
    
    for algo in algorithms:
        results[algo] = analyze_algorithm(algo)
    
    print(f"\n=== SUMMARY ===")
    expected_df = pd.read_csv('examples/network_analysis/expected_results/kstest_inferences.csv')
    expected = len(expected_df[expected_df['congestion'] == 1.0])
    print(f"Expected: {expected}")
    
    for algo, count in results.items():
        success_rate = count / expected * 100
        print(f"{algo.upper()}: {count} ({success_rate:.1f}%)")

if __name__ == "__main__":
    main()
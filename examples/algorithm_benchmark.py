#!/usr/bin/env python
"""
Algorithm comparison benchmark for Jitterbug change point detection.

This script compares the performance, accuracy, and characteristics of
different change point detection algorithms on various types of data.
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jitterbug.models import (
    RTTMeasurement,
    RTTDataset,
    MinimumRTTDataset,
    ChangePointDetectionConfig,
    JitterbugConfig
)
from jitterbug.detection import ChangePointDetector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DatasetGenerator:
    """Generate synthetic datasets with known change points for benchmarking."""
    
    @staticmethod
    def generate_step_changes(
        n_points: int = 100,
        change_points: List[int] = None,
        step_sizes: List[float] = None,
        noise_std: float = 1.0,
        base_value: float = 20.0
    ) -> MinimumRTTDataset:
        """Generate dataset with step changes."""
        if change_points is None:
            change_points = [n_points // 3, 2 * n_points // 3]
        if step_sizes is None:
            step_sizes = [10.0, -5.0, 8.0]
        
        # Generate time series
        values = []
        current_value = base_value
        segment_idx = 0
        
        for i in range(n_points):
            # Check for change point
            if segment_idx < len(change_points) and i >= change_points[segment_idx]:
                if segment_idx < len(step_sizes):
                    current_value += step_sizes[segment_idx]
                segment_idx += 1
            
            # Add noise
            value = current_value + np.random.normal(0, noise_std)
            values.append(max(1.0, value))
        
        # Create measurements
        measurements = []
        start_time = datetime.now() - timedelta(hours=2)
        
        for i, value in enumerate(values):
            timestamp = start_time + timedelta(minutes=i * 15)
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=value
            ))
        
        return MinimumRTTDataset(
            measurements=measurements,
            interval_minutes=15,
            metadata={
                'dataset_type': 'step_changes',
                'true_change_points': change_points,
                'step_sizes': step_sizes,
                'noise_std': noise_std,
                'base_value': base_value
            }
        )
    
    @staticmethod
    def generate_gradual_changes(
        n_points: int = 100,
        change_points: List[int] = None,
        slopes: List[float] = None,
        noise_std: float = 1.0,
        base_value: float = 20.0
    ) -> MinimumRTTDataset:
        """Generate dataset with gradual changes."""
        if change_points is None:
            change_points = [n_points // 4, 3 * n_points // 4]
        if slopes is None:
            slopes = [0.0, 0.5, -0.3, 0.2]
        
        # Generate time series
        values = []
        current_value = base_value
        segment_idx = 0
        
        for i in range(n_points):
            # Check for change point
            if segment_idx < len(change_points) and i >= change_points[segment_idx]:
                segment_idx += 1
            
            # Apply slope
            slope = slopes[min(segment_idx, len(slopes) - 1)]
            current_value += slope
            
            # Add noise
            value = current_value + np.random.normal(0, noise_std)
            values.append(max(1.0, value))
        
        # Create measurements
        measurements = []
        start_time = datetime.now() - timedelta(hours=2)
        
        for i, value in enumerate(values):
            timestamp = start_time + timedelta(minutes=i * 15)
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=value
            ))
        
        return MinimumRTTDataset(
            measurements=measurements,
            interval_minutes=15,
            metadata={
                'dataset_type': 'gradual_changes',
                'true_change_points': change_points,
                'slopes': slopes,
                'noise_std': noise_std,
                'base_value': base_value
            }
        )
    
    @staticmethod
    def generate_noisy_data(
        n_points: int = 100,
        change_points: List[int] = None,
        noise_std: float = 5.0,
        base_values: List[float] = None
    ) -> MinimumRTTDataset:
        """Generate dataset with high noise."""
        if change_points is None:
            change_points = [n_points // 2]
        if base_values is None:
            base_values = [20.0, 30.0]
        
        # Generate time series
        values = []
        segment_idx = 0
        
        for i in range(n_points):
            # Check for change point
            if segment_idx < len(change_points) and i >= change_points[segment_idx]:
                segment_idx += 1
            
            # Get base value
            base_value = base_values[min(segment_idx, len(base_values) - 1)]
            
            # Add high noise
            value = base_value + np.random.normal(0, noise_std)
            values.append(max(1.0, value))
        
        # Create measurements
        measurements = []
        start_time = datetime.now() - timedelta(hours=2)
        
        for i, value in enumerate(values):
            timestamp = start_time + timedelta(minutes=i * 15)
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=value
            ))
        
        return MinimumRTTDataset(
            measurements=measurements,
            interval_minutes=15,
            metadata={
                'dataset_type': 'noisy_data',
                'true_change_points': change_points,
                'base_values': base_values,
                'noise_std': noise_std
            }
        )
    
    @staticmethod
    def generate_real_world_like(
        n_points: int = 200,
        congestion_periods: List[tuple] = None,
        noise_std: float = 2.0
    ) -> MinimumRTTDataset:
        """Generate realistic network congestion data."""
        if congestion_periods is None:
            congestion_periods = [
                (60, 90),   # Congestion from point 60 to 90
                (140, 170)  # Congestion from point 140 to 170
            ]
        
        # Generate time series
        values = []
        base_rtt = 20.0
        
        for i in range(n_points):
            # Check if in congestion period
            is_congested = any(start <= i < end for start, end in congestion_periods)
            
            if is_congested:
                # Higher RTT with more variation during congestion
                rtt = base_rtt + 15 + np.random.exponential(5)
                jitter = np.random.normal(0, noise_std * 2)
            else:
                # Normal RTT
                rtt = base_rtt + np.random.normal(0, 2)
                jitter = np.random.normal(0, noise_std)
            
            value = max(1.0, rtt + jitter)
            values.append(value)
        
        # Create measurements
        measurements = []
        start_time = datetime.now() - timedelta(hours=4)
        
        for i, value in enumerate(values):
            timestamp = start_time + timedelta(minutes=i * 15)
            measurements.append(RTTMeasurement(
                timestamp=timestamp,
                epoch=timestamp.timestamp(),
                rtt_value=value
            ))
        
        return MinimumRTTDataset(
            measurements=measurements,
            interval_minutes=15,
            metadata={
                'dataset_type': 'real_world_like',
                'congestion_periods': congestion_periods,
                'noise_std': noise_std,
                'base_rtt': base_rtt
            }
        )


class AlgorithmBenchmark:
    """Benchmark different change point detection algorithms."""
    
    def __init__(self):
        """Initialize the benchmark."""
        self.algorithms = ['ruptures', 'bcp', 'torch']
        self.results = []
    
    def run_single_test(
        self,
        algorithm: str,
        dataset: MinimumRTTDataset,
        config_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run a single test on an algorithm."""
        if config_params is None:
            config_params = {}
        
        # Create configuration
        config = ChangePointDetectionConfig(
            algorithm=algorithm,
            **config_params
        )
        
        try:
            # Initialize detector
            detector = ChangePointDetector(config)
            
            # Measure execution time
            start_time = time.time()
            change_points = detector.detect(dataset)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Calculate metrics
            num_detected = len(change_points)
            confidence_scores = [cp.confidence for cp in change_points]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Memory usage (rough estimate)
            memory_usage = sys.getsizeof(change_points) + sum(sys.getsizeof(cp) for cp in change_points)
            
            return {
                'algorithm': algorithm,
                'success': True,
                'execution_time': execution_time,
                'num_detected': num_detected,
                'avg_confidence': avg_confidence,
                'confidence_scores': confidence_scores,
                'memory_usage': memory_usage,
                'change_points': change_points,
                'error': None
            }
            
        except Exception as e:
            return {
                'algorithm': algorithm,
                'success': False,
                'execution_time': float('inf'),
                'num_detected': 0,
                'avg_confidence': 0.0,
                'confidence_scores': [],
                'memory_usage': 0,
                'change_points': [],
                'error': str(e)
            }
    
    def run_benchmark(
        self,
        datasets: Dict[str, MinimumRTTDataset],
        config_variations: Dict[str, Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Run benchmark on multiple datasets and algorithms."""
        if config_variations is None:
            config_variations = {
                'default': {},
                'sensitive': {'threshold': 0.15},
                'conservative': {'threshold': 0.35}
            }
        
        results = []
        
        print("🔬 Running Algorithm Benchmark")
        print("=" * 50)
        
        for dataset_name, dataset in datasets.items():
            print(f"\n📊 Testing dataset: {dataset_name}")
            print(f"   Data points: {len(dataset)}")
            print(f"   Dataset type: {dataset.metadata.get('dataset_type', 'unknown')}")
            
            for config_name, config_params in config_variations.items():
                print(f"\n   Configuration: {config_name}")
                
                for algorithm in self.algorithms:
                    print(f"      Testing {algorithm}...", end=" ")
                    
                    # Run test
                    result = self.run_single_test(algorithm, dataset, config_params)
                    
                    # Add metadata
                    result.update({
                        'dataset_name': dataset_name,
                        'dataset_size': len(dataset),
                        'config_name': config_name,
                        'dataset_type': dataset.metadata.get('dataset_type', 'unknown')
                    })
                    
                    results.append(result)
                    
                    # Print result
                    if result['success']:
                        print(f"✅ {result['execution_time']:.3f}s, {result['num_detected']} CPs")
                    else:
                        print(f"❌ {result['error']}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        self.results = df
        
        return df
    
    def print_summary(self):
        """Print benchmark summary."""
        if self.results.empty:
            print("No benchmark results available.")
            return
        
        print("\n📈 Benchmark Summary")
        print("=" * 50)
        
        # Overall statistics
        successful_tests = self.results[self.results['success']]
        
        print(f"Total tests: {len(self.results)}")
        print(f"Successful tests: {len(successful_tests)}")
        print(f"Failed tests: {len(self.results) - len(successful_tests)}")
        
        if len(successful_tests) == 0:
            print("No successful tests to analyze.")
            return
        
        # Algorithm performance
        print("\n🏆 Algorithm Performance:")
        algo_stats = successful_tests.groupby('algorithm').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'num_detected': ['mean', 'std'],
            'avg_confidence': ['mean', 'std'],
            'success': 'count'
        }).round(4)
        
        print(algo_stats)
        
        # Best performers
        print("\n🥇 Best Performers:")
        
        # Fastest algorithm
        fastest = successful_tests.loc[successful_tests['execution_time'].idxmin()]
        print(f"   Fastest: {fastest['algorithm']} ({fastest['execution_time']:.3f}s)")
        
        # Most confident
        most_confident = successful_tests.loc[successful_tests['avg_confidence'].idxmax()]
        print(f"   Most confident: {most_confident['algorithm']} ({most_confident['avg_confidence']:.3f})")
        
        # Most consistent
        consistency = successful_tests.groupby('algorithm')['num_detected'].std()
        most_consistent = consistency.idxmin()
        print(f"   Most consistent: {most_consistent} (std: {consistency[most_consistent]:.3f})")
        
        # Dataset-specific performance
        print("\n📊 Dataset-Specific Performance:")
        dataset_performance = successful_tests.groupby(['dataset_name', 'algorithm']).agg({
            'execution_time': 'mean',
            'num_detected': 'mean',
            'avg_confidence': 'mean'
        }).round(4)
        
        print(dataset_performance)
    
    def generate_report(self, output_file: Path = None):
        """Generate detailed benchmark report."""
        if output_file is None:
            output_file = Path(__file__).parent / "benchmark_report.html"
        
        if self.results.empty:
            print("No benchmark results available.")
            return
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jitterbug Algorithm Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Jitterbug Algorithm Benchmark Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <p>Total tests: {len(self.results)}</p>
                <p>Successful tests: {len(self.results[self.results['success']])}</p>
                <p>Failed tests: {len(self.results[~self.results['success']])}</p>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                {self.results.to_html(classes='table', escape=False)}
            </div>
            
            <div class="section">
                <h2>Algorithm Performance</h2>
                {self.results[self.results['success']].groupby('algorithm').agg({
                    'execution_time': ['mean', 'std'],
                    'num_detected': ['mean', 'std'],
                    'avg_confidence': ['mean', 'std']
                }).round(4).to_html(classes='table')}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Detailed report saved to: {output_file}")


def main():
    """Run the algorithm benchmark."""
    print("🚀 Jitterbug Algorithm Benchmark")
    print("=" * 60)
    
    # Generate test datasets
    print("📊 Generating test datasets...")
    datasets = {
        'step_changes_small': DatasetGenerator.generate_step_changes(
            n_points=50, change_points=[15, 35], noise_std=1.0
        ),
        'step_changes_large': DatasetGenerator.generate_step_changes(
            n_points=150, change_points=[50, 100], noise_std=1.5
        ),
        'gradual_changes': DatasetGenerator.generate_gradual_changes(
            n_points=80, change_points=[20, 60], noise_std=1.2
        ),
        'noisy_data': DatasetGenerator.generate_noisy_data(
            n_points=60, change_points=[30], noise_std=3.0
        ),
        'real_world_like': DatasetGenerator.generate_real_world_like(
            n_points=120, congestion_periods=[(40, 60), (80, 100)]
        )
    }
    
    print(f"   Generated {len(datasets)} test datasets")
    
    # Configuration variations
    config_variations = {
        'default': {},
        'sensitive': {'threshold': 0.15},
        'conservative': {'threshold': 0.35},
        'fast': {'threshold': 0.25, 'min_time_elapsed': 900}
    }
    
    print(f"   Testing {len(config_variations)} configuration variations")
    
    # Run benchmark
    benchmark = AlgorithmBenchmark()
    results_df = benchmark.run_benchmark(datasets, config_variations)
    
    # Print summary
    benchmark.print_summary()
    
    # Generate report
    benchmark.generate_report()
    
    # Save results
    results_file = Path(__file__).parent / "benchmark_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    
    successful_results = results_df[results_df['success']]
    if not successful_results.empty:
        # Best overall performer
        performance_score = (
            successful_results['execution_time'].rank(ascending=True) +
            successful_results['avg_confidence'].rank(ascending=False) +
            successful_results['num_detected'].rank(ascending=False)
        )
        best_overall = successful_results.loc[performance_score.idxmin()]
        
        print(f"   🏆 Best overall performer: {best_overall['algorithm']} (config: {best_overall['config_name']})")
        
        # Best for real-time
        real_time_mask = successful_results['execution_time'] < 1.0
        if real_time_mask.any():
            real_time_best = successful_results[real_time_mask].loc[
                successful_results[real_time_mask]['avg_confidence'].idxmax()
            ]
            print(f"   ⚡ Best for real-time: {real_time_best['algorithm']} (config: {real_time_best['config_name']})")
        
        # Best for accuracy
        accuracy_best = successful_results.loc[successful_results['avg_confidence'].idxmax()]
        print(f"   🎯 Best for accuracy: {accuracy_best['algorithm']} (config: {accuracy_best['config_name']})")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
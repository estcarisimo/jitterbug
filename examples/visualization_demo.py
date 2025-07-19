#!/usr/bin/env python
"""
Demonstration of Jitterbug visualization capabilities.

This example shows how to create comprehensive visualizations of
network analysis results using both static and interactive plots.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jitterbug.models import RTTMeasurement, RTTDataset, MinimumRTTDataset
from jitterbug.analyzer import JitterbugAnalyzer
from jitterbug.visualization import JitterbugDashboard
from jitterbug.models import JitterbugConfig


def create_sample_data():
    """Create sample RTT data for demonstration."""
    print("📊 Creating sample RTT data...")
    
    # Create synthetic RTT data with congestion periods
    measurements = []
    start_time = datetime.now() - timedelta(hours=2)
    
    # Generate 120 measurements over 2 hours (1 per minute)
    for i in range(120):
        timestamp = start_time + timedelta(minutes=i)
        
        # Base RTT around 20ms
        base_rtt = 20.0
        
        # Add congestion periods
        if 30 <= i <= 45:  # Congestion period 1
            base_rtt = 35.0
        elif 80 <= i <= 100:  # Congestion period 2
            base_rtt = 45.0
        
        # Add some noise
        noise = np.random.normal(0, 2.0)
        rtt_value = max(1.0, base_rtt + noise)
        
        measurements.append(RTTMeasurement(
            timestamp=timestamp,
            epoch=timestamp.timestamp(),
            rtt_value=rtt_value,
            source="192.168.1.1",
            destination="8.8.8.8"
        ))
    
    return RTTDataset(measurements=measurements)


def create_min_rtt_data(raw_data: RTTDataset):
    """Create minimum RTT data from raw data."""
    print("🔍 Creating minimum RTT dataset...")
    
    # Group measurements by 15-minute intervals
    intervals = {}
    for measurement in raw_data.measurements:
        # Round to 15-minute intervals
        interval_start = measurement.timestamp.replace(minute=0, second=0, microsecond=0)
        interval_key = interval_start.timestamp()
        
        if interval_key not in intervals:
            intervals[interval_key] = []
        intervals[interval_key].append(measurement)
    
    # Create minimum RTT measurements
    min_measurements = []
    for interval_timestamp, measurements in intervals.items():
        if measurements:
            min_rtt = min(m.rtt_value for m in measurements)
            min_measurements.append(RTTMeasurement(
                timestamp=datetime.fromtimestamp(interval_timestamp),
                epoch=interval_timestamp,
                rtt_value=min_rtt,
                source=measurements[0].source,
                destination=measurements[0].destination
            ))
    
    return MinimumRTTDataset(
        measurements=min_measurements,
        interval_minutes=15
    )


def demonstrate_static_plots():
    """Demonstrate static plotting capabilities."""
    print("\n🎨 Demonstrating static plots...")
    
    # Create sample data
    raw_data = create_sample_data()
    min_rtt_data = create_min_rtt_data(raw_data)
    
    # Analyze data
    config = JitterbugConfig()
    analyzer = JitterbugAnalyzer(config)
    
    # Simulate analysis results
    results = analyzer.analyze_dataset(min_rtt_data)
    
    # Create dashboard
    dashboard = JitterbugDashboard()
    
    # Generate static plots
    output_dir = Path(__file__).parent / "static_plots"
    
    try:
        static_plots = dashboard.plotter.save_all_plots(
            raw_data=raw_data,
            min_rtt_data=min_rtt_data,
            results=results,
            change_points=[],  # Would be populated by change point detection
            output_dir=output_dir,
            prefix="demo"
        )
        
        print(f"✅ Static plots saved to: {output_dir}")
        for name, path in static_plots.items():
            print(f"   📄 {name}: {path}")
            
    except Exception as e:
        print(f"⚠️  Static plotting failed: {e}")
        print("   This might be due to missing matplotlib dependencies")


def demonstrate_interactive_plots():
    """Demonstrate interactive plotting capabilities."""
    print("\n🌐 Demonstrating interactive plots...")
    
    # Create sample data
    raw_data = create_sample_data()
    min_rtt_data = create_min_rtt_data(raw_data)
    
    # Analyze data
    config = JitterbugConfig()
    analyzer = JitterbugAnalyzer(config)
    
    # Simulate analysis results
    results = analyzer.analyze_dataset(min_rtt_data)
    
    # Create dashboard
    dashboard = JitterbugDashboard()
    
    # Generate interactive plots
    output_dir = Path(__file__).parent / "interactive_plots"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Main timeline
        timeline_fig = dashboard.interactive.create_interactive_timeline(
            raw_data, min_rtt_data, results, [],
            title="Demo Interactive Timeline"
        )
        timeline_path = output_dir / "timeline.html"
        dashboard.interactive.save_html(timeline_fig, timeline_path)
        
        # Dashboard
        dashboard_fig = dashboard.interactive.create_dashboard(
            raw_data, min_rtt_data, results, [],
            title="Demo Dashboard"
        )
        dashboard_path = output_dir / "dashboard.html"
        dashboard.interactive.save_html(dashboard_fig, dashboard_path)
        
        # Confidence scatter
        confidence_fig = dashboard.interactive.create_confidence_scatter(
            results, title="Demo Confidence Analysis"
        )
        confidence_path = output_dir / "confidence.html"
        dashboard.interactive.save_html(confidence_fig, confidence_path)
        
        print(f"✅ Interactive plots saved to: {output_dir}")
        print(f"   🌐 Timeline: {timeline_path}")
        print(f"   📊 Dashboard: {dashboard_path}")
        print(f"   🎯 Confidence: {confidence_path}")
        print(f"   💡 Open these files in your web browser to interact with them!")
        
    except Exception as e:
        print(f"⚠️  Interactive plotting failed: {e}")
        print("   This might be due to missing plotly dependencies")


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive report generation."""
    print("\n📋 Demonstrating comprehensive report generation...")
    
    # Create sample data
    raw_data = create_sample_data()
    min_rtt_data = create_min_rtt_data(raw_data)
    
    # Analyze data
    config = JitterbugConfig()
    analyzer = JitterbugAnalyzer(config)
    
    # Simulate analysis results
    results = analyzer.analyze_dataset(min_rtt_data)
    
    # Create dashboard
    dashboard = JitterbugDashboard()
    
    # Generate comprehensive report
    output_dir = Path(__file__).parent / "comprehensive_report"
    
    try:
        report = dashboard.create_comprehensive_report(
            raw_data=raw_data,
            min_rtt_data=min_rtt_data,
            results=results,
            change_points=[],  # Would be populated by change point detection
            output_dir=output_dir,
            title="Jitterbug Visualization Demo Report",
            include_interactive=True
        )
        
        print(f"✅ Comprehensive report generated: {output_dir}")
        print(f"   📄 Open {output_dir / 'index.html'} to view the report")
        
        # Display key statistics
        stats = report['statistics']
        print(f"\n📊 Key Statistics:")
        print(f"   • Total periods: {stats['total_periods']}")
        print(f"   • Congested periods: {stats['congested_periods']}")
        print(f"   • Congestion ratio: {stats['congestion_ratio']:.1%}")
        
        # Show file structure
        print(f"\n📁 Generated Files:")
        for root, dirs, files in (output_dir).walk():
            level = root.relative_to(output_dir).parts
            indent = "  " * len(level)
            print(f"{indent}{root.name}/")
            subindent = "  " * (len(level) + 1)
            for file in files:
                print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
        print("   This might be due to missing dependencies")


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples."""
    print("\n💻 CLI Usage Examples:")
    print("=" * 50)
    
    print("📊 Basic visualization:")
    print("   jitterbug visualize data.csv")
    
    print("\n🎨 Static plots only:")
    print("   jitterbug visualize data.csv --static-only")
    
    print("\n🌐 Interactive plots only:")
    print("   jitterbug visualize data.csv --interactive-only")
    
    print("\n📁 Custom output directory:")
    print("   jitterbug visualize data.csv --output-dir my_plots")
    
    print("\n🏷️ Custom title:")
    print("   jitterbug visualize data.csv --title 'My Network Analysis'")
    
    print("\n⚙️ With custom algorithm:")
    print("   jitterbug visualize data.csv --algorithm bcp --threshold 0.3")
    
    print("\n🔍 With verbose output:")
    print("   jitterbug visualize data.csv --verbose")


def main():
    """Run all visualization demonstrations."""
    print("🎯 Jitterbug Visualization Demo")
    print("=" * 60)
    
    try:
        # Check dependencies
        dependencies_ok = True
        
        try:
            import matplotlib
            print("✅ matplotlib available")
        except ImportError:
            print("⚠️  matplotlib not available - static plots will be disabled")
            dependencies_ok = False
        
        try:
            import plotly
            print("✅ plotly available")
        except ImportError:
            print("⚠️  plotly not available - interactive plots will be disabled")
            dependencies_ok = False
        
        if not dependencies_ok:
            print("\n💡 Install visualization dependencies with:")
            print("   pip install jitterbug[visualization]")
            print("   or: pip install matplotlib plotly")
        
        # Run demonstrations
        demonstrate_static_plots()
        demonstrate_interactive_plots()
        demonstrate_comprehensive_report()
        demonstrate_cli_usage()
        
        print("\n🎉 Visualization demo complete!")
        print("   Check the generated files in the examples directory")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
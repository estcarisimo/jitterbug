#!/usr/bin/env python
"""
Demonstration of various output formats using Pydantic models.

This example shows how to use Jitterbug's Pydantic models to output
analysis results in different formats with proper validation.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jitterbug.models import (
    ChangePoint,
    CongestionInference,
    CongestionInferenceResult,
    JitterAnalysis,
    JitterbugConfig,
    LatencyJump,
)


def create_sample_results():
    """Create sample analysis results for demonstration."""
    print("📊 Creating sample analysis results...")

    # Create sample change points
    base_time = datetime.now() - timedelta(hours=2)

    [
        ChangePoint(
            timestamp=base_time + timedelta(minutes=30),
            epoch=(base_time + timedelta(minutes=30)).timestamp(),
            confidence=0.85,
            algorithm="ruptures_rbf",
        ),
        ChangePoint(
            timestamp=base_time + timedelta(minutes=60),
            epoch=(base_time + timedelta(minutes=60)).timestamp(),
            confidence=0.72,
            algorithm="ruptures_rbf",
        ),
        ChangePoint(
            timestamp=base_time + timedelta(minutes=90),
            epoch=(base_time + timedelta(minutes=90)).timestamp(),
            confidence=0.91,
            algorithm="ruptures_rbf",
        ),
    ]

    # Create sample latency jumps
    latency_jumps = [
        LatencyJump(
            start_timestamp=base_time + timedelta(minutes=30),
            end_timestamp=base_time + timedelta(minutes=60),
            start_epoch=(base_time + timedelta(minutes=30)).timestamp(),
            end_epoch=(base_time + timedelta(minutes=60)).timestamp(),
            has_jump=True,
            magnitude=12.5,
            threshold=0.5,
        ),
        LatencyJump(
            start_timestamp=base_time + timedelta(minutes=60),
            end_timestamp=base_time + timedelta(minutes=90),
            start_epoch=(base_time + timedelta(minutes=60)).timestamp(),
            end_epoch=(base_time + timedelta(minutes=90)).timestamp(),
            has_jump=False,
            magnitude=-2.1,
            threshold=0.5,
        ),
    ]

    # Create sample jitter analyses
    jitter_analyses = [
        JitterAnalysis(
            start_timestamp=base_time + timedelta(minutes=30),
            end_timestamp=base_time + timedelta(minutes=60),
            start_epoch=(base_time + timedelta(minutes=30)).timestamp(),
            end_epoch=(base_time + timedelta(minutes=60)).timestamp(),
            has_significant_jitter=True,
            jitter_metric=0.45,
            method="jitter_dispersion",
            threshold=0.25,
        ),
        JitterAnalysis(
            start_timestamp=base_time + timedelta(minutes=60),
            end_timestamp=base_time + timedelta(minutes=90),
            start_epoch=(base_time + timedelta(minutes=60)).timestamp(),
            end_epoch=(base_time + timedelta(minutes=90)).timestamp(),
            has_significant_jitter=False,
            jitter_metric=0.15,
            method="jitter_dispersion",
            threshold=0.25,
        ),
    ]

    # Create sample congestion inferences
    congestion_inferences = [
        CongestionInference(
            start_timestamp=base_time + timedelta(minutes=30),
            end_timestamp=base_time + timedelta(minutes=60),
            start_epoch=(base_time + timedelta(minutes=30)).timestamp(),
            end_epoch=(base_time + timedelta(minutes=60)).timestamp(),
            is_congested=True,
            confidence=0.82,
            latency_jump=latency_jumps[0],
            jitter_analysis=jitter_analyses[0],
        ),
        CongestionInference(
            start_timestamp=base_time + timedelta(minutes=60),
            end_timestamp=base_time + timedelta(minutes=90),
            start_epoch=(base_time + timedelta(minutes=60)).timestamp(),
            end_epoch=(base_time + timedelta(minutes=90)).timestamp(),
            is_congested=False,
            confidence=0.0,
            latency_jump=latency_jumps[1],
            jitter_analysis=jitter_analyses[1],
        ),
    ]

    # Create final result
    result = CongestionInferenceResult(
        inferences=congestion_inferences,
        metadata={
            "total_measurements": 120,
            "min_intervals": 8,
            "change_points": 3,
            "latency_jumps": 2,
            "jitter_analyses": 2,
            "congestion_periods": 1,
            "algorithm_used": "ruptures_rbf",
            "analysis_duration": 7200,  # 2 hours
            "config": JitterbugConfig().dict(),
        },
    )

    return result


def demonstrate_json_output(results: CongestionInferenceResult):
    """Demonstrate JSON output format."""
    print("\n🔧 JSON Output Format")
    print("=" * 40)

    # Standard JSON output
    json_output = results.dict()

    # Pretty-print JSON
    print("📄 Standard JSON Output:")
    print(json.dumps(json_output, indent=2, default=str))

    # Compact JSON output
    compact_json = json.dumps(json_output, separators=(",", ":"), default=str)
    print(f"\n📦 Compact JSON Output (length: {len(compact_json)} chars):")
    print(compact_json[:200] + "..." if len(compact_json) > 200 else compact_json)

    # Custom JSON serialization
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)

    custom_json = json.dumps(json_output, cls=CustomJSONEncoder, indent=2)
    print("\n🎨 Custom JSON Output (with datetime formatting):")
    print(custom_json[:500] + "..." if len(custom_json) > 500 else custom_json)

    # Save to file
    output_file = Path(__file__).parent / "sample_results.json"
    with output_file.open("w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\n💾 JSON saved to: {output_file}")


def demonstrate_csv_output(results: CongestionInferenceResult):
    """Demonstrate CSV output format."""
    print("\n📊 CSV Output Format")
    print("=" * 40)

    # Convert to DataFrame
    df = results.to_dataframe()

    # Basic CSV output
    print("📄 Basic CSV Output:")
    print(df.to_csv(index=False))

    # Enhanced CSV with additional columns
    enhanced_df = df.copy()

    # Add detailed information
    enhanced_df["latency_jump_magnitude"] = [
        inf.latency_jump.magnitude if inf.latency_jump else None for inf in results.inferences
    ]
    enhanced_df["jitter_method"] = [
        inf.jitter_analysis.method if inf.jitter_analysis else None for inf in results.inferences
    ]
    enhanced_df["jitter_metric"] = [
        inf.jitter_analysis.jitter_metric if inf.jitter_analysis else None
        for inf in results.inferences
    ]
    enhanced_df["confidence"] = [inf.confidence for inf in results.inferences]

    print("\n📈 Enhanced CSV Output:")
    print(enhanced_df.to_csv(index=False))

    # Save to file
    output_file = Path(__file__).parent / "sample_results.csv"
    enhanced_df.to_csv(output_file, index=False)
    print(f"\n💾 CSV saved to: {output_file}")


def demonstrate_parquet_output(results: CongestionInferenceResult):
    """Demonstrate Parquet output format."""
    print("\n🗂️  Parquet Output Format")
    print("=" * 40)

    try:
        # Convert to DataFrame with proper types
        df = results.to_dataframe()

        # Add metadata as columns
        df["analysis_duration"] = results.metadata.get("analysis_duration", 0)
        df["total_measurements"] = results.metadata.get("total_measurements", 0)
        df["algorithm_used"] = results.metadata.get("algorithm_used", "unknown")

        # Add detailed analysis results
        for i, inf in enumerate(results.inferences):
            df.loc[i, "latency_jump_magnitude"] = (
                inf.latency_jump.magnitude if inf.latency_jump else None
            )
            df.loc[i, "latency_jump_threshold"] = (
                inf.latency_jump.threshold if inf.latency_jump else None
            )
            df.loc[i, "jitter_method"] = inf.jitter_analysis.method if inf.jitter_analysis else None
            df.loc[i, "jitter_metric"] = (
                inf.jitter_analysis.jitter_metric if inf.jitter_analysis else None
            )
            df.loc[i, "jitter_threshold"] = (
                inf.jitter_analysis.threshold if inf.jitter_analysis else None
            )
            df.loc[i, "confidence"] = inf.confidence

        print("📄 Parquet Schema:")
        print(df.dtypes)
        print(f"\n📊 DataFrame Shape: {df.shape}")
        print(f"📏 Memory Usage: {df.memory_usage().sum()} bytes")

        # Save to Parquet
        output_file = Path(__file__).parent / "sample_results.parquet"
        df.to_parquet(output_file)
        print(f"\n💾 Parquet saved to: {output_file}")

        # Show file size comparison
        json_file = Path(__file__).parent / "sample_results.json"
        csv_file = Path(__file__).parent / "sample_results.csv"

        print("\n📏 File Size Comparison:")
        if json_file.exists():
            print(f"   JSON: {json_file.stat().st_size} bytes")
        if csv_file.exists():
            print(f"   CSV: {csv_file.stat().st_size} bytes")
        if output_file.exists():
            print(f"   Parquet: {output_file.stat().st_size} bytes")

    except ImportError:
        print("⚠️  Parquet output requires 'pyarrow' or 'fastparquet' library")
        print("   Install with: pip install pyarrow")


def demonstrate_xml_output(results: CongestionInferenceResult):
    """Demonstrate XML output format."""
    print("\n🏷️  XML Output Format")
    print("=" * 40)

    def dict_to_xml(data, root_name="root"):
        """Convert dictionary to XML string."""

        def build_xml(obj, name):
            if isinstance(obj, dict):
                xml = f"<{name}>"
                for key, value in obj.items():
                    xml += build_xml(value, key)
                xml += f"</{name}>"
                return xml
            elif isinstance(obj, list):
                xml = ""
                for item in obj:
                    xml += build_xml(item, name[:-1] if name.endswith("s") else name)
                return xml
            else:
                return f"<{name}>{obj}</{name}>"

        return f'<?xml version="1.0" encoding="UTF-8"?>\n{build_xml(data, root_name)}'

    # Convert to XML
    xml_data = results.dict()
    xml_output = dict_to_xml(xml_data, "JitterbugResults")

    print("📄 XML Output:")
    print(xml_output[:1000] + "..." if len(xml_output) > 1000 else xml_output)

    # Save to file
    output_file = Path(__file__).parent / "sample_results.xml"
    with output_file.open("w") as f:
        f.write(xml_output)
    print(f"\n💾 XML saved to: {output_file}")


def demonstrate_yaml_output(results: CongestionInferenceResult):
    """Demonstrate YAML output format."""
    print("\n📝 YAML Output Format")
    print("=" * 40)

    try:
        import yaml

        # Convert to YAML
        yaml_data = results.dict()
        yaml_output = yaml.dump(yaml_data, default_flow_style=False, default_str=str)

        print("📄 YAML Output:")
        print(yaml_output[:1000] + "..." if len(yaml_output) > 1000 else yaml_output)

        # Save to file
        output_file = Path(__file__).parent / "sample_results.yaml"
        with output_file.open("w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, default_str=str)
        print(f"\n💾 YAML saved to: {output_file}")

    except ImportError:
        print("⚠️  YAML output requires 'PyYAML' library")
        print("   Install with: pip install PyYAML")


def demonstrate_summary_output(results: CongestionInferenceResult):
    """Demonstrate summary output format."""
    print("\n📋 Summary Output Format")
    print("=" * 40)

    # Calculate summary statistics
    total_periods = len(results.inferences)
    congested_periods = len(results.get_congested_periods())
    total_duration = results.get_total_congestion_duration()

    # Create summary dictionary
    summary = {
        "analysis_summary": {
            "total_periods": total_periods,
            "congested_periods": congested_periods,
            "congestion_ratio": congested_periods / total_periods if total_periods > 0 else 0,
            "total_congestion_duration_seconds": total_duration,
            "average_confidence": np.mean(
                [inf.confidence for inf in results.get_congested_periods()]
            )
            if results.get_congested_periods()
            else 0,
        },
        "metadata": results.metadata,
        "congested_periods": [
            {
                "start_time": inf.start_timestamp.isoformat(),
                "end_time": inf.end_timestamp.isoformat(),
                "duration_seconds": inf.end_epoch - inf.start_epoch,
                "confidence": inf.confidence,
                "has_latency_jump": inf.latency_jump.has_jump if inf.latency_jump else False,
                "latency_jump_magnitude": inf.latency_jump.magnitude if inf.latency_jump else None,
                "has_jitter_increase": inf.jitter_analysis.has_significant_jitter
                if inf.jitter_analysis
                else False,
                "jitter_metric": inf.jitter_analysis.jitter_metric if inf.jitter_analysis else None,
            }
            for inf in results.get_congested_periods()
        ],
    }

    print("📊 Summary JSON:")
    print(json.dumps(summary, indent=2, default=str))

    # Save summary
    output_file = Path(__file__).parent / "sample_results_summary.json"
    with output_file.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n💾 Summary saved to: {output_file}")


def demonstrate_custom_serialization(results: CongestionInferenceResult):
    """Demonstrate custom serialization options."""
    print("\n🎨 Custom Serialization Options")
    print("=" * 40)

    # Custom serialization for specific fields
    class JitterbugResultsSerializer:
        def __init__(self, results: CongestionInferenceResult):
            self.results = results

        def to_minimal_dict(self):
            """Minimal representation with only essential fields."""
            return {
                "congestion_periods": [
                    {
                        "start": inf.start_epoch,
                        "end": inf.end_epoch,
                        "congested": inf.is_congested,
                        "confidence": round(inf.confidence, 2),
                    }
                    for inf in self.results.inferences
                ],
                "summary": {
                    "total": len(self.results.inferences),
                    "congested": len(self.results.get_congested_periods()),
                    "duration": self.results.get_total_congestion_duration(),
                },
            }

        def to_detailed_dict(self):
            """Detailed representation with all available information."""
            return {
                "analysis_results": [
                    {
                        "time_period": {
                            "start": inf.start_timestamp.isoformat(),
                            "end": inf.end_timestamp.isoformat(),
                            "duration_minutes": (inf.end_epoch - inf.start_epoch) / 60,
                        },
                        "congestion_inference": {
                            "is_congested": inf.is_congested,
                            "confidence": inf.confidence,
                            "algorithm_confidence": inf.confidence,
                        },
                        "latency_analysis": {
                            "has_jump": inf.latency_jump.has_jump if inf.latency_jump else None,
                            "magnitude_ms": inf.latency_jump.magnitude
                            if inf.latency_jump
                            else None,
                            "threshold_ms": inf.latency_jump.threshold
                            if inf.latency_jump
                            else None,
                        },
                        "jitter_analysis": {
                            "method": inf.jitter_analysis.method if inf.jitter_analysis else None,
                            "has_significant_change": inf.jitter_analysis.has_significant_jitter
                            if inf.jitter_analysis
                            else None,
                            "metric_value": inf.jitter_analysis.jitter_metric
                            if inf.jitter_analysis
                            else None,
                            "threshold": inf.jitter_analysis.threshold
                            if inf.jitter_analysis
                            else None,
                            "p_value": inf.jitter_analysis.p_value if inf.jitter_analysis else None,
                        },
                    }
                    for inf in self.results.inferences
                ],
                "metadata": self.results.metadata,
            }

        def to_time_series_dict(self):
            """Time series representation suitable for plotting."""
            return {
                "timestamps": [inf.start_timestamp.isoformat() for inf in self.results.inferences],
                "congestion_status": [inf.is_congested for inf in self.results.inferences],
                "confidence_scores": [inf.confidence for inf in self.results.inferences],
                "latency_jumps": [
                    inf.latency_jump.magnitude if inf.latency_jump else 0
                    for inf in self.results.inferences
                ],
                "jitter_metrics": [
                    inf.jitter_analysis.jitter_metric if inf.jitter_analysis else 0
                    for inf in self.results.inferences
                ],
            }

    serializer = JitterbugResultsSerializer(results)

    # Demonstrate different serialization formats
    print("📦 Minimal Format:")
    minimal = serializer.to_minimal_dict()
    print(json.dumps(minimal, indent=2))

    print("\n📊 Time Series Format:")
    time_series = serializer.to_time_series_dict()
    print(json.dumps(time_series, indent=2))

    print("\n🔍 Detailed Format:")
    detailed = serializer.to_detailed_dict()
    print(
        json.dumps(detailed, indent=2)[:500] + "..."
        if len(json.dumps(detailed, indent=2)) > 500
        else json.dumps(detailed, indent=2)
    )

    # Save custom formats
    formats = {"minimal": minimal, "time_series": time_series, "detailed": detailed}

    for format_name, data in formats.items():
        output_file = Path(__file__).parent / f"sample_results_{format_name}.json"
        with output_file.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n💾 {format_name.title()} format saved to: {output_file}")


def demonstrate_pydantic_validation():
    """Demonstrate Pydantic validation features."""
    print("\n✅ Pydantic Validation Features")
    print("=" * 40)

    print("🔍 Testing data validation:")

    # Test valid data
    try:
        valid_inference = CongestionInference(
            start_timestamp=datetime.now(),
            end_timestamp=datetime.now() + timedelta(minutes=30),
            start_epoch=datetime.now().timestamp(),
            end_epoch=(datetime.now() + timedelta(minutes=30)).timestamp(),
            is_congested=True,
            confidence=0.85,
        )
        print("✅ Valid inference created successfully")

        # Test validation
        assert valid_inference.confidence == 0.85
        assert valid_inference.is_congested is True
        print("✅ Validation passed")

    except Exception as e:
        print(f"❌ Validation failed: {e}")

    # Test invalid data
    print("\n🚫 Testing invalid data:")

    # Invalid confidence (> 1.0)
    try:
        CongestionInference(
            start_timestamp=datetime.now(),
            end_timestamp=datetime.now() + timedelta(minutes=30),
            start_epoch=datetime.now().timestamp(),
            end_epoch=(datetime.now() + timedelta(minutes=30)).timestamp(),
            is_congested=True,
            confidence=1.5,  # Invalid!
        )
        print("❌ Should have failed validation!")
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")

    # Invalid time order
    try:
        now = datetime.now()
        CongestionInference(
            start_timestamp=now,
            end_timestamp=now - timedelta(minutes=30),  # Earlier than start!
            start_epoch=now.timestamp(),
            end_epoch=(now - timedelta(minutes=30)).timestamp(),
            is_congested=True,
            confidence=0.85,
        )
        print("❌ Should have failed validation!")
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")


def main():
    """Run all output format demonstrations."""
    print("🎯 Jitterbug Output Formats Demonstration")
    print("=" * 60)

    # Create sample results
    results = create_sample_results()

    # Demonstrate all output formats
    demonstrate_json_output(results)
    demonstrate_csv_output(results)
    demonstrate_parquet_output(results)
    demonstrate_xml_output(results)
    demonstrate_yaml_output(results)
    demonstrate_summary_output(results)
    demonstrate_custom_serialization(results)
    demonstrate_pydantic_validation()

    print("\n" + "=" * 60)
    print("🎉 Output Format Demonstration Complete!")
    print("=" * 60)

    print("\n💡 Key Benefits of Pydantic Models:")
    print("   ✅ Automatic data validation")
    print("   ✅ Type safety and conversion")
    print("   ✅ Multiple output formats")
    print("   ✅ JSON schema generation")
    print("   ✅ Custom serialization options")
    print("   ✅ IDE support with type hints")

    print("\n📁 Generated Files:")
    example_dir = Path(__file__).parent
    for file in example_dir.glob("sample_results*"):
        print(f"   📄 {file.name}")


if __name__ == "__main__":
    main()

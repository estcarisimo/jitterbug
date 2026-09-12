#!/usr/bin/env python
"""
Demo client for Jitterbug API.

This example shows how to interact with the Jitterbug REST API
to analyze network RTT data for congestion detection.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import requests


class JitterbugAPIClient:
    """Client for interacting with the Jitterbug API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def health_check(self) -> dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

    def get_status(self) -> dict[str, Any]:
        """Get API status and statistics."""
        response = self.session.get(f"{self.base_url}/api/v1/status")
        response.raise_for_status()
        return response.json()

    def get_algorithms(self) -> dict[str, Any]:
        """Get available algorithms."""
        response = self.session.get(f"{self.base_url}/api/v1/algorithms")
        response.raise_for_status()
        return response.json()

    def get_methods(self) -> dict[str, Any]:
        """Get available jitter analysis methods."""
        response = self.session.get(f"{self.base_url}/api/v1/methods")
        response.raise_for_status()
        return response.json()

    def get_config_template(self) -> dict[str, Any]:
        """Get configuration template."""
        response = self.session.get(f"{self.base_url}/api/v1/config/template")
        response.raise_for_status()
        return response.json()

    def validate_data(self, measurements: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate RTT data."""
        request_data = {
            "data": {
                "measurements": measurements,
                "metadata": {"source": "api_client_demo", "timestamp": datetime.now().isoformat()},
            }
        }

        response = self.session.post(f"{self.base_url}/api/v1/validate", json=request_data)
        response.raise_for_status()
        return response.json()

    def analyze_data(
        self,
        measurements: list[dict[str, Any]],
        algorithm: str = "ruptures",
        method: str = "jitter_dispersion",
        threshold: float = 0.25,
        min_time_elapsed: int = 1800,
    ) -> dict[str, Any]:
        """Analyze RTT data for congestion detection."""
        request_data = {
            "data": {
                "measurements": measurements,
                "metadata": {"source": "api_client_demo", "timestamp": datetime.now().isoformat()},
            },
            "algorithm": algorithm,
            "method": method,
            "threshold": threshold,
            "min_time_elapsed": min_time_elapsed,
        }

        response = self.session.post(f"{self.base_url}/api/v1/analyze", json=request_data)
        response.raise_for_status()
        return response.json()

    def compare_algorithms(
        self, measurements: list[dict[str, Any]], algorithms: list[str] = None
    ) -> dict[str, Any]:
        """Compare different algorithms."""
        if algorithms is None:
            algorithms = ["ruptures", "bcp"]

        request_data = {
            "data": {
                "measurements": measurements,
                "metadata": {"source": "api_client_demo", "timestamp": datetime.now().isoformat()},
            },
            "algorithms": algorithms,
        }

        response = self.session.post(
            f"{self.base_url}/api/v1/compare-algorithms", json=request_data
        )
        response.raise_for_status()
        return response.json()


def create_sample_data(num_points: int = 120) -> list[dict[str, Any]]:
    """Create sample RTT data with congestion periods."""
    print(f"📊 Creating {num_points} sample RTT measurements...")

    measurements = []
    start_time = datetime.now() - timedelta(hours=2)

    for i in range(num_points):
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

        measurements.append(
            {
                "timestamp": timestamp.isoformat(),
                "epoch": timestamp.timestamp(),
                "rtt_value": rtt_value,
                "source": "192.168.1.1",
                "destination": "8.8.8.8",
            }
        )

    return measurements


def demonstrate_api_usage():
    """Demonstrate API usage with comprehensive examples."""
    print("🎯 Jitterbug API Client Demo")
    print("=" * 50)

    # Initialize client
    client = JitterbugAPIClient()

    try:
        # Health check
        print("\n🏥 Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Uptime: {health['uptime']:.1f}s")

        # Get status
        print("\n📊 Getting service status...")
        status = client.get_status()
        print(f"   Requests processed: {status['requests_processed']}")
        print(f"   Errors: {status['errors']}")
        print(f"   Avg response time: {status['avg_response_time']:.3f}s")

        # Get available algorithms
        print("\n🧮 Available algorithms:")
        algorithms = client.get_algorithms()
        for name, info in algorithms.items():
            available = "✅" if info["available"] else "❌"
            print(f"   {available} {name}: {info['description']}")

        # Get available methods
        print("\n📈 Available methods:")
        methods = client.get_methods()
        for name, info in methods.items():
            available = "✅" if info["available"] else "❌"
            print(f"   {available} {name}: {info['description']}")

        # Create sample data
        measurements = create_sample_data(120)

        # Validate data
        print("\n🔍 Validating data...")
        validation = client.validate_data(measurements)
        if validation["valid"]:
            print("   ✅ Data validation passed")
            print(f"   Total measurements: {validation['metrics']['total_measurements']}")
            print(f"   Duration: {validation['metrics']['duration_seconds']:.1f}s")
        else:
            print("   ❌ Data validation failed")
            for error in validation["errors"]:
                print(f"     • {error}")

        # Analyze data
        print("\n🔬 Analyzing data...")
        analysis = client.analyze_data(
            measurements, algorithm="ruptures", method="jitter_dispersion", threshold=0.25
        )

        if analysis["success"]:
            print("   ✅ Analysis completed successfully")
            print(f"   Execution time: {analysis['execution_time']:.2f}s")
            print(f"   Request ID: {analysis['request_id']}")

            result = analysis["result"]
            print(f"   Total periods: {result['metadata']['total_periods']}")
            print(f"   Congested periods: {result['metadata']['congested_periods']}")
            print(f"   Change points: {result['metadata']['change_points']}")

            # Show congestion periods
            congested_periods = [inf for inf in result["inferences"] if inf["is_congested"]]

            if congested_periods:
                print("\n   🚨 Congestion periods detected:")
                for period in congested_periods:
                    start = period["start_timestamp"]
                    end = period["end_timestamp"]
                    confidence = period["confidence"]
                    print(f"     • {start} to {end} (confidence: {confidence:.2f})")
            else:
                print("   ✅ No congestion periods detected")
        else:
            print(f"   ❌ Analysis failed: {analysis['error']}")

        # Compare algorithms
        print("\n⚖️  Comparing algorithms...")
        comparison = client.compare_algorithms(measurements, algorithms=["ruptures", "bcp"])

        if comparison["success"]:
            print("   ✅ Algorithm comparison completed")
            print(f"   Execution time: {comparison['execution_time']:.2f}s")

            for algorithm, change_points in comparison["results"].items():
                performance = comparison["performance"][algorithm]
                print(f"   {algorithm}: {len(change_points)} change points in {performance:.3f}s")
        else:
            print(f"   ❌ Algorithm comparison failed: {comparison['error']}")

        # Get configuration template
        print("\n⚙️  Configuration template:")
        config = client.get_config_template()
        print(f"   Change point algorithm: {config['change_point_detection']['algorithm']}")
        print(f"   Jitter method: {config['jitter_analysis']['method']}")
        print(f"   Threshold: {config['change_point_detection']['threshold']}")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server")
        print("   Make sure the server is running:")
        print("   python -m jitterbug.api.server")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def demonstrate_error_handling():
    """Demonstrate error handling."""
    print("\n🚨 Error Handling Demo")
    print("=" * 30)

    client = JitterbugAPIClient()

    try:
        # Test with invalid data
        print("\n🔍 Testing with invalid data...")
        invalid_measurements = [
            {
                "timestamp": "invalid-timestamp",
                "epoch": "not-a-number",
                "rtt_value": -5.0,  # Invalid negative RTT
                "source": "192.168.1.1",
                "destination": "8.8.8.8",
            }
        ]

        validation = client.validate_data(invalid_measurements)
        print(f"   Valid: {validation['valid']}")
        if not validation["valid"]:
            print("   Errors:")
            for error in validation["errors"]:
                print(f"     • {error}")

        # Test with empty data
        print("\n🔍 Testing with empty data...")
        empty_validation = client.validate_data([])
        print(f"   Valid: {empty_validation['valid']}")
        if not empty_validation["valid"]:
            print("   Errors:")
            for error in empty_validation["errors"]:
                print(f"     • {error}")

    except Exception as e:
        print(f"❌ Error handling demo failed: {e}")


def main():
    """Main function."""
    print("🌐 Starting Jitterbug API Client Demo")
    print("=" * 60)

    # Basic usage demo
    success = demonstrate_api_usage()

    if success:
        # Error handling demo
        demonstrate_error_handling()

        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   • Check the API documentation at http://localhost:8000/docs")
        print("   • Integrate the API into your network monitoring system")
        print("   • Customize algorithms and thresholds for your use case")
    else:
        print("\n❌ Demo failed. Please ensure the API server is running.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Docker deployment demo for Jitterbug.

This script demonstrates how to use Jitterbug with Docker containers
for both API and CLI usage.
"""

import subprocess
import time
import requests
import json
import os
from pathlib import Path


def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    return result


def check_docker():
    """Check if Docker is available."""
    try:
        result = run_command("docker --version", check=False)
        if result.returncode != 0:
            print("❌ Docker is not installed or not running")
            return False
        print(f"✅ Docker is available: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False


def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        result = run_command("docker-compose --version", check=False)
        if result.returncode != 0:
            print("❌ Docker Compose is not installed")
            return False
        print(f"✅ Docker Compose is available: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Docker Compose check failed: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample CSV data
    csv_data = """epoch,values
1512144000.0,25.6
1512144060.0,26.2
1512144120.0,45.3
1512144180.0,44.8
1512144240.0,46.1
1512144300.0,27.1
1512144360.0,26.8
1512144420.0,25.9
"""
    
    with open(data_dir / "sample_rtts.csv", "w") as f:
        f.write(csv_data)
    
    print(f"✅ Sample data created in {data_dir}/sample_rtts.csv")


def demo_docker_build():
    """Demonstrate building the Docker image."""
    print("\n🏗️  Building Docker Image")
    print("=" * 40)
    
    try:
        # Build the image
        result = run_command(
            "docker build -t jitterbug:demo .",
            capture_output=False
        )
        
        if result.returncode == 0:
            print("✅ Docker image built successfully")
            
            # Show image info
            result = run_command("docker images jitterbug:demo")
            print(f"Image info:\n{result.stdout}")
            
            return True
        else:
            print("❌ Docker build failed")
            return False
            
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False


def demo_docker_compose():
    """Demonstrate using Docker Compose."""
    print("\n🐳 Docker Compose Demo")
    print("=" * 40)
    
    try:
        # Start services
        print("Starting services...")
        run_command("docker-compose up -d", capture_output=False)
        
        # Wait for services to be ready
        print("Waiting for services to be ready...")
        time.sleep(10)
        
        # Check if API is running
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is running and healthy")
                health_data = response.json()
                print(f"   Status: {health_data['status']}")
                print(f"   Version: {health_data['version']}")
            else:
                print(f"⚠️  API responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to API: {e}")
        
        # Show running containers
        result = run_command("docker-compose ps")
        print(f"Running containers:\n{result.stdout}")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker Compose demo failed: {e}")
        return False


def demo_api_usage():
    """Demonstrate API usage with Docker."""
    print("\n🌐 API Usage Demo")
    print("=" * 40)
    
    try:
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/api/v1/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test algorithms endpoint
        print("Getting available algorithms...")
        response = requests.get(f"{base_url}/api/v1/algorithms")
        if response.status_code == 200:
            algorithms = response.json()
            print("✅ Available algorithms:")
            for name, info in algorithms.items():
                status = "✅" if info['available'] else "❌"
                print(f"   {status} {name}: {info['description']}")
        
        # Test with sample data
        print("Testing analysis with sample data...")
        sample_data = {
            "data": {
                "measurements": [
                    {
                        "timestamp": "2024-01-01T10:00:00Z",
                        "epoch": 1704110400.0,
                        "rtt_value": 25.6,
                        "source": "192.168.1.1",
                        "destination": "8.8.8.8"
                    },
                    {
                        "timestamp": "2024-01-01T10:01:00Z",
                        "epoch": 1704110460.0,
                        "rtt_value": 45.3,
                        "source": "192.168.1.1",
                        "destination": "8.8.8.8"
                    }
                ]
            },
            "algorithm": "ruptures",
            "method": "jitter_dispersion"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/analyze",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("✅ Analysis completed successfully")
                print(f"   Execution time: {result['execution_time']:.2f}s")
                print(f"   Request ID: {result['request_id']}")
            else:
                print(f"❌ Analysis failed: {result['error']}")
        else:
            print(f"❌ Analysis request failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ API usage demo failed: {e}")
        return False


def demo_cli_usage():
    """Demonstrate CLI usage with Docker."""
    print("\n💻 CLI Usage Demo")
    print("=" * 40)
    
    try:
        # Test help command
        print("Testing help command...")
        result = run_command(
            "docker-compose run --rm jitterbug-cli help",
            capture_output=False
        )
        
        # Test analysis command
        print("Testing analysis command...")
        result = run_command(
            "docker-compose run --rm jitterbug-cli analyze /app/data/sample_rtts.csv",
            capture_output=False
        )
        
        if result.returncode == 0:
            print("✅ CLI analysis completed successfully")
        else:
            print(f"❌ CLI analysis failed with code {result.returncode}")
        
        # Test validation command
        print("Testing validation command...")
        result = run_command(
            "docker-compose run --rm jitterbug-cli validate /app/data/sample_rtts.csv",
            capture_output=False
        )
        
        if result.returncode == 0:
            print("✅ CLI validation completed successfully")
        else:
            print(f"❌ CLI validation failed with code {result.returncode}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI usage demo failed: {e}")
        return False


def cleanup():
    """Clean up Docker resources."""
    print("\n🧹 Cleaning Up")
    print("=" * 40)
    
    try:
        # Stop and remove containers
        print("Stopping containers...")
        run_command("docker-compose down", capture_output=False)
        
        # Remove demo image
        print("Removing demo image...")
        run_command("docker rmi jitterbug:demo", check=False)
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main demo function."""
    print("🐳 Jitterbug Docker Demo")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        print("Please install Docker and try again")
        return False
    
    if not check_docker_compose():
        print("Please install Docker Compose and try again")
        return False
    
    # Create sample data
    create_sample_data()
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")
    
    success = True
    
    try:
        # Demo steps
        if not demo_docker_build():
            success = False
        
        if success and not demo_docker_compose():
            success = False
        
        if success and not demo_api_usage():
            success = False
        
        if success and not demo_cli_usage():
            success = False
        
    finally:
        # Always cleanup
        cleanup()
    
    if success:
        print("\n🎉 Docker demo completed successfully!")
        print("\n💡 Next steps:")
        print("   • Build and deploy your own Docker image")
        print("   • Set up Docker Compose for production")
        print("   • Configure monitoring and logging")
        print("   • Scale with Kubernetes or Docker Swarm")
    else:
        print("\n❌ Docker demo encountered issues")
        print("   • Check Docker installation")
        print("   • Review error messages above")
        print("   • Ensure required ports are available")
    
    return success


if __name__ == "__main__":
    main()
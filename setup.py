"""
Setup script for Jitterbug 2.0.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements_path = this_directory / "requirements-new.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "ruptures>=1.1.0",
        "typer>=0.9.0",
        "rich>=12.0.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
    ]

# Optional dependencies
extras_require = {
    "torch": ["torch>=1.9.0"],
    "bayesian": ["bayesian_changepoint_detection @ git+https://github.com/estcarisimo/bayesian_changepoint_detection.git"],
    "influx": ["influxdb-client>=1.30.0"],
    "visualization": ["matplotlib>=3.5.0", "plotly>=5.0.0"],
    "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0"],
    "jupyter": ["ipykernel>=6.0.0", "jupyter>=1.0.0"],
    "dev": [
        "pytest>=6.2.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=4.0.0",
        "mypy>=0.910",
    ],
    "all": [
        "torch>=1.9.0",
        "bayesian_changepoint_detection @ git+https://github.com/estcarisimo/bayesian_changepoint_detection.git",
        "influxdb-client>=1.30.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "ipykernel>=6.0.0",
        "jupyter>=1.0.0",
    ],
}

setup(
    name="jitterbug",
    version="2.1.0-dev",
    author="Esteban Carisimo",
    author_email="esteban.carisimo@northwestern.edu",
    description="Framework for jitter-based network congestion inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/estcarisimo/jitterbug",
    project_urls={
        "Bug Reports": "https://github.com/estcarisimo/jitterbug/issues",
        "Source": "https://github.com/estcarisimo/jitterbug",
        "Documentation": "https://github.com/estcarisimo/jitterbug/blob/main/README.md",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "jitterbug=jitterbug.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Internet :: Log Analysis",
    ],
    keywords="network congestion rtt jitter analysis changepoint detection",
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
#!/bin/bash
# Development installation script for Jitterbug using uv

echo "🚀 Installing Jitterbug for development with uv..."

# Check if we're in the jitterbug directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: Please run this script from the jitterbug repository root directory"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Install core package in editable mode
echo "📚 Installing Jitterbug core with uv..."
uv pip install -e .

# Install optional dependencies
echo "🔧 Installing optional dependencies..."

# Install bayesian dependency directly from GitHub
echo "  • Installing Bayesian change point detection..."
uv pip install git+https://github.com/estcarisimo/bayesian_changepoint_detection.git

# Try to install other extras
echo "  • Installing PyTorch support..."
uv pip install -e ".[torch]" 2>/dev/null || echo "    ⚠️  PyTorch installation failed (optional)"

echo "  • Installing visualization support..."
uv pip install -e ".[visualization]" 2>/dev/null || echo "    ⚠️  Visualization installation failed (optional)"

echo "  • Installing API server support..."
uv pip install -e ".[api]" 2>/dev/null || echo "    ⚠️  API installation failed (optional)"

# Test installation
echo "🧪 Testing installation..."
python -c "
from jitterbug import JitterbugAnalyzer, JitterbugConfig
print('✅ Core import successful')
try:
    from jitterbug.detection.algorithms import BayesianChangePointDetector
    print('✅ Bayesian algorithm available')
except:
    print('❌ Bayesian algorithm not available')
try:
    from jitterbug.detection.algorithms import TorchChangePointDetector
    print('✅ PyTorch algorithm available')
except:
    print('⚠️  PyTorch algorithm not available (optional)')
"

echo ""
echo "✨ Installation complete!"
echo ""
echo "To use Jitterbug:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run analysis: jitterbug analyze examples/network_analysis/data/raw.csv"
echo ""
echo "Available algorithms:"
python -c "
from jitterbug.detection import get_available_algorithms
for algo in get_available_algorithms():
    print(f'  • {algo}')
" 2>/dev/null || echo "  Run 'jitterbug analyze --help' to see available algorithms"
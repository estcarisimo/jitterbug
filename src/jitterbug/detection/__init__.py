"""
Change point detection algorithms for Jitterbug.
"""

from .change_point_detector import ChangePointDetector
from .algorithms import RupturesDetector

# Make optional algorithms available but don't fail if dependencies are missing
try:
    from .algorithms import BayesianChangePointDetector
except ImportError:
    BayesianChangePointDetector = None

try:
    from .algorithms import TorchChangePointDetector
except ImportError:
    TorchChangePointDetector = None

try:
    from .algorithms import RbeastDetector
except ImportError:
    RbeastDetector = None

try:
    from .algorithms import ADTKDetector
except ImportError:
    ADTKDetector = None

__all__ = [
    "ChangePointDetector",
    "RupturesDetector",
]

# Add optional algorithms to __all__ if available
if BayesianChangePointDetector is not None:
    __all__.append("BayesianChangePointDetector")
if TorchChangePointDetector is not None:
    __all__.append("TorchChangePointDetector")
if RbeastDetector is not None:
    __all__.append("RbeastDetector")
if ADTKDetector is not None:
    __all__.append("ADTKDetector")


def get_available_algorithms():
    """Get list of available change point detection algorithms."""
    algorithms = ["ruptures"]
    if BayesianChangePointDetector is not None:
        algorithms.append("bcp")
    if TorchChangePointDetector is not None:
        algorithms.append("torch")
    if RbeastDetector is not None:
        algorithms.append("rbeast")
    if ADTKDetector is not None:
        algorithms.append("adtk")
    return algorithms
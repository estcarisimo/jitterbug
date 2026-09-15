"""
Change point detection algorithms for Jitterbug.
"""

from importlib.util import find_spec

from .algorithms import BayesianChangePointDetector, RupturesDetector
from .change_point_detector import ChangePointDetector

__all__ = [
    "BayesianChangePointDetector",
    "ChangePointDetector",
    "RupturesDetector",
    "get_available_algorithms",
]

# Algorithm name -> the importable package it needs (None when it is a core dependency).
_ALGORITHM_REQUIREMENTS: dict[str, str | None] = {
    "ruptures": None,
    "bcp": "bayesian_changepoint_detection",
}


def get_available_algorithms() -> list[str]:
    """
    Return the change point detection algorithms whose dependencies are installed.

    Returns
    -------
    list[str]
        Algorithm names accepted by ``ChangePointDetectionConfig.algorithm`` that can
        actually run in this environment.
    """
    return [
        name
        for name, package in _ALGORITHM_REQUIREMENTS.items()
        if package is None or find_spec(package) is not None
    ]

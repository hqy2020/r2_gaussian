#
# FSGS: Few-Shot Gaussian Splatting (ECCV 2024)
# Paper: https://arxiv.org/abs/2312.00451
# Authors: Zehao Zhu, Zhiwen Fan, Yifan Jiang, Zhangyang Wang
#

"""
FSGS Module for R²-Gaussian

This module implements the Proximity-Guided Densification algorithm from the FSGS paper.

Key Components:
    - ProximityGuidedDensifier: Core FSGS algorithm for sparse region densification
    - FSGSConfig: Configuration dataclass
    - Visualization utilities

Usage:
    from r2_gaussian.innovations.fsgs import ProximityGuidedDensifier

    # Initialize
    densifier = ProximityGuidedDensifier(k_neighbors=5, proximity_threshold=0.05)

    # Compute proximity scores
    scores, neighbor_indices, _ = densifier.compute_proximity_scores(positions, return_neighbors=True)

    # Identify densify candidates
    candidates = densifier.identify_densify_candidates(scores)

    # Generate new Gaussians
    new_gaussians = densifier.generate_new_gaussians(source, neighbor_indices, all_positions, attributes)
"""

from .proximity_densifier import ProximityGuidedDensifier
from .config import FSGSConfig
from .utils import compute_knn, log_proximity_stats

__all__ = [
    "ProximityGuidedDensifier",
    "FSGSConfig",
    "compute_knn",
    "log_proximity_stats",
]

__version__ = "2.1.0"  # Simplified version without Medical Constraints
__paper__ = "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting (ECCV 2024)"

#
# FSGS: Few-Shot Gaussian Splatting (ECCV 2024)
# Paper: https://arxiv.org/abs/2312.00451
# Authors: Zehao Zhu, Zhiwen Fan, Yifan Jiang, Zhangyang Wang
#
# Integrated into R²-Gaussian with Medical Constraints enhancement
#

"""
FSGS Module for R²-Gaussian

This module implements the Proximity-Guided Densification algorithm from the FSGS paper,
with additional Medical Constraints enhancement for CT reconstruction.

Key Components:
    - ProximityGuidedDensifier: Core FSGS algorithm
    - MedicalConstraints: Tissue-aware adaptive parameters
    - FSGSConfig: Configuration dataclass
    - Visualization utilities

Usage:
    from r2_gaussian.innovations.fsgs import ProximityGuidedDensifier, MedicalConstraints

    # Initialize
    densifier = ProximityGuidedDensifier(k_neighbors=6, proximity_threshold=8.0)
    medical = MedicalConstraints.from_organ_type("foot")

    # Compute proximity scores
    scores = densifier.compute_proximity_scores(positions)

    # Apply medical constraints
    tissue_types = medical.classify_tissue(opacity_values)
    adaptive_params = medical.get_adaptive_params(tissue_types)

    # Identify densify candidates
    candidates = densifier.identify_densify_candidates(scores, **adaptive_params)

    # Generate new Gaussians
    new_gaussians = densifier.generate_new_gaussians(source, destination, attributes)
"""

from .proximity_densifier import ProximityGuidedDensifier
from .medical_constraints import MedicalConstraints, TissueConfig
from .config import FSGSConfig
from .utils import compute_knn, log_proximity_stats

__all__ = [
    "ProximityGuidedDensifier",
    "MedicalConstraints",
    "TissueConfig",
    "FSGSConfig",
    "compute_knn",
    "log_proximity_stats",
]

__version__ = "2.0.0"  # Rewritten version with modular design
__paper__ = "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting (ECCV 2024)"

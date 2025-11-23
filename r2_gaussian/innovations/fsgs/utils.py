#
# FSGS Utility Functions
#

import torch
from typing import Tuple, Dict, Optional
from torch.utils.tensorboard import SummaryWriter


def compute_knn(
    positions: torch.Tensor,  # (N, 3)
    k: int,
    use_cuda: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute K-nearest neighbors (standalone utility)

    This is a standalone version for use outside ProximityGuidedDensifier.

    Args:
        positions: (N, 3) Gaussian positions
        k: Number of nearest neighbors
        use_cuda: Whether to try CUDA acceleration

    Returns:
        neighbor_distances: (N, K) distances to K nearest neighbors
        neighbor_indices: (N, K) indices of K nearest neighbors
    """
    # Try CUDA-accelerated K-NN
    if use_cuda:
        try:
            from simple_knn._C import distCUDA2
            distances_sorted = distCUDA2(positions)
            neighbor_distances = distances_sorted[:, 1:k+1]
            _, neighbor_indices = torch.topk(
                distances_sorted,
                k=k+1,
                dim=1,
                largest=False
            )
            neighbor_indices = neighbor_indices[:, 1:]
            return neighbor_distances, neighbor_indices
        except:
            pass  # Fall through to PyTorch implementation

    # PyTorch fallback
    N = positions.shape[0]
    distances = torch.cdist(positions, positions, p=2)  # (N, N)

    # Set self-distance to inf
    distances.fill_diagonal_(float('inf'))

    # Find K nearest neighbors
    neighbor_distances, neighbor_indices = torch.topk(
        distances,
        k=k,
        dim=1,
        largest=False
    )

    return neighbor_distances, neighbor_indices


def log_proximity_stats(
    tb_writer: SummaryWriter,
    proximity_scores: torch.Tensor,  # (N,)
    tissue_types: Optional[torch.Tensor],  # (N,) or None
    densify_mask: torch.Tensor,  # (N,) bool
    iteration: int,
    tissue_names: Optional[list] = None
):
    """
    Log FSGS proximity statistics to TensorBoard

    Args:
        tb_writer: TensorBoard SummaryWriter
        proximity_scores: (N,) proximity scores
        tissue_types: (N,) tissue type IDs (optional, for medical constraints)
        densify_mask: (N,) boolean mask of densification candidates
        iteration: Current training iteration
        tissue_names: List of tissue type names (optional)
    """
    # Overall statistics
    tb_writer.add_histogram(
        "fsgs/proximity_scores",
        proximity_scores,
        iteration
    )

    tb_writer.add_scalar(
        "fsgs/avg_proximity_score",
        proximity_scores.mean().item(),
        iteration
    )

    tb_writer.add_scalar(
        "fsgs/num_densify_candidates",
        densify_mask.sum().item(),
        iteration
    )

    tb_writer.add_scalar(
        "fsgs/densify_ratio",
        densify_mask.sum().item() / len(densify_mask),
        iteration
    )

    # Per-tissue statistics (if medical constraints enabled)
    if tissue_types is not None and tissue_names is not None:
        for tissue_id, tissue_name in enumerate(tissue_names):
            tissue_mask = (tissue_types == tissue_id)
            if tissue_mask.sum() > 0:
                # Average proximity score for this tissue
                avg_prox = proximity_scores[tissue_mask].mean().item()
                tb_writer.add_scalar(
                    f"fsgs/proximity_{tissue_name}",
                    avg_prox,
                    iteration
                )

                # Number of Gaussians in this tissue
                tb_writer.add_scalar(
                    f"fsgs/num_{tissue_name}",
                    tissue_mask.sum().item(),
                    iteration
                )

                # Densification rate for this tissue
                densify_in_tissue = (densify_mask & tissue_mask).sum().item()
                tissue_count = tissue_mask.sum().item()
                densify_rate = densify_in_tissue / tissue_count if tissue_count > 0 else 0.0
                tb_writer.add_scalar(
                    f"fsgs/densify_rate_{tissue_name}",
                    densify_rate,
                    iteration
                )


def visualize_proximity_graph(
    positions: torch.Tensor,  # (N, 3)
    neighbor_indices: torch.Tensor,  # (N, K)
    proximity_scores: torch.Tensor,  # (N,)
    save_path: str,
    max_points: int = 1000
):
    """
    Visualize proximity graph (for debugging/analysis)

    Creates a 3D visualization of Gaussians colored by proximity score,
    with edges to nearest neighbors.

    Args:
        positions: (N, 3) Gaussian positions
        neighbor_indices: (N, K) K-nearest neighbor indices
        proximity_scores: (N,) proximity scores
        save_path: Path to save visualization (e.g., "proximity_graph.png")
        max_points: Maximum points to visualize (subsample if N > max_points)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("⚠️ matplotlib not available. Skipping proximity graph visualization.")
        return

    # Subsample if too many points
    if positions.shape[0] > max_points:
        indices = torch.randperm(positions.shape[0])[:max_points]
        positions = positions[indices]
        proximity_scores = proximity_scores[indices]
        neighbor_indices = neighbor_indices[indices]

    # Convert to numpy
    pos_np = positions.cpu().numpy()
    scores_np = proximity_scores.cpu().numpy()
    neighbors_np = neighbor_indices.cpu().numpy()

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot colored by proximity score
    scatter = ax.scatter(
        pos_np[:, 0],
        pos_np[:, 1],
        pos_np[:, 2],
        c=scores_np,
        cmap='viridis',
        s=20,
        alpha=0.6
    )

    # Add edges to nearest neighbors (only show a subset)
    num_edges_to_show = min(100, len(pos_np))
    for i in range(num_edges_to_show):
        for j in range(min(2, neighbors_np.shape[1])):  # Show 2 nearest neighbors
            neighbor_idx = neighbors_np[i, j]
            if neighbor_idx < len(pos_np):
                ax.plot(
                    [pos_np[i, 0], pos_np[neighbor_idx, 0]],
                    [pos_np[i, 1], pos_np[neighbor_idx, 1]],
                    [pos_np[i, 2], pos_np[neighbor_idx, 2]],
                    'gray',
                    alpha=0.1,
                    linewidth=0.5
                )

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Proximity Score')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Proximity Graph Visualization')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✓ Proximity graph saved to: {save_path}")


def compute_densification_efficiency(
    num_new_gaussians: int,
    prev_psnr: float,
    current_psnr: float
) -> float:
    """
    Compute densification efficiency metric

    Measures how much PSNR improvement per new Gaussian added.

    Args:
        num_new_gaussians: Number of Gaussians added in densification
        prev_psnr: PSNR before densification
        current_psnr: PSNR after densification

    Returns:
        efficiency: PSNR improvement per 1000 new Gaussians
    """
    if num_new_gaussians == 0:
        return 0.0

    psnr_improvement = current_psnr - prev_psnr
    efficiency = (psnr_improvement / num_new_gaussians) * 1000

    return efficiency

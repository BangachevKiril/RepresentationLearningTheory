import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensures 3D projection is registered


def plot_vectors(U: torch.Tensor,
                 V: torch.Tensor,
                 criterion,
                 ax=None,
                 plot_grid: bool = False,
                 title: str | None = None):
    """
    Plot the optimized vectors on a unit sphere (expects dim=3).

    Args:
        U: Tensor [n, 3]
        V: Tensor [n, 3]
        criterion: Loss object that provides get_temperature() and get_bias()
        ax: Optional matplotlib 3D axes
        plot_grid: Whether to show axis grid/labels
        title: Optional figure title; if None, uses rb/T from criterion
    """
    if U.ndim != 2 or V.ndim != 2 or U.shape[1] != 3 or V.shape[1] != 3:
        raise ValueError("plot_vectors expects U and V with shape [n, 3]")

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    U_np = U.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    n = U_np.shape[0]

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

    # Points
    ax.scatter(U_np[:, 0], U_np[:, 1], U_np[:, 2], c='blue', s=20, label='U vectors')
    ax.scatter(V_np[:, 0], V_np[:, 1], V_np[:, 2], c='red', s=20, label='V vectors')

    # Lines between pairs
    for i in range(n):
        ax.plot([U_np[i, 0], V_np[i, 0]],
                [U_np[i, 1], V_np[i, 1]],
                [U_np[i, 2], V_np[i, 2]], 'k--', alpha=0.2)

    # Labels/title
    if plot_grid:
        ax.set_xlabel('X', fontsize=16)
        ax.set_ylabel('Y', fontsize=16)
        ax.set_zlabel('Z', fontsize=16)

    temp = criterion.get_temperature()
    relative_bias = criterion.get_bias()
    if title is None:
        title = f'rb={relative_bias:.2f}, T={temp:.2f}'
    ax.set_title(title, fontsize=16)

    ax.set_box_aspect([1, 1, 1])
    ax.grid(plot_grid)
    ax.set_axis_off()
    ax.legend(fontsize=16)

    return ax


def plot_inner_product_gap(U_final: torch.Tensor,
                           V_final: torch.Tensor,
                           bins: int = 15,
                           log: bool = True):
    """
    Plot histograms for matching vs non-matching inner products.
    """
    if U_final.ndim != 2 or V_final.ndim != 2 or U_final.shape != V_final.shape:
        raise ValueError("U_final and V_final must be 2D tensors with same shape [n, d]")

    n_classes = U_final.shape[0]
    device = U_final.device

    inner_products = torch.matmul(U_final, V_final.t())
    matching_pairs = torch.diag(inner_products).detach().cpu().numpy()

    mask = ~torch.eye(n_classes, dtype=torch.bool, device=device)
    non_matching_pairs = inner_products[mask].detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(matching_pairs, bins=bins, alpha=0.5, label='Matching pairs (U_i, V_i)',
             color='blue', density=True, log=log)
    plt.hist(non_matching_pairs, bins=bins, alpha=0.5, label='Non-matching pairs (U_i, V_j)',
             color='red', density=True, log=log)

    min_matching = np.min(matching_pairs)
    max_non_matching = np.max(non_matching_pairs)
    midpoint = (min_matching + max_non_matching) / 2
    plt.axvline(x=midpoint, color='red', linestyle='--', label='Separation Point')

    plt.xlabel('Inner Product Value', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title('Distribution of Inner Products (Normalized)', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_margin(U: torch.Tensor, V: torch.Tensor):
    """
    Calculate the margin between matching and non-matching pairs.

    Returns:
        tuple: (margin, min_matching_sim, max_non_matching_sim)
    """
    cosine_sim = torch.matmul(U, V.t())

    diag_indices = torch.arange(U.shape[0], device=U.device)
    matching_sims = cosine_sim[diag_indices, diag_indices]
    min_matching_sim = torch.min(matching_sims).item()

    mask = torch.ones_like(cosine_sim, dtype=torch.bool)
    mask[diag_indices, diag_indices] = False

    non_matching_sims = cosine_sim[mask]
    max_non_matching_sim = torch.max(non_matching_sims).item()

    margin = (min_matching_sim - max_non_matching_sim) / 2

    return margin, min_matching_sim, max_non_matching_sim


def analyze_results(all_results, relative_biases):
    """
    Analyze and plot results from multiple experiments.

    Args:
        all_results (list): List of (U, V, criterion, losses) tuples
        relative_biases (list): List of relative bias values used
    """
    margins = []
    min_matching_sims = []
    max_non_matching_sims = []
    final_temps = []

    for U, V, criterion, _ in all_results:
        margin, min_match, max_non_match = calculate_margin(U, V)
        margins.append(margin)
        min_matching_sims.append(min_match)
        max_non_matching_sims.append(max_non_match)
        final_temps.append(criterion.get_temperature())

    # Plot margins
    plt.figure(figsize=(12, 8))
    plt.plot(relative_biases, margins, 'o-', linewidth=2, color='green', markersize=8)
    plt.xlabel('Relative Bias', fontsize=14)
    plt.ylabel('Margin', fontsize=14)
    plt.title('Margin Between Matching and Non-matching Pairs', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('siglip_margins.png', dpi=300)
    plt.show()

    # Plot similarity values
    plt.figure(figsize=(12, 8))
    plt.plot(relative_biases, min_matching_sims, 'o-', linewidth=2, label='Min Matching Similarity', markersize=6)
    plt.plot(relative_biases, max_non_matching_sims, 'o-', linewidth=2, label='Max Non-matching Similarity', markersize=6)
    plt.fill_between(relative_biases, min_matching_sims, max_non_matching_sims, alpha=0.2, color='green', label='Margin')
    plt.xlabel('Relative Bias', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.title('Similarity Values and Margins', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('siglip_similarities.png', dpi=300)
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Tuple


def _ensure_ax(ax=None, projection: Optional[str] = None):
    if ax is not None:
        return ax
    if projection:
        fig = plt.figure(figsize=(10, 10))
        return fig.add_subplot(111, projection=projection)
    fig, ax = plt.subplots(figsize=(8, 5))
    return ax


def _title_from_params(criterion=None, temperature=None, relative_bias=None, fallback: Optional[str] = None) -> Optional[str]:
    t = None
    rb = None
    if criterion is not None:
        # Accept objects that expose get_temperature/get_bias
        try:
            t = float(getattr(criterion, "get_temperature")())
        except Exception:
            pass
        try:
            rb = float(getattr(criterion, "get_bias")())
        except Exception:
            pass
    if temperature is not None:
        t = float(temperature)
    if relative_bias is not None:
        rb = float(relative_bias)

    if t is not None and rb is not None:
        return f"rb={rb:.2f}, T={t:.2f}"
    return fallback


def plot_vectors(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    criterion=None,
    temperature: Optional[float] = None,
    relative_bias: Optional[float] = None,
    title: Optional[str] = None,
    ax=None,
    indices: Tuple[int, int, int] = (0, 1, 2),
    show_sphere: bool = True,
    connect_pairs: bool = True,
    u_color: str = "blue",
    v_color: str = "red",
    s: int = 20,
):
    """3D plot of U and V vectors on the unit sphere.

    Uses the first three (or provided indices) coordinates for visualization.
    """
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2D tensors of shape (n, d)")
    if U.size(1) <= max(indices) or V.size(1) <= max(indices):
        raise ValueError(
            f"U/V must have at least {max(indices)+1} dims; got {U.size(1)} and {V.size(1)}"
        )

    ax = _ensure_ax(ax, projection="3d")

    U_np = U.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()

    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    i, j, k = indices
    ax.scatter(U_np[:, i], U_np[:, j], U_np[:, k], c=u_color, s=s, label="U vectors")
    ax.scatter(V_np[:, i], V_np[:, j], V_np[:, k], c=v_color, s=s, label="V vectors")

    if connect_pairs:
        for r in range(U_np.shape[0]):
            ax.plot(
                [U_np[r, i], V_np[r, i]],
                [U_np[r, j], V_np[r, j]],
                [U_np[r, k], V_np[r, k]],
                "k--",
                alpha=0.2,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    auto_title = _title_from_params(criterion, temperature, relative_bias, title)
    if auto_title:
        ax.set_title(auto_title)

    ax.set_box_aspect([1, 1, 1])
    return ax


def plot_losses(
    losses: Iterable[float],
    *,
    ax=None,
    title: str = "Training Loss Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    label: str = "Loss",
    grid: bool = True,
    logy: bool = False,
    savepath: Optional[str] = None,
):
    ax = _ensure_ax(ax)
    ax.plot(list(losses), label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if label:
        ax.legend()
    if savepath:
        ax.figure.savefig(savepath, dpi=300)
    return ax


def plot_final_metric_vs_param(
    df,
    param_name: str,
    metric_name: str,
    *,
    ax=None,
    style: str = "o-",
    linewidth: int = 2,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
):
    ax = _ensure_ax(ax)
    ax.plot(df[param_name], df[metric_name], style, linewidth=linewidth)
    ax.set_xlabel(xlabel or param_name.capitalize())
    pretty_metric = metric_name.replace("_", " ").capitalize()
    ax.set_ylabel(ylabel or pretty_metric)
    ax.set_title(title or f"{pretty_metric} vs {param_name}")
    if grid:
        ax.grid(True, alpha=0.3)
    return ax


def plot_inner_product_gaps_across_sweep(
    all_U: Iterable[torch.Tensor],
    all_V: Iterable[torch.Tensor],
    sweep_param: str,
    values: Iterable,
    *,
    bins: int = 15,
    density: bool = True,
    log: bool = False,
    cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = False,
):
    all_U = list(all_U)
    all_V = list(all_V)
    values = list(values)
    n = len(values)
    cols = min(cols, n) if n > 0 else cols
    rows = int(np.ceil(n / cols)) if n > 0 else 1
    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()

    for i, (U, V, val) in enumerate(zip(all_U, all_V, values)):
        ax = axes[i]
        inner_products = torch.matmul(U, V.t())

        matching = torch.diag(inner_products).detach().cpu().numpy()
        mask = ~torch.eye(U.shape[0], dtype=bool, device=U.device)
        non_matching = inner_products[mask].detach().cpu().numpy()

        ax.hist(matching, bins=bins, alpha=0.5, label="Matching", color="blue", density=density, log=log)
        ax.hist(non_matching, bins=bins, alpha=0.5, label="Non-matching", color="green", density=density, log=log)

        ax.set_title(f"{sweep_param}={val}")
        ax.legend(fontsize=8)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_inner_product_gap(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    bins: int = 15,
    density: bool = True,
    log: bool = True,
    ax=None,
    title: str = "Distribution of Inner Products (Normalized)",
    show_sep: bool = True,
    colors: Tuple[str, str] = ("blue", "green"),
):
    ax = _ensure_ax(ax)

    inner_products = torch.matmul(U, V.t())
    matching_pairs = torch.diag(inner_products).detach().cpu().numpy()
    mask = ~torch.eye(U.shape[0], dtype=bool, device=U.device)
    non_matching_pairs = inner_products[mask].detach().cpu().numpy()

    ax.hist(matching_pairs, bins=bins, alpha=0.5, label="Matching pairs (U_i, V_i)", color=colors[0], density=density, log=log)
    ax.hist(non_matching_pairs, bins=bins, alpha=0.5, label="Non-matching pairs (U_i, V_j)", color=colors[1], density=density, log=log)

    min_matching = float(np.min(matching_pairs))
    max_non_matching = float(np.max(non_matching_pairs))
    midpoint = (min_matching + max_non_matching) / 2.0
    if show_sep:
        ax.axvline(x=midpoint, color="red", linestyle="--", label="Separation Point")

    ax.set_xlabel("Inner Product Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return {"min_matching": min_matching, "max_non_matching": max_non_matching, "midpoint": midpoint, "ax": ax}


def plot_margins_vs_relative_bias(
    relative_biases: Iterable[float],
    margins: Iterable[float],
    *,
    ax=None,
    title: str = "Margin Between Matching and Non-matching Pairs",
    xlabel: str = "Relative Bias",
    ylabel: str = "Margin",
    grid: bool = True,
    savepath: Optional[str] = "siglip_margins.png",
):
    ax = _ensure_ax(ax)
    ax.plot(list(relative_biases), list(margins), "o-", linewidth=2, color="green", markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(True)
    if savepath:
        ax.figure.savefig(savepath, dpi=300)
    return ax


def plot_similarities_vs_relative_bias(
    relative_biases: Iterable[float],
    min_matching_sims: Iterable[float],
    max_non_matching_sims: Iterable[float],
    *,
    ax=None,
    title: str = "Similarity Values and Margins",
    xlabel: str = "Relative Bias",
    ylabel: str = "Cosine Similarity",
    fill_between: bool = True,
    grid: bool = True,
    savepath: Optional[str] = "siglip_similarities.png",
):
    ax = _ensure_ax(ax)
    rb = list(relative_biases)
    min_s = list(min_matching_sims)
    max_s = list(max_non_matching_sims)
    ax.plot(rb, min_s, "o-", linewidth=2, label="Min Matching Similarity", markersize=6)
    ax.plot(rb, max_s, "o-", linewidth=2, label="Max Non-matching Similarity", markersize=6)
    if fill_between:
        ax.fill_between(rb, min_s, max_s, alpha=0.2, color="green", label="Margin")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=12)
    if grid:
        ax.grid(True)
    if savepath:
        ax.figure.savefig(savepath, dpi=300)
    return ax

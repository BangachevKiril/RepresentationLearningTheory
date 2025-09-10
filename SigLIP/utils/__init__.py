"""Utility package facade for SigLIP experiments, losses, initialization, and plotting.

Usage examples:
    from utils import SigLIPExperiment, SigLIPLoss
    from utils import generate_class_vectors
    from utils import plot_vectors, plot_losses
"""

from .siglip_experiment import SigLIPExperiment
from .siglip_loss import SigLIPLoss
from .sphere_initialization import (
    generate_class_vectors,
    generate_class_vectors_hemispheres,
)
from .plottingutils import (
    plot_vectors,
    plot_losses,
    plot_final_metric_vs_param,
    plot_inner_product_gap,
    plot_inner_product_gaps_across_sweep,
    plot_margins_vs_relative_bias,
    plot_similarities_vs_relative_bias,
)

__all__ = [
    # Core experiment/loss
    "SigLIPExperiment",
    "SigLIPLoss",
    # Initialization helpers
    "generate_class_vectors",
    "generate_class_vectors_hemispheres",
    # Plotting utilities
    "plot_vectors",
    "plot_losses",
    "plot_final_metric_vs_param",
    "plot_inner_product_gap",
    "plot_inner_product_gaps_across_sweep",
    "plot_margins_vs_relative_bias",
    "plot_similarities_vs_relative_bias",
]

__version__ = "0.1.0"

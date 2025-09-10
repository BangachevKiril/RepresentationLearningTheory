"""Utility package for SigLIP experiments.

Exposes commonly used classes and functions:
- SigLIPExperiment
- SigLIPLoss
- generate_class_vectors, generate_class_vectors_hemispheres
- plot_vectors, plot_inner_product_gap, calculate_margin, analyze_results

Example usage:
    from utils import (
        SigLIPExperiment, SigLIPLoss,
        generate_class_vectors, generate_class_vectors_hemispheres,
        plot_vectors, plot_inner_product_gap, calculate_margin, analyze_results,
    )
"""

from .siglip_experiment import SigLIPExperiment
from .siglip_loss import SigLIPLoss
from .sphere_initialization import (
    generate_class_vectors,
    # generate_class_vectors_hemispheres may not exist in some copies; handle gracefully
)

# Try to expose optional helpers if present
try:
    from .sphere_initialization import generate_class_vectors_hemispheres  # type: ignore
except Exception:  # pragma: no cover - optional
    generate_class_vectors_hemispheres = None  # type: ignore

from .plottingutils import (
    plot_vectors,
    plot_inner_product_gap,
    calculate_margin,
    analyze_results,
)

__all__ = [
    'SigLIPExperiment',
    'SigLIPLoss',
    'generate_class_vectors',
    'generate_class_vectors_hemispheres',
    'plot_vectors',
    'plot_inner_product_gap',
    'calculate_margin',
    'analyze_results',
]

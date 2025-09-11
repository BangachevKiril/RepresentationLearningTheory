## SigLIP Synthetic + Empirical Experiments

This directory contains the experiment code accompanying the paper:

Global Minimizers of Sigmoid Contrastive Loss (Bangachev, Noman, Bresler, Polyanskiy, 2025).

We study the (multi‑modal) sigmoid contrastive loss with *trainable inverse temperature* and *bias / relative bias* parameters. Our results characterize global minimizers, geometric constellation structure, modality gap, and optimization behavior under different parameterizations (absolute bias vs relative bias) and constraints (frozen modalities, adapters, multiple modalities, fixed vs trainable temperature, etc.).

Core prior art: SigLIP (Google DeepMind, 2023), SigLIP2 (2025) and theoretical analysis of fixed‑temp sigmoid loss (Lee et al., 2024). We extend theory to the practically used trainable-parameter setting and validate with controlled synthetic experiments plus pretrained model embeddings.

---
### Repository Layout

```
SigLIP/
	README.md              <- (this file)
	*.ipynb                <- Individual experiment notebooks
	utils/
		siglip_loss.py       <- Sigmoid contrastive loss (trainable T + (relative) bias)
		siglip_experiment.py <- Training harness for synthetic spherical embeddings
		sphere_initialization.py <- Sampling utilities for unit-sphere initialization
		plottingutils.py     <- Shared plotting + analysis helpers
	logs/                  <- Generated figures (saved automatically by notebooks)
```

---
### Conceptual Overview

Let U_i, V_i ∈ S^{d-1} denote paired embeddings across two modalities (can extend to M>2). For a similarity s_{ij} = ⟨U_i, V_j⟩ the SigLIP objective uses a *sigmoid classification* view instead of softmax:

	L = - E[ y_{ij} log σ( T (s_{ij} - rb) ) + (1 - y_{ij}) log (1 - σ( T (s_{ij} - rb) )) ]

where:
* T > 0 is a (trainable) inverse temperature.
* rb is the (trainable) relative bias (our parameterization) OR an absolute bias b (original parameterization). In the absolute bias view logits = T s_{ij} - b; in the relative-bias view logits = T ( s_{ij} - rb ). The transformation b = T * rb couples parameters; decoupling via rb improves conditioning and optimization stability.
* y_{ij}=1 only for matching class pairs (i=j); 0 otherwise (full pairwise matrix supervision in a batch).

Geometric Predictions (validated here):
1. Constellation Structure: At optimum, matched pairs align with elevated cosine similarity while non‑matches remain bounded away, producing a margin that scales with T and rb.
2. Modality Gap: Learned embeddings of different modalities occupy shifted “caps” on the sphere when bias / relative bias adjusts the separating hyperplane.
3. Scaling Laws: Trainable T grows to enlarge margins; with absolute bias, b/T tends toward 0 (degenerate relative bias), while direct rb parameterization keeps rb stable.
4. Multi‑Modal Extension: For M modalities, optimal geometry forms M interleaved constellations with controlled pairwise angular separation depending on a common (T, rb).
5. Frozen / Adapter Setting: Freezing one modality and learning the other + (T, rb) (optionally a low‑rank adapter scalar δ) still attains separation predicted by theory; δ interpolates between alignment strength and orthogonal augmentation.

---
### Provided Utilities

`utils/siglip_loss.py`
	Implements SigLIPLoss with choices:
	- trainable_temp (log‑parameterized) → exp(log_T)
	- relative_bias_parameterization flag: if True learns rb directly; else learns absolute b
	- numerically stable per‑pair binary cross entropy with sigmoid

`utils/siglip_experiment.py`
	- Manages synthetic optimization of U, V (and optionally δ adapter scalar) using Adam.
	- Normalizes embeddings each step (projection onto sphere).
	- Tracks temperature/bias history (when return_t_b_history=True).
	- Supports frozen U (fixed encoder analogue) and optional explicit adapter.

`utils/plottingutils.py`
	- 3D constellation plotting (dim=3) with pair lines.
	- Inner-product gap histograms (match vs non‑match).
	- Margin computation and aggregate multi‑run analysis.

`utils/sphere_initialization.py`
	- Uniform sampling on S^{d-1}; hemisphere variants for controlled separation scenarios.

---
### Experiment Notebooks (Synthetic)

1. `BasicExperiment.ipynb`
	 - Two modalities (U,V), dim=3 for visualization, trainable T, (optionally) trainable rb.
	 - Demonstrates emergence of a clear separation and growth of T; produces Figure 5 (constellation + loss curves + inner-product separation histogram).

2. `MoreModalities.ipynb`
	 - Extends to M ∈ {4,6,8,10} modalities. Joint optimization of all modality embeddings with shared (T, rb).
	 - Compares trainable vs fixed large temperature (ablation). Generates multi‑panel constellations and margin statistics (Figure 8 style) plus loss comparison (`multiplemodalities_loss_comparison.png`).

3. `FrozenModalityExperiments.ipynb`
	 - Freezes one modality (U) to simulate using a locked pretrained encoder while training V and (T, rb).
	 - Optional adapter scalar δ (through an extended embedding dimension) to interpolate geometry; logs δ and demonstrates maintained margin.
	 - Produces figures `frozen_loss_comparison_.png`, `frozenmodalities_ip_separation_.png` etc.

4. `AblationStudy.ipynb`
	 - Systematically varies initial relative bias values and whether temperature is trainable vs fixed large value.
	 - Shows how fixing T alters attainable margin and slows convergence (`ablationfixedlargetemperature.png`, `ablationtrainablelargetemperature.png`).

5. `BiasParamLeadsToZeroRB.ipynb`
	 - Compares absolute bias parameterization vs relative bias parameterization.
	 - Empirically shows b/T → 0 when learning absolute bias, validating preference for direct rb parameterization (Appendix E.4). Figures: `bisavsrelativebias_evolution.png`, `bisavsrelativebias_losses.png`, `bisavsrelativebias_margins.png`.

6. `FixedRelativeBias.ipynb`
	 - Trains embeddings with rb fixed (not trainable) while T is trainable (or vice‑versa in variants) to isolate influence of rb on margin and geometry.

---
### Pretrained Model Embedding Study

`ImageNetEmbedding.ipynb`
	- Downloads a Hugging Face SigLIP checkpoint.
	- Embeds ImageNet validation (50k images, 1k classes) and corresponding text prompts / class names.
	- Analyzes empirical pairwise similarities to verify:
		* Constellation structure across classes.
		* Non‑trivial modality gap (image vs text embedding centroids) — Figure 1 & 3 analogues.
	- Produces margin plots & inner-product distributions (`single_experiment_inner_product_separation.png`, `siglip_similarities.png`, `siglip_margins.png`).

---
### Figures Directory (`logs/`)

Representative saved outputs (non‑exhaustive):
* `basicpicture.png` – Basic 2‑modality constellation.
* `single_experiment_inner_product_separation.png` – Histogram separation.
* `siglip_margins.png` / `siglip_similarities.png` – Margin vs rb and similarity bands.
* `multiplemodalities_ip_separation.png` / `multiplemodalities_loss_comparison.png` – Multi‑modal geometry & optimization.
* `ablationfixedlargetemperature.png` / `ablationtrainablelargetemperature.png` – Ablation on temperature training.
* `bisavsrelativebias_*` – Evolution of relative bias under two parameterizations.
* `frozenmodalities_ip_separation_.png` – Frozen modality adapter study.

---
### Quick Start (Synthetic Experiments)

Dependencies: Python 3.11+, PyTorch, matplotlib, numpy, (optional) Hugging Face transformers + datasets for ImageNet study (you must supply ImageNet or a subset; script expects accessible validation samples / class names).

Minimal installation (example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib numpy transformers datasets tqdm
```

Run a basic synthetic experiment from a Python shell / notebook:
```python
from utils.siglip_experiment import SigLIPExperiment
exp = SigLIPExperiment(n_classes=50, dim=3, n_epochs=20000, return_t_b_history=True)
U,V,crit,losses,temps,rb_hist = exp.train(relative_bias=0.2, temperature=5.0, trainable_temp=True, trainable_bias=True)
```

Plot constellation (dim=3):
```python
exp.plot_vectors(U, V, crit)
```

Plot inner-product gap:
```python
exp.plot_inner_product_gap(U, V)
```

Compute margin:
```python
from utils.plottingutils import calculate_margin
margin, min_match, max_non_match = calculate_margin(U,V)
print(margin, min_match, max_non_match)
```

Frozen modality + adapter example:
```python
U_fixed,_ = generate_class_vectors(100, 3)
exp = SigLIPExperiment(n_classes=100, dim=3, n_epochs=15000, return_t_b_history=True)
U_ext,V_ext,crit,losses,temps,rb_hist,x = exp.train(fixed_U=U_fixed, explicit_adapter=True, initial_x=0.1)
```

---
### Reproducibility Notes
* Random initialization on the sphere → run seeds if you need variance estimates.
* Projection step (renormalization) maintains spherical constraint each iteration.
* Learning rate defaults (1e-2) chosen for stability with Adam; adjust for higher dimensions.
* Numerical stability: small 1e-8 term inside log for BCE; for extreme T growth consider gradient clipping.

---
### Extending to More Modalities
The `MoreModalities.ipynb` notebook shows a pattern: maintain a list of modality embedding Parameter tensors, compute all positive pairs (same index across modalities) vs negatives (all mismatched pairs) and sum per-pair sigmoid BCE. For large M or class count, memory can grow O(M^2 n^2); use minibatching or blockwise evaluation if scaling beyond demonstration sizes.

---
### Key Empirical Takeaways
1. Relative bias parameterization stabilizes optimization and preserves a non‑vanishing rb.
2. Trainable temperature is crucial for achieving large margins; fixing T bottlenecks separation.
3. Adapter scalar δ in frozen modality setting offers a controlled degree of geometric coupling.
4. Multi‑modal extension exhibits predictable widening of non‑match similarity spread but preserves margin scaling with T.
5. Pretrained SigLIP embeddings qualitatively match theoretical constellation + modality gap predictions.

---
### Citation
If you use this code or the synthetic framework, please cite the accompanying paper (bibtex to be provided upon publication).

---
### Contact
Questions / issues: open a GitHub issue or contact the authors.

---
### License
Specify license terms here (e.g., MIT, Apache 2.0) – currently NOT PROVIDED in repo.

---
### Checklist / Future Improvements
* Add reproducible seeds & config loader.
* Add automated unit tests for margin & parameterization invariants.
* Provide benchmarking script for scaling to higher dimensions.
* Integrate WandB / TensorBoard logging.

---
End of README.








# SigLIP Experiments

This directory contains the experiment code accompanying the paper:

Global Minimizers of Sigmoid Contrastive Loss (Bangachev, Noman, Bresler, Polyanskiy, 2025).

We study the sigmoid contrastive loss with *trainable inverse temperature* and *bias / relative bias* parameters. Our results characterize global minimizers, geometric constellation structure, modality gap, and optimization behavior under different parameterizations (absolute bias vs relative bias) and constraints (frozen modalities, adapters, multiple modalities, fixed vs trainable temperature, etc.). Throughout, $t$ denotes inverse temperature, $r_b$ the relative bias, and $b$ an absolute bias when that parameterization is used.

---
### Conceptual Overview

Let $U_i, V_i \in \mathbb{S}^{d-1}$ denote paired embeddings across two modalities (extendable to $M>2$ modalities). Define pairwise similarities
$$s_{ij} = \langle U_i, V_j \rangle.$$
The SigLIP objective treats matching vs non‑matching pairs as a binary classification problem with sigmoid logits. Using the relative bias parameterization (our preferred form) the (per‑batch) loss is

$$\mathcal{L} = - \sum_{i,j} \Big[ y_{ij} \log \sigma\big( t ( s_{ij} - r_b ) \big) + (1-y_{ij}) \log \big( 1 - \sigma\big( t ( s_{ij} - r_b ) \big) \big) \Big],$$

where $y_{ij}=1$ iff $i=j$ (positive pair) and $0$ otherwise. We often normalize by $n^2$; the constant factor is omitted here.

Parameterizations:

* (Relative bias) logits: $ t( s_{ij} - r_b)$ with trainable $t>0$ and $r_b\in \mathbb{R}$.
* (Absolute bias) logits: $ t s_{ij} - b$ with trainable $t>0, b\in \mathbb{R}$.

They are related by $b = t\, r_b$. Directly optimizing $r_b$ avoids the coupling that can drive $b/t \to 0$ empirically.

Margin (reported in figures) between matching and non‑matching similarities:

$$\mathrm{margin} = \frac{ \min_i s_{ii} - \max_{i\neq j} s_{ij} }{2}.$$

In frozen‑modality + adapter experiments we introduce an interpolation scalar $\delta = \sigma(x)$ and extend dimensionality so effective paired vectors become

$$\widetilde U_i = (\delta U_i, \sqrt{1-\delta^2}), \qquad \widetilde V_i = (\delta V_i, -\sqrt{1-\delta^2}),$$

preserving unit norm while controlling alignment strength via $\delta$.

---
### Provided Utilities

`utils/siglip_loss.py`
	Implements SigLIPLoss with choices:
	- trainable_temp (log‑parameterized) → exp(log_t)
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
		- Two modalities $(U,V)$, $d=3$ for visualization, trainable $t$, (optionally) trainable $r_b$.
	 	- Demonstrates emergence of a clear separation and growth of t; produces Figure 5 (constellation + loss curves + inner-product separation histogram).

2. `MoreModalities.ipynb`
		- Extends to $M \in \{4,6,8,10\}$ modalities. Joint optimization of all modality embeddings with shared $(t, r_b)$.
		- Compares trainable vs fixed large temperature (ablation). Constellations and margin statistics (Figure 8 style) and loss comparison (`multiplemodalities_loss_comparison.png`).

3. `FrozenModalityExperiments.ipynb`
		- Freezes one modality ($U$) to simulate using a locked pretrained encoder while training $V$ and $(t, r_b)$.
		- Optional adapter scalar $\delta$ (through an extended embedding dimension) to interpolate geometry; logs $\delta$ and demonstrates maintained margin.
	 - Produces figures `frozen_loss_comparison_.png`, `frozenmodalities_ip_separation_.png` etc.

4. `AblationStudy.ipynb`
		- Systematically varies initial relative bias values and whether temperature is trainable vs fixed large value.
		- Shows how fixing $t$ alters attainable margin and slows convergence (`ablationfixedlargetemperature.png`, `ablationtrainablelargetemperature.png`).

5. `BiasParamLeadsToZeroRB.ipynb`
		- Compares absolute bias parameterization vs relative bias parameterization.
		- Empirically shows $b/t \to 0$ when learning absolute bias, validating preference for direct $r_b$ parameterization (Appendix E.4). Figures: `bisavsrelativebias_evolution.png`, `bisavsrelativebias_losses.png`, `bisavsrelativebias_margins.png`.

6. `FixedRelativeBias.ipynb`
		- Trains embeddings with $r_b$ fixed (not trainable) while $t$ is trainable (or vice‑versa in variants) to isolate influence of $r_b$ on margin and geometry.

---
### Pretrained Model Embedding Study

`ImageNetEmbedding.ipynb`
	- Downloads a Hugging Face SigLIP checkpoint (currently the .2B 'siglip-base-patch16-224' model).
	- Embeds ImageNet validation (50k images, 1k classes) and corresponding text prompts / class names.
	- Analyzes empirical pairwise similarities to verify:
		* Constellation structure across classes - Figure 1.
		* Modality gap — Figure 3 analogues.
	- Produces margin plots & inner-product distributions (`single_experiment_inner_product_separation.png`, `siglip_similarities.png`, `siglip_margins.png`).

---
### Figures Directory (`logs/`)

Representative saved outputs (non‑exhaustive):
* `basicpicture.png` – Basic 2‑modality constellation in 3 dimensions found by running Adam on a random initialization.
* `single_experiment_inner_product_separation.png` – Separation between inner-products of matching an non-matching texts.
* `siglip_margins.png` / `siglip_similarities.png` – Margin vs rb when training with different values for a fixed relative bias.
* `multiplemodalities_ip_separation.png` / `multiplemodalities_loss_comparison.png` – Experiments when training with multiple modalities. 
* `ablationfixedlargetemperature.png` / `ablationtrainablelargetemperature.png` – Ablation study of training with a fixed large temperature instead of a trainable temperature parameter.
* `bisavsrelativebias_*` – Comparision of training with a relative bias parameterizatio versus a bias parameterization.
* `frozenmodalities_ip_separation_.png` Experiments when one modality is locked.

---
### Quick Start (Synthetic Experiments)

Dependencies: Python 3.11+, PyTorch, matplotlib, numpy, Hugging Face transformers + ImageNet validation dataset (you will need the labels file `labels_text.txt').

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
* Learning rate defaults ($1\times 10^{-2}$) chosen for stability with Adam; adjust for higher dimensions.
* Numerical stability: small $10^{-8}$ term inside $\log$ for BCE; for extreme $t$ growth consider gradient clipping.

---
### Extending to More Modalities
The `MoreModalities.ipynb` notebook extends the framewok to more modalities via summing pairwise sigmoid lossses. F

---
### Key Empirical Takeaways
1. Relative bias parameterization stabilizes optimization and preserves a non‑vanishing $r_b$.
2. Trainable temperature is crucial for achieving an inner-product separation with large margins; fixing $t$ bottlenecks separation.
3. Structure is preserved in the case of more than two modalities 
4. Pretrained SigLIP models qualitatively match theoretical constellation + modality gap predictions.

---
### Citation
If you use this code or the synthetic framework, please cite the accompanying paper (bibtex to be provided upon publication).

---
### Contact
Questions / issues: open a GitHub issue or contact the authors.








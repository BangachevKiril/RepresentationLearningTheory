import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .sphere_initialization import generate_class_vectors
from .siglip_loss import SigLIPLoss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from .plottingutils import (
    plot_vectors as _plot_vectors,
    plot_losses as _plot_losses,
    plot_final_metric_vs_param as _plot_final_metric_vs_param,
    plot_inner_product_gaps_across_sweep as _plot_inner_product_gaps_across_sweep,
    plot_inner_product_gap as _plot_inner_product_gap,
    plot_margins_vs_relative_bias as _plot_margins_vs_relative_bias,
    plot_similarities_vs_relative_bias as _plot_similarities_vs_relative_bias,
)

class SigLIPExperiment:
    def __init__(self, n_classes=100, dim=3, n_epochs=int(5e4), device=None, when_to_print=100000):
        self.n_classes = n_classes
        self.dim = dim
        self.n_epochs = n_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.U = None
        self.V = None
        self.when_to_print = when_to_print

    def train(self,
              relative_bias: float,
              temperature: float = 10.0,
              trainable_bias: bool = False,
              trainable_temp: bool = True,
              fixed_U: torch.Tensor = None,
              initial_x: float = 0.0,
              lr: float = 1e-2):
        """
        If `fixed_U` is not None, we:
          - normalize & freeze it as self.U
          - introduce a scalar x (initialized to initial_x) so that δ = sigmoid(x)
          - only train x and V (plus any bias/temperature in the criterion)
        Otherwise we train both U and V as before.
        """

        if fixed_U is not None:
            U0 = fixed_U.to(self.device)
            U0 = U0 / U0.norm(dim=1, keepdim=True)
            self.U = nn.Parameter(U0, requires_grad=False)

            V0 = torch.randn(self.n_classes, self.dim, device=self.device)
            V0 = V0 / V0.norm(dim=1, keepdim=True)
            self.V = nn.Parameter(V0, requires_grad=True)

            self.x = nn.Parameter(torch.tensor(initial_x, device=self.device))
        else:
            U_init, V_init = generate_class_vectors(self.n_classes, self.dim, self.device)
            self.U = nn.Parameter(U_init / torch.norm(U_init, dim=1, keepdim=True))
            self.V = nn.Parameter(V_init / torch.norm(V_init, dim=1, keepdim=True))
            self.x = None

        criterion = SigLIPLoss(
            temperature=temperature,
            relative_bias=relative_bias,
            trainable_temp=trainable_temp,
            trainable_bias=trainable_bias
        ).to(self.device)

        params = [{'params': self.V, 'lr': lr},
                  *([{'params': [self.x], 'lr': lr}] if self.x is not None else [{'params': [self.U], 'lr': lr}]),
                  {'params': criterion.parameters(), 'lr': lr}]
        optimizer = torch.optim.Adam(params)

        losses = []
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # if we're in the "fixed_U" regime, build the extended vectors
            if self.x is not None:
                delta = torch.sigmoid(self.x)
                extra = torch.sqrt(1 - delta**2)
                extra_col = extra.expand(self.n_classes, 1)
                U_ext = torch.cat([delta * self.U, extra_col], dim=1)
                V_ext = torch.cat([delta * self.V, -extra_col], dim=1)
                loss = criterion(U_ext, V_ext)
            else:
                loss = criterion(self.U, self.V)


            loss.backward()
            optimizer.step()

            # re‐project U,V back onto unit‐sphere
            with torch.no_grad():
                if self.x is None: 
                    self.U.data = self.U.data / self.U.data.norm(dim=1, keepdim=True)
                self.V.data = self.V.data / self.V.data.norm(dim=1, keepdim=True)
                # print(self.U.data.norm(dim =1, keepdim=True))

            losses.append(loss.item())

            if (epoch + 1) % self.when_to_print == 0:
                tb = criterion.get_temperature().item()
                rb = criterion.get_bias().item()
                if self.x is not None:
                    print(f"[{epoch+1}/{self.n_epochs}]  "
                          f"loss={loss:.4f}  δ={delta:.4f}  T={tb:.4f}  rb={rb:.4f}")
                else:
                    print(f"[{epoch+1}/{self.n_epochs}]  "
                          f"loss={loss:.4f}  T={tb:.4f}  rb={rb:.4f}")
        if self.x is None:
            return self.U, self.V, criterion, losses
        else:
            delta = torch.sigmoid(self.x)
            extra = torch.sqrt(1 - delta**2)
            extra_col = extra.expand(self.n_classes, 1)
            U_ext = torch.cat([delta * self.U, extra_col], dim=1)
            V_ext = torch.cat([delta * self.V, -extra_col], dim=1)
            return U_ext, V_ext, criterion, losses, self.x
    
    def plot_vectors(self, U, V, criterion, ax=None, title=None, **kwargs):
        """Delegate to plottingutils.plot_vectors for 3D vector visualization.
        Accepts optional title and extra plotting kwargs (e.g., indices, colors).
        """
        ax = _plot_vectors(U, V, criterion=criterion, ax=ax, title=title, **kwargs)
        plt.show()
        return ax
    
    def run_sweep(self, sweep_param, values, **train_kwargs):
        results = []
        self.all_UV = []  # reset storage for this sweep

        for val in values:
            print(f"\n=== Running {sweep_param}={val} ===")

            train_args = {**train_kwargs, sweep_param: val}
            train_out = self.train(**train_args)

            if len(train_out) == 4:
                U, V, criterion, losses = train_out
            else:
                U, V, criterion, losses, x = train_out

            margin, min_match, max_non_match = self.calculate_margin(U, V)

            results.append({
                sweep_param: val,
                "final_loss": losses[-1],
                "final_temp": criterion.get_temperature().item(),
                "final_bias": criterion.get_bias().item(),
                "margin": margin,
                "min_matching": min_match,
                "max_non_matching": max_non_match,
            })

            # store U, V for later visualization
            self.all_UV.append((U.detach().cpu(), V.detach().cpu()))

        df = pd.DataFrame(results)
        self.sweep_results = df
        return df


    def plot_losses(self, losses, title="Training Loss Curve"):
        """Delegate to plottingutils.plot_losses to display the loss curve."""
        _plot_losses(losses, title=title)
        plt.show()

    def plot_final_metric_vs_param(self, param_name, metric_name):
        """Delegate to plottingutils.plot_final_metric_vs_param using stored sweep_results."""
        if not hasattr(self, "sweep_results"):
            raise ValueError("No sweep results found. Run run_sweep() first.")
        _plot_final_metric_vs_param(self.sweep_results, param_name, metric_name)
        plt.show()

    def plot_inner_product_gaps_across_sweep(self, all_U, all_V, sweep_param, values):
        """Delegate to plottingutils.plot_inner_product_gaps_across_sweep."""
        _plot_inner_product_gaps_across_sweep(all_U, all_V, sweep_param, values)
        plt.tight_layout()
        plt.show()

    def plot_inner_product_gap(self, U_final, V_final):
        """Delegate to plottingutils.plot_inner_product_gap."""
        _plot_inner_product_gap(U_final, V_final)
        plt.show()


    def calculate_margin(self, U, V):
        """
        Calculate the margin between matching and non-matching pairs.
        
        Returns:
            tuple: (margin, min_matching_sim, max_non_matching_sim)
        """
        # Calculate all pairwise cosine similarities
        cosine_sim = torch.matmul(U, V.t())
        
        # Get diagonals (matching pairs)
        diag_indices = torch.arange(U.shape[0], device=U.device)
        matching_sims = cosine_sim[diag_indices, diag_indices]
        min_matching_sim = torch.min(matching_sims).item()
        
        # Create mask to exclude diagonal elements
        mask = torch.ones_like(cosine_sim, dtype=torch.bool)
        mask[diag_indices, diag_indices] = False
        
        # Get maximum non-matching similarity
        non_matching_sims = cosine_sim[mask]
        max_non_matching_sim = torch.max(non_matching_sims).item()
        
        # Calculate margin
        margin = (min_matching_sim - max_non_matching_sim)/2
        
        return margin, min_matching_sim, max_non_matching_sim
    
    def analyze_results(self, all_results, relative_biases):
        """Compute metrics and use plottingutils to render summary plots."""
        margins = []
        min_matching_sims = []
        max_non_matching_sims = []
        final_temps = []

        for U, V, criterion, _ in all_results:
            margin, min_match, max_non_match = self.calculate_margin(U, V)
            margins.append(margin)
            min_matching_sims.append(min_match)
            max_non_matching_sims.append(max_non_match)
            final_temps.append(criterion.get_temperature())

        _plot_margins_vs_relative_bias(relative_biases, margins)
        plt.show()
        _plot_similarities_vs_relative_bias(relative_biases, min_matching_sims, max_non_matching_sims)
        plt.show()

        return {
            "margins": margins,
            "min_matching_sims": min_matching_sims,
            "max_non_matching_sims": max_non_matching_sims,
            "final_temps": final_temps,
        }
        
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sphere_initialization import generate_class_vectors
from siglip_loss import SigLIPLoss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class SigLIPExperiment:
    def __init__(self, n_classes=100, dim=3, n_epochs=int(5e4), device=None):
        self.n_classes = n_classes
        self.dim = dim
        self.n_epochs = n_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.U = None
        self.V = None

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

            if (epoch + 1) % 100 == 0:
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
    
    def plot_vectors(self, U, V, criterion, ax=None):
        """
        Plot the optimized vectors on a unit sphere.
        
        Args:
            U (torch.Tensor): U vectors
            V (torch.Tensor): V vectors
            criterion (SigLIPLoss): Loss function to get parameters
            ax (matplotlib.axes.Axes, optional): Axis to plot on
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        U_np = U.detach().cpu().numpy()
        V_np = V.detach().cpu().numpy()
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
        
        # Plot points
        ax.scatter(U_np[:, 0], U_np[:, 1], U_np[:, 2], c='blue', s=20, label='U vectors')
        ax.scatter(V_np[:, 0], V_np[:, 1], V_np[:, 2], c='red', s=20, label='V vectors')
        
        # Draw lines between corresponding vectors
        for i in range(self.n_classes):
            ax.plot([U_np[i, 0], V_np[i, 0]], 
                    [U_np[i, 1], V_np[i, 1]], 
                    [U_np[i, 2], V_np[i, 2]], 'k--', alpha=0.2)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        temp = criterion.get_temperature()
        relative_bias = criterion.get_bias()
        title = f'rb={relative_bias:.2f}, T={temp:.2f}'
        ax.set_title(title)
        
        ax.set_box_aspect([1, 1, 1])
        
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
        plt.figure(figsize=(8, 5))
        plt.plot(losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def plot_final_metric_vs_param(self, param_name, metric_name):
        """
        Plot the relationship between a swept parameter and a final metric.

        Args:
            param_name (str): the swept parameter (must exist in self.sweep_results)
            metric_name (str): the metric to plot (must exist in self.sweep_results)
        """
        if not hasattr(self, "sweep_results"):
            raise ValueError("No sweep results found. Run run_sweep() first.")

        df = self.sweep_results
        plt.figure(figsize=(8, 5))
        plt.plot(df[param_name], df[metric_name], "o-", linewidth=2)
        plt.xlabel(param_name.capitalize())
        plt.ylabel(metric_name.replace("_", " ").capitalize())
        plt.title(f"{metric_name.replace('_', ' ').capitalize()} vs {param_name}")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_inner_product_gaps_across_sweep(self, all_U, all_V, sweep_param, values):
        """
        Plot inner product histograms for each sweep value side by side.

        Args:
            all_U, all_V: list of U and V tensors from multiple runs
            sweep_param (str): the parameter that was swept
            values (list): list of sweep values
        """
        n = len(values)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, (U, V, val) in enumerate(zip(all_U, all_V, values)):
            plt.sca(axes[i])  # set current axis
            inner_products = torch.matmul(U, V.t())

            matching = torch.diag(inner_products).cpu().numpy()
            mask = ~torch.eye(U.shape[0], dtype=bool, device=U.device)
            non_matching = inner_products[mask].cpu().numpy()

            plt.hist(matching, bins=15, alpha=0.5, label="Matching", color="blue", density=True)
            plt.hist(non_matching, bins=15, alpha=0.5, label="Non-matching", color="green", density=True)

            plt.title(f"{sweep_param}={val}")
            plt.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_inner_product_gap(self, U_final, V_final):
        inner_products = torch.matmul(U_final, V_final.t())

        # Get matching pairs (diagonal elements)
        matching_pairs = torch.diag(inner_products).detach().cpu().numpy()

        # Get non-matching pairs (off-diagonal elements)
        mask = ~torch.eye(self.n_classes, dtype=bool, device=self.device)
        non_matching_pairs = inner_products[mask].detach().cpu().numpy()

        # Create histogram plot
        plt.figure(figsize=(10, 6))
        plt.hist(matching_pairs, bins= 15, alpha=0.5, label='Matching pairs (U_i, V_i)', color='blue', density=True, log = True)
        plt.hist(non_matching_pairs, bins=15, alpha=0.5, label='Non-matching pairs (U_i, V_j)', color='green', density=True, log = True)

        # Add red line showing separation between max non-matching and min matching
        min_matching = np.min(matching_pairs)
        max_non_matching = np.max(non_matching_pairs)
        midpoint = (min_matching + max_non_matching) / 2
        plt.axvline(x=midpoint, color='red', linestyle='--', label='Separation Point')

        plt.xlabel('Inner Product Value')
        plt.ylabel('Density')
        plt.title('Distribution of Inner Products (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
            margin, min_match, max_non_match = self.calculate_margin(U, V)
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
        # plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        # plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
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
        # plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('siglip_similarities.png', dpi=300)
        plt.show()
        
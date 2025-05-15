import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sphere_initialization import generate_class_vectors
from siglip_loss import SigLIPLoss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

class SigLIPExperiment:
    def __init__(self, n_classes=100, dim=3, n_epochs=int(5e4), device=None, when_to_print =100,
                 relative_bias_parameterization = True, return_t_b_history = False):
        self.n_classes = n_classes
        self.dim = dim
        self.n_epochs = n_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.U = None
        self.V = None
        self.when_to_print = when_to_print
        self.relative_bias_parameterization = relative_bias_parameterization
        self.return_t_b_history = return_t_b_history

    def train(self,
              relative_bias: float = 0.0,
              bias: float = 0.0,
              temperature: float = 10.0,
              trainable_bias: bool = False,
              trainable_temp: bool = True,
              fixed_U: torch.Tensor = None,
              explicit_adapter: bool = False,
              initial_x: float = 0.0,
              lr: float = 1e-2):
        """
        If `fixed_U` is not None, we:
          - normalize & freeze it as self.U
          - introduce a scalar x (initialized to initial_x) so that δ = sigmoid(x)
          - only train x and V (plus any bias/temperature in the criterion)
        Otherwise we train both U and V as before.
        """

        if (fixed_U is not None) and explicit_adapter:
            U0 = fixed_U.to(self.device)
            U0 = U0 / U0.norm(dim=1, keepdim=True)
            self.U = nn.Parameter(U0, requires_grad=False)

            V0 = torch.randn(self.n_classes, self.dim, device=self.device)
            V0 = V0 / V0.norm(dim=1, keepdim=True)
            self.V = nn.Parameter(V0, requires_grad=True)

            self.x = nn.Parameter(torch.tensor(initial_x, device=self.device))
        elif fixed_U is not None:
            U0 = fixed_U.to(self.device)
            U0 = U0 / U0.norm(dim=1, keepdim=True)
            self.U = nn.Parameter(U0, requires_grad=False)

            V0 = torch.randn(self.n_classes, self.dim, device=self.device)
            V0 = V0 / V0.norm(dim=1, keepdim=True)
            self.V = nn.Parameter(V0, requires_grad=True)
            self.x = None
        else:
            U_init, V_init = generate_class_vectors(self.n_classes, self.dim, self.device)
            self.U = nn.Parameter(U_init / torch.norm(U_init, dim=1, keepdim=True))
            self.V = nn.Parameter(V_init / torch.norm(V_init, dim=1, keepdim=True))
            self.x = None

        criterion = SigLIPLoss(
            temperature=temperature,
            relative_bias=relative_bias,
            trainable_temp=trainable_temp,
            trainable_bias=trainable_bias,
            relative_bias_parameterization=self.relative_bias_parameterization,
            bias = bias
        ).to(self.device)

        params = [{'params': self.V, 'lr': lr},
                  *([{'params': [self.x], 'lr': lr}] if self.x is not None else [{'params': [self.U], 'lr': lr}]),
                  {'params': criterion.parameters(), 'lr': lr}]
        optimizer = torch.optim.Adam(params)

        losses = []
        relativebiases = []
        temperatures = []
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

            if self.return_t_b_history:
                tb = criterion.get_temperature()
                rb = criterion.get_bias()
                if not self.relative_bias_parameterization:
                    rb = rb / tb
                relativebiases.append(rb)
                temperatures.append(tb)
            if (epoch + 1) % self.when_to_print == 0:
                tb = criterion.get_temperature()
                rb = criterion.get_bias()
                if not self.relative_bias_parameterization:
                    rb = rb / tb
                if self.x is not None:
                    print(f"[{epoch+1}/{self.n_epochs}]  "
                          f"loss={loss:.4f}  δ={delta:.4f}  T={tb:.4f}  rb={rb:.4f}")
                else:
                    print(f"[{epoch+1}/{self.n_epochs}]  "
                          f"loss={loss:.4f}  T={tb:.4f}  rb={rb:.4f}")
        if not self.return_t_b_history:
            if self.x is None:
                return self.U, self.V, criterion, losses
            else:
                delta = torch.sigmoid(self.x)
                extra = torch.sqrt(1 - delta**2)
                extra_col = extra.expand(self.n_classes, 1)
                U_ext = torch.cat([delta * self.U, extra_col], dim=1)
                V_ext = torch.cat([delta * self.V, -extra_col], dim=1)
                return U_ext, V_ext, criterion, losses, self.x
        else:
            if self.x is None:
                return self.U, self.V, criterion, losses, temperatures,relativebiases
            else:
                delta = torch.sigmoid(self.x)
                extra = torch.sqrt(1 - delta**2)
                extra_col = extra.expand(self.n_classes, 1)
                U_ext = torch.cat([delta * self.U, extra_col], dim=1)
                V_ext = torch.cat([delta * self.V, -extra_col], dim=1)
                return U_ext, V_ext, criterion, losses, temperatures, relativebiases, self.x
    
    def plot_vectors(self, U, V, criterion, ax=None,plot_grid = False,title = None):
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
        if plot_grid:
            ax.set_xlabel('X', fontsize = 16)
            ax.set_ylabel('Y', fontsize = 16)
            ax.set_zlabel('Z', fontsize = 16)
        
        temp = criterion.get_temperature()
        relative_bias = criterion.get_bias()
        if title is None:
            title = f'rb={relative_bias:.2f}, T={temp:.2f}'
            ax.set_title(title, fontsize = 16)
        else:
            ax.set_title(title, fontsize = 16)
        
        ax.set_box_aspect([1, 1, 1])
        ax.grid(plot_grid)
        ax.set_axis_off()
        ax.legend(fontsize = 16)

        return ax
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
        plt.hist(non_matching_pairs, bins=15, alpha=0.5, label='Non-matching pairs (U_i, V_j)', color='red', density=True, log = True)

        # Add red line showing separation between max non-matching and min matching
        min_matching = np.min(matching_pairs)
        max_non_matching = np.max(non_matching_pairs)
        midpoint = (min_matching + max_non_matching) / 2
        plt.axvline(x=midpoint, color='red', linestyle='--', label='Separation Point')

        plt.xlabel('Inner Product Value',fontsize= 16)
        plt.ylabel('Density',fontsize= 16)
        plt.title('Distribution of Inner Products (Normalized)',fontsize= 20)
        plt.legend(fontsize= 16)
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
        
        # # Plot temperatures
        # plt.figure(figsize=(12, 8))
        # plt.plot(relative_biases, final_temps, 'o-', linewidth=2, color='purple', markersize=8)
        # plt.xlabel('Relative Bias', fontsize=14)
        # plt.ylabel('Final Temperature', fontsize=14)
        # plt.title('Final Temperature vs Relative Bias', fontsize=16)
        # plt.grid(True)
        # plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # plt.savefig('siglip_temperatures.png', dpi=300)
        # plt.show() 
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
    def __init__(self, n_classes=100, dim=3, n_epochs=int(5e4), device=None):
        """
        Initialize a SigLIP experiment.
        
        Args:
            n_classes (int): Number of classes/vector pairs
            dim (int): Dimension of the vectors
            n_epochs (int): Number of training epochs
            device (str): Device to run the experiment on ('cuda' or 'cpu')
        """
        self.n_classes = n_classes
        self.dim = dim
        self.n_epochs = n_epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize vectors
        self.U_init, self.V_init = generate_class_vectors(n_classes, dim, self.device)
        self.U = nn.Parameter(self.U_init / torch.norm(self.U_init, dim=1, keepdim=True))
        self.V = nn.Parameter(self.V_init / torch.norm(self.V_init, dim=1, keepdim=True))
        
    def train(self, relative_bias, temperature=10.0, trainable_bias=False):
        """
        Train the model with specified parameters.
        
        Args:
            relative_bias (float): Initial relative bias value
            temperature (float): Initial temperature value
            trainable_bias (bool): Whether to make bias trainable
            
        Returns:
            tuple: (U, V, criterion, losses)
        """
        # Initialize loss function
        criterion = SigLIPLoss(temperature=temperature, 
                             relative_bias=relative_bias, 
                             trainable_bias=trainable_bias).to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': self.U, 'lr': 0.01},
            {'params': self.V, 'lr': 0.01},
            {'params': criterion.parameters(), 'lr': 0.01}
        ])
        
        # Training loop
        losses = []
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            loss = criterion(self.U, self.V)
            loss.backward()
            optimizer.step()
            
            # Project back onto unit sphere
            with torch.no_grad():
                self.U.data = self.U.data / torch.norm(self.U.data, dim=1, keepdim=True)
                self.V.data = self.V.data / torch.norm(self.V.data, dim=1, keepdim=True)
            
            losses.append(loss.item())
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}, '
                      f'Temperature: {criterion.get_temperature():.4f}, '
                      f'Relative bias: {criterion.get_bias():.4f}')
        
        return self.U, self.V, criterion, losses
    
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
        
        # Move vectors to CPU and convert to numpy
        U_np = U.detach().cpu().numpy()
        V_np = V.detach().cpu().numpy()
        
        # Plot unit sphere wireframe
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
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .sphere_initialization import generate_class_vectors
from .siglip_loss import SigLIPLoss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from .plottingutils import (
    plot_vectors as plot_vectors_util,
    plot_inner_product_gap as plot_inner_product_gap_util,
    calculate_margin as calculate_margin_util,
    analyze_results as analyze_results_util,
)

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
                tb = float(criterion.get_temperature())
                rb = float(criterion.get_bias())
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
                return self.U, self.V, criterion, losses, temperatures, relativebiases
            else:
                delta = torch.sigmoid(self.x)
                extra = torch.sqrt(1 - delta**2)
                extra_col = extra.expand(self.n_classes, 1)
                U_ext = torch.cat([delta * self.U, extra_col], dim=1)
                V_ext = torch.cat([delta * self.V, -extra_col], dim=1)
                return U_ext, V_ext, criterion, losses, temperatures, relativebiases, self.x
    
    def plot_vectors(self, U, V, criterion, ax=None, plot_grid=False, title=None):
        # Delegate to plotting utility to keep backward compatibility
        return plot_vectors_util(U, V, criterion, ax=ax, plot_grid=plot_grid, title=title)
    def plot_inner_product_gap(self, U_final, V_final):
        # Delegate to plotting utility
        return plot_inner_product_gap_util(U_final, V_final)


    def calculate_margin(self, U, V):
        return calculate_margin_util(U, V)
    
    def analyze_results(self, all_results, relative_biases):
        return analyze_results_util(all_results, relative_biases)
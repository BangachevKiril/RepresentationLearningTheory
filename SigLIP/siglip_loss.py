import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """
    SigLIP (Sigmoid Loss for Vector Pairs) loss function.
    
    This loss function is designed to align pairs of vectors using a sigmoid-based approach.
    It includes trainable temperature and bias parameters.
    
    Args:
        temperature (float, optional): Initial temperature parameter. Defaults to 1.0.
        relative_bias (float, optional): Initial relative_bias parameter. Defaults to 0.0.
        trainable_temp (bool, optional): Whether to make temperature trainable. Defaults to True.
        trainable_bias (bool, optional): Whether to make realtive_bias is trainable. Defaults to True.
    """
    def __init__(self, temperature=1.0, relative_bias=0.0, trainable_temp=True, trainable_bias=True):
        super().__init__()
        
        # Initialize temperature and bias as parameters or buffers
        log_temp = torch.log(torch.tensor(temperature, dtype=torch.float32))
        rel_bias = torch.tensor(relative_bias, dtype=torch.float32)
        
        if trainable_temp:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer('log_temperature', log_temp)
            
        if trainable_bias:
            self.relative_bias = nn.Parameter(rel_bias)
        else:
            self.register_buffer('relative_bias', rel_bias)
    def forward(self, u, v, labels=None):
        """
        Compute the SigLIP loss.
        
        Args:
            u (torch.Tensor): First set of vectors of shape [batch_size, embedding_dim]
            v (torch.Tensor): Second set of vectors of shape [batch_size, embedding_dim]
            labels (torch.Tensor, optional): Labels indicating positive pairs. If None, assumes diagonal pairs are positive.
                                           Shape [batch_size, batch_size] or [batch_size].
        
        Returns:
            torch.Tensor: The computed loss value
        """
        # Normalize features
        u = F.normalize(u, dim=1)
        v = F.normalize(v, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(u, v.t())
        
        # Apply temperature and bias
        logits = similarity * torch.exp(self.log_temperature) - self.relative_bias * torch.exp(self.log_temperature)
        
        # If labels are not provided, assume diagonal pairs are positive
        if labels is None:
            labels = torch.eye(u.size(0), device=u.device)
        
        sigmoid_logits = torch.sigmoid(logits)
        
        pos_loss = -torch.log(sigmoid_logits + 1e-8) * labels

        neg_loss = -torch.log(1 - sigmoid_logits + 1e-8) * (1 - labels)
        
        loss = (pos_loss + neg_loss).mean() 
        
        return loss
    
    def get_temperature(self):
        """Return the current temperature value."""
        return torch.exp(self.log_temperature)
    
    def get_bias(self):
        """Return the current bias as a Tensor (0-dim)."""
        # Ensure a Tensor is returned for consistency with get_temperature
        if isinstance(self.relative_bias, torch.Tensor):
            return self.relative_bias
        return torch.tensor(self.relative_bias, dtype=torch.float32)
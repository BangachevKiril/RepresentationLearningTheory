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
        
        # Initialize temperature and bias as parameters if they should be trainable
        if trainable_temp:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature, dtype=torch.float32)))
        else:
            self.log_temperature = torch.log(torch.tensor(temperature, dtype=torch.float32))
            
        if trainable_bias:
            self.relative_bias = nn.Parameter(torch.tensor(relative_bias, dtype=torch.float32))
        else:
            self.relative_bias = torch.tensor(relative_bias, dtype=torch.float32)
    
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
        """Return the current bias value."""
        return self.relative_bias.item() if isinstance(self.relative_bias, nn.Parameter) else self.relative_bias 
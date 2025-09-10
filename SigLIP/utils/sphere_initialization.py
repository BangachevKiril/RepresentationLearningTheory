import torch
import numpy as np

def generate_uniform_on_sphere(num_points, dim, device='cpu'):
    """
    Generate points uniformly distributed on a unit sphere in R^dim.
    
    Args:
        num_points (int): Number of points to generate
        dim (int): Dimension of the space
        device (str): Device to place the tensor on ('cpu' or 'cuda')
        
    Returns:
        torch.Tensor: Tensor of shape [num_points, dim] with points uniformly distributed on the unit sphere
    """
    # Generate points from a standard normal distribution
    try:
        points = torch.randn(size=(num_points, dim), device=device)
    except:
        print(f"Error generating {num_points} points in {dim} dimensions on {device}")
    
    # Normalize to get points on the unit sphere
    points = points / torch.norm(points, dim=1, keepdim=True)
    
    return points

def generate_class_vectors(num_classes, dim, device='cpu'):
    """
    Generate pairs of vectors for classes where each class has a corresponding U and V vector.
    All vectors are uniformly distributed on the unit sphere.
    
    Args:
        num_classes (int): Number of classes (each class will have 2 vectors)
        dim (int): Dimension of the vectors
        device (str): Device to place the tensors on ('cpu' or 'cuda')
        
    Returns:
        tuple: (U, V) where U and V are tensors of shape [num_classes, dim] with points uniformly distributed on the unit sphere
    """
    U = generate_uniform_on_sphere(num_classes, dim, device)
    V = generate_uniform_on_sphere(num_classes, dim, device)
    
    return U, V

def generate_class_vectors_hemispheres(num_classes, dim, device='cpu'):
    """
    Generate pairs of vectors for classes where each class has a corresponding U vector in one hemisphere
    and a V vector in the opposite hemisphere. All vectors are iid uniformly distributed.
    
    Args:
        num_classes (int): Number of classes (each class will have 2 vectors)
        dim (int): Dimension of the vectors
        device (str): Device to place the tensors on ('cpu' or 'cuda')
        
    Returns:
        tuple: (U, V) where U and V are tensors of shape [num_classes, dim] with U in one hemisphere and V in the other.
    """
    U = generate_uniform_on_sphere(num_classes, dim, device)
    V = generate_uniform_on_sphere(num_classes, dim, device)
    
    U[0, :] = torch.abs(U[0, :])
    V[0, :] = -torch.abs(V[0, :])
    
    return U, V
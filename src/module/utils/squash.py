import torch

def squash(x, dim=-1):
    squared = torch.sum(x * x, dim=dim, keepdim=True)
    scale = torch.sqrt(squared) / (1.0 + squared)
    return scale * x
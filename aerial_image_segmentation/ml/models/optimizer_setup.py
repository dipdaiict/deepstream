import torch
from torch.optim import Optimizer

def create_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
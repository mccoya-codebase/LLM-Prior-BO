"""
Prior module for PiBO: maps OHE-encoded points to prior membership
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

class PriorModule(nn.Module):
    """
    Decodes one-hot encoded points back to parameters
    """

    def __init__(
        self,
        prior_set: List[Tuple[int, ...]],
        param_values: List[List[str]],
    ):
        super().__init__()
        self.prior_set = set(prior_set)
        self.param_values = param_values
        
        # Calculate offset
        param_sizes = [len(v) for v in param_values]
        self.offsets = [0] + list(np.cumsum(param_sizes[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        output = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        for i in range(x.shape[0]):
            for q in range(x.shape[1]):
                indices = torch.where(x[i, q, :] > 0.5)[0]
                if len(indices) != len(self.offsets):
                    output[i, q] = 1.0
                    continue
                    
                label_list = []
                for j in range(len(self.offsets)):
                    pos_idx = int(indices[j].item() - self.offsets[j])
                    actual_val = int(self.param_values[j][pos_idx])
                    label_list.append(actual_val)

                output[i, q] = 2.0 if tuple(label_list) in self.prior_set else 1.0
                
        return output


from typing import Sequence

import numpy as np
import torch


def save_correlation_matrix(matrix: torch.Tensor, path: str) -> None:
    
    np.save(path, matrix.detach().cpu().numpy())
    print(f"[ablation] Saved correlation matrix to {path}")


def load_correlation_matrix(path: str) -> np.ndarray:
    
    return np.load(path)


def average_correlation(matrices: Sequence[torch.Tensor]) -> np.ndarray:
    
    stacked = torch.stack([m.detach().float() for m in matrices], dim=0)
    return stacked.mean(dim=0).cpu().numpy()

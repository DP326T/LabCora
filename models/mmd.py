from typing import Iterable, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma_list: Iterable[float]) -> torch.Tensor:
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    dist = torch.pow(x - y, 2).sum(2)

    k_xy = torch.zeros_like(dist)
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        k_xy = k_xy + torch.exp(-gamma * dist)
    return k_xy


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_list: List[float] = (1.0, 5.0, 10.0, 20.0),
) -> torch.Tensor:
    k_xx = rbf_kernel(x, x, sigma_list)
    k_yy = rbf_kernel(y, y, sigma_list)
    k_xy = rbf_kernel(x, y, sigma_list)
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return torch.sqrt(torch.abs(mmd))


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> torch.Tensor:
    backbone.eval()
    feats = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        _, features, _ = backbone(x)
        feats.append(features)
    if not feats:
        raise RuntimeError("No features extracted; loader was empty.")
    return torch.cat(feats, dim=0)


def evaluate_domain_mmd(
    backbone: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    sigma_list: List[float] = (1.0, 5.0, 10.0, 20.0),
) -> float:
    features_l = extract_features(backbone, labeled_loader, device, num_batches)
    features_u = extract_features(backbone, unlabeled_loader, device, num_batches)
    return float(compute_mmd(features_l, features_u, sigma_list).item())

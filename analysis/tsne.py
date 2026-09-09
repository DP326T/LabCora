

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3D projection
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


@torch.no_grad()
def _collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> torch.Tensor:
    model.eval()
    feats = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        _, features, _ = model(x)
        feats.append(features.cpu().numpy())
    return np.concatenate(feats, axis=0)


def visualize_3d_tsne(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    device: torch.device,
    save_path: str = "tsne_alignment.png",
    max_batches: int = 20,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    
    feats_l = _collect_features(model, labeled_loader, device, max_batches)
    feats_u = _collect_features(model, unlabeled_loader, device, max_batches)
    all_feats = np.concatenate([feats_l, feats_u], axis=0)
    n_l = len(feats_l)

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
    )
    embedded = tsne.fit_transform(all_feats)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        embedded[:n_l, 0], embedded[:n_l, 1], embedded[:n_l, 2],
        c="tab:blue", s=8, alpha=0.6, label="Labelled",
    )
    ax.scatter(
        embedded[n_l:, 0], embedded[n_l:, 1], embedded[n_l:, 2],
        c="tab:orange", s=8, alpha=0.6, label="Unlabelled",
    )
    ax.set_title("3D t-SNE of Labelled vs Unlabelled Features")
    ax.legend(loc="best")

    # Make sure the legend renders above the scatter points.
    legend = ax.get_legend()
    if legend is not None:
        legend.set_zorder(100)
        frame = legend.get_frame()
        frame.set_facecolor("none")
        frame.set_edgecolor("black")
        frame.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"[visualize_3d_tsne] Saved {save_path}")

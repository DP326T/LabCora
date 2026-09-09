

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    
    model.eval()
    all_probs = []
    all_preds = []
    all_y = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_np = labels.cpu().numpy()
        logits, _, _ = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(np.float32)

        all_probs.append(probs)
        all_preds.append(preds)
        all_y.append(labels_np)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # ----- Per-class AUC ----------------------------------------------------
    per_class_auc: List[float] = []
    for c in range(num_classes):
        y_col = all_y[:, c]
        # Skip classes that have no positive examples — AUC is undefined.
        if y_col.sum() == 0 or y_col.sum() == len(y_col):
            continue
        try:
            per_class_auc.append(roc_auc_score(y_col, all_probs[:, c]))
        except ValueError:
            continue
    mean_auc = float(np.mean(per_class_auc)) if per_class_auc else 0.0

    # ----- Flat metrics -----------------------------------------------------
    valid_mask = ((all_y == 0.0) | (all_y == 1.0)).astype(bool)
    flat_labels = all_y[valid_mask]
    flat_preds = all_preds[valid_mask]
    flat_probs = all_probs[valid_mask]

    if flat_labels.size > 0:
        f1 = float(f1_score(flat_labels, flat_preds, average="macro"))
        map_score = float(
            average_precision_score(flat_labels, flat_probs, average="macro")
        )
        hamming = float(hamming_loss(flat_labels, flat_preds))
    else:
        f1, map_score, hamming = 0.0, 0.0, 1.0

    return {
        "f1": f1,
        "auc": mean_auc,
        "mAP": map_score,
        "hamming": hamming,
        "per_class_auc": per_class_auc,
    }

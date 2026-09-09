

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.cotraining import CoTrainingModel


def supervised_warmup(
    model: CoTrainingModel,
    labeled_loader: DataLoader,
    epochs: int = 10,
    log_every: int = 50,
) -> None:
    
    scaler = GradScaler()
    model.classifier1.train()
    model.classifier2.train()

    for epoch in range(epochs):
        for batch_idx, (x_l, y_l) in enumerate(labeled_loader):
            x_l = x_l.to(model.device, non_blocking=True)
            y_l = y_l.to(model.device, non_blocking=True)

            model.optimizer1.zero_grad()
            model.optimizer2.zero_grad()

            with autocast():
                logits1, features1, _ = model.classifier1(x_l)
                logits2, features2, _ = model.classifier2(x_l)
                loss1 = model._supervised_loss(logits1, y_l)
                loss2 = model._supervised_loss(logits2, y_l)
                loss = loss1 + loss2

            scaler.scale(loss).backward()
            scaler.step(model.optimizer1)
            scaler.step(model.optimizer2)
            scaler.update()

            with torch.no_grad():
                model._update_prototypes(features1, y_l, model.prototypes1)
                model._update_prototypes(features2, y_l, model.prototypes2)

            if batch_idx % log_every == 0:
                print(
                    f"[Warm-up] epoch {epoch + 1}/{epochs} "
                    f"batch {batch_idx}/{len(labeled_loader)} "
                    f"loss={float(loss):.4f}"
                )

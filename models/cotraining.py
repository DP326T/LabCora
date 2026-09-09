from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from .discriminator import DomainDiscriminator
from .resnet import ResNet101MultiLabel


class CoTrainingModel:
    

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        lr_backbone: float = 1e-3,
        lr_discriminator: float = 1e-4,
        feature_dim: int = 2048,
        tau: float = 0.80,
        lambda_u: float = 1.0,
        lambda_con: float = 1.0,
        lambda_d: float = 1.0,
        prototype_momentum: float = 0.9,
        alpha_start: float = 0.0,
        alpha_end: float = 1.0,
        alpha_ramp_steps: int = 1000,
        alpha_ramp_type: str = "linear",
        alpha_step_values: Optional[Tuple[float, ...]] = None,
    ):
        self.num_classes = num_classes
        self.device = device
        self.feature_dim = feature_dim
        self.scaler = GradScaler()

        # ----- Backbones and discriminators ----------------------------------
        self.classifier1 = ResNet101MultiLabel(num_classes).to(device)
        self.classifier2 = ResNet101MultiLabel(num_classes).to(device)
        self.domain_discriminator1 = DomainDiscriminator(feature_dim).to(device)
        self.domain_discriminator2 = DomainDiscriminator(feature_dim).to(device)

        # ----- Optimisers -----------------------------------------------------
        self.optimizer1 = optim.Adam(self.classifier1.parameters(), lr=lr_backbone)
        self.optimizer2 = optim.Adam(self.classifier2.parameters(), lr=lr_backbone)
        self.discriminator_optimizer1 = optim.Adam(
            self.domain_discriminator1.parameters(), lr=lr_discriminator,
        )
        self.discriminator_optimizer2 = optim.Adam(
            self.domain_discriminator2.parameters(), lr=lr_discriminator,
        )

        # ----- Losses ---------------------------------------------------------
        self.classification_criterion_s = nn.BCEWithLogitsLoss(reduction="none")
        self.classification_criterion_u = nn.BCEWithLogitsLoss(reduction="none")
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # ----- Loss weights and threshold ------------------------------------
        self.lambda_u = lambda_u
        self.lambda_con = lambda_con
        self.lambda_d = lambda_d
        self.tau = tau

        # ----- Prototypes (EMA) ----------------------------------------------
        self.prototype_momentum = prototype_momentum
        self.prototypes1 = torch.zeros(num_classes, feature_dim).to(device)
        self.prototypes2 = torch.zeros(num_classes, feature_dim).to(device)

        # ----- alpha ramp-up --------------------------------------------------
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_ramp_steps = alpha_ramp_steps
        self.alpha_ramp_type = alpha_ramp_type
        self.alpha_step_values = alpha_step_values or (alpha_start, alpha_end)

    # ------------------------------------------------------------------ helpers

    def get_alpha_by_ramp(self, global_step: int) -> float:
        
        t = float(global_step) / max(1.0, float(self.alpha_ramp_steps))
        frac = max(0.0, min(1.0, t))

        a0 = float(self.alpha_start)
        a1 = float(self.alpha_end)

        if self.alpha_ramp_type == "cosine":
            cos_val = 0.5 * (1.0 - np.cos(np.pi * frac))
            alpha = a0 + (a1 - a0) * cos_val
        elif self.alpha_ramp_type == "linear":
            alpha = a0 + (a1 - a0) * frac
        elif self.alpha_ramp_type == "gaussian":
            sigma = 0.4
            x = 2.0 * frac - 1.0
            gauss = np.exp(-0.5 * (x / sigma) ** 2)
            gauss_min = np.exp(-0.5 * (1.0 / sigma) ** 2)
            gauss_norm = (gauss - gauss_min) / (1.0 - gauss_min + 1e-12)
            alpha = a0 + (a1 - a0) * gauss_norm
        elif self.alpha_ramp_type == "step":
            steps = len(self.alpha_step_values)
            if steps <= 1:
                alpha = float(self.alpha_step_values[0]) if steps == 1 else a1
            else:
                idx = int(frac * steps)
                if idx >= steps:
                    idx = steps - 1
                alpha = float(self.alpha_step_values[idx])
        else:
            alpha = a0
        return float(np.clip(alpha, 0.0, 1.0))

    def _supervised_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        mask = (labels == 0.0) | (labels == 1.0)
        loss_per_element = self.classification_criterion_s(logits, labels)
        masked = loss_per_element * mask
        valid_rows = mask.sum(dim=1) > 0
        if valid_rows.sum() > 0:
            valid_masked = masked[valid_rows]
            valid_mask = mask[valid_rows]
            total_valid = valid_mask.sum()
            if total_valid > 0:
                return valid_masked.sum() / total_valid
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        prototype_buf: torch.Tensor,
    ) -> None:
        
        for i in range(self.num_classes):
            class_mask = (labels[:, i] == 1) | (labels[:, i] == 0)
            if class_mask.sum() == 0:
                continue
            class_feats = features[class_mask].mean(dim=0)
            prototype_buf[i] = (
                self.prototype_momentum * prototype_buf[i]
                + (1.0 - self.prototype_momentum) * class_feats
            )

    # ------------------------------------------------------------------ training

    def train_step(
        self,
        labeled_data,
        unlabeled_data,
        global_step: int,
    ) -> dict:
        
        x_l, y_l = labeled_data
        x_l, y_l = x_l.to(self.device), y_l.to(self.device)

        x_u_w, x_u_s1, x_u_s2 = unlabeled_data
        x_u_w = x_u_w.to(self.device)
        x_u_s1 = x_u_s1.to(self.device)
        x_u_s2 = x_u_s2.to(self.device)

        for opt in (
            self.optimizer1,
            self.optimizer2,
            self.discriminator_optimizer1,
            self.discriminator_optimizer2,
        ):
            opt.zero_grad()

        with autocast():
            # ----- Supervised branch -----------------------------------------
            logits_l1, features_l1, label_corr_l1 = self.classifier1(x_l)
            logits_l2, features_l2, label_corr_l2 = self.classifier2(x_l)
            loss_sup1 = self._supervised_loss(logits_l1, y_l)
            loss_sup2 = self._supervised_loss(logits_l2, y_l)

            # ----- Prototype EMA update (no grad) -----------------------------
            with torch.no_grad():
                self._update_prototypes(features_l1, y_l, self.prototypes1)
                self._update_prototypes(features_l2, y_l, self.prototypes2)

            # ----- Unlabelled forward (three views) --------------------------
            logits_u1_weak, features_u1_weak, label_corr_u1_weak = self.classifier1(x_u_w)
            logits_u2_weak, features_u2_weak, label_corr_u2_weak = self.classifier2(x_u_w)

            logits_u1_strong, _, _ = self.classifier1(x_u_s1)
            logits_u2_strong, _, _ = self.classifier2(x_u_s2)

            # ----- Pseudo-label fusion (prototype + classifier probability) --
            with torch.no_grad():
                probs_u1_weak = torch.sigmoid(logits_u1_weak).detach()
                probs_u2_weak = torch.sigmoid(logits_u2_weak).detach()

                proto_sims1 = torch.mm(
                    nn.functional.normalize(features_u1_weak, dim=1),
                    nn.functional.normalize(self.prototypes1.T, dim=0),
                )  # [B, C]
                norm1 = label_corr_u1_weak / (
                    label_corr_u1_weak.sum(dim=-1, keepdim=True) + 1e-8
                )
                proto_sims_corr1 = torch.bmm(norm1, proto_sims1.unsqueeze(2)).squeeze(2)

                proto_sims2 = torch.mm(
                    nn.functional.normalize(features_u2_weak, dim=1),
                    nn.functional.normalize(self.prototypes2.T, dim=0),
                )
                norm2 = label_corr_u2_weak / (
                    label_corr_u2_weak.sum(dim=-1, keepdim=True) + 1e-8
                )
                proto_sims_corr2 = torch.bmm(norm2, proto_sims2.unsqueeze(2)).squeeze(2)

                alpha = self.get_alpha_by_ramp(global_step)
                fused_pseudo_probs1 = alpha * proto_sims_corr1 + (1.0 - alpha) * probs_u1_weak
                fused_pseudo_probs2 = alpha * proto_sims_corr2 + (1.0 - alpha) * probs_u2_weak

                fused_mask1 = (fused_pseudo_probs1 > self.tau).float()
                fused_mask2 = (fused_pseudo_probs2 > self.tau).float()
                fused_pseudo_labels1 = (fused_pseudo_probs1 > self.tau).float()
                fused_pseudo_labels2 = (fused_pseudo_probs2 > self.tau).float()

            # ----- Consistency loss (cross-student) --------------------------
            loss_u1 = (
                self.classification_criterion_u(
                    logits_u1_strong, fused_pseudo_labels2,
                )
                * fused_mask2
            ).mean()
            loss_u2 = (
                self.classification_criterion_u(
                    logits_u2_strong, fused_pseudo_labels1,
                )
                * fused_mask1
            ).mean()
            loss_consistency = (loss_u1 + loss_u2) * 0.5

            # ----- Label-correlation consistency ------------------------------
            corr_diff = (label_corr_l1 - label_corr_l2).pow(2).mean()

            # ----- DANN domain loss ------------------------------------------
            d1_l = self.domain_discriminator1(features_l1)
            d1_u = self.domain_discriminator1(features_u1_weak)
            d2_l = self.domain_discriminator2(features_l2)
            d2_u = self.domain_discriminator2(features_u2_weak)
            d_lab1 = torch.ones_like(d1_l)
            d_unl1 = torch.zeros_like(d1_u)
            d_lab2 = torch.ones_like(d2_l)
            d_unl2 = torch.zeros_like(d2_u)
            loss_d1 = self.domain_criterion(d1_l, d_lab1) + self.domain_criterion(
                d1_u, d_unl1
            )
            loss_d2 = self.domain_criterion(d2_l, d_lab2) + self.domain_criterion(
                d2_u, d_unl2
            )
            loss_domain = (loss_d1 + loss_d2) * 0.5

            loss_total = (
                loss_sup1
                + loss_sup2
                + self.lambda_u * loss_consistency
                + self.lambda_con * corr_diff
                + self.lambda_d * loss_domain
            )

        self.scaler.scale(loss_total).backward()
        self.scaler.step(self.optimizer1)
        self.scaler.step(self.optimizer2)
        self.scaler.update()

        # Discriminator update uses a fresh forward (no gradient reversal
        # applied here — keep the discriminator objective independent).
        self._discriminator_step(features_l1, features_u1_weak, features_l2, features_u2_weak)

        return {
            "loss_sup": float((loss_sup1 + loss_sup2).item()),
            "loss_u": float(loss_consistency.item()),
            "loss_corr": float(corr_diff.item()),
            "loss_domain": float(loss_domain.item()),
            "alpha": alpha,
        }

    def _discriminator_step(
        self,
        features_l1: torch.Tensor,
        features_u1: torch.Tensor,
        features_l2: torch.Tensor,
        features_u2: torch.Tensor,
    ) -> None:
        
        for opt in (self.discriminator_optimizer1, self.discriminator_optimizer2):
            opt.zero_grad()
        d1_l = self.domain_discriminator1(features_l1.detach())
        d1_u = self.domain_discriminator1(features_u1.detach())
        d2_l = self.domain_discriminator2(features_l2.detach())
        d2_u = self.domain_discriminator2(features_u2.detach())
        loss_d1 = self.domain_criterion(
            d1_l, torch.ones_like(d1_l)
        ) + self.domain_criterion(d1_u, torch.zeros_like(d1_u))
        loss_d2 = self.domain_criterion(
            d2_l, torch.ones_like(d2_l)
        ) + self.domain_criterion(d2_u, torch.zeros_like(d2_u))
        loss_d = (loss_d1 + loss_d2) * 0.5
        loss_d.backward()
        self.discriminator_optimizer1.step()
        self.discriminator_optimizer2.step()

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

from .label_corr import LabelCorrelationModule


class ResNet101MultiLabel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)
        self.label_corr = LabelCorrelationModule(num_classes)
        self.E = self.classifier.weight

    def forward(self, x):
        features = self.feature_extractor(x)
        global_features = self.gap(features).squeeze(-1).squeeze(-1)
        logits = self.classifier(global_features)

        B = features.size(0)
        E_expanded = self.E.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            B, -1, -1, -1, -1
        )
        features_expanded = features.unsqueeze(1)
        label_specific = E_expanded * features_expanded
        label_specific = label_specific.view(B, self.label_corr.num_classes, 2048, -1)
        label_specific = label_specific.permute(0, 1, 3, 2)
        label_specific = label_specific.mean(dim=2)

        label_corr_matrix = self.label_corr(label_specific)
        return logits, global_features, label_corr_matrix

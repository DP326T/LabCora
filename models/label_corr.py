import torch
import torch.nn as nn


class LabelCorrelationModule(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 2048, num_heads: int = 8):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        self.query = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.key = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        Q = self.query(features)
        K = self.key(features)

        Q = Q.view(batch_size, self.num_classes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.num_classes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)

        attention = attention.mean(dim=1)
        return attention

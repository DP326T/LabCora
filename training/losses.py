

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    

    lambda_u: float = 1.0       # Unlabelled consistency loss.
    lambda_con: float = 1.0     # Label-correlation consistency.
    lambda_d: float = 1.0       # Domain-adversarial loss.

    # Pseudo-label confidence threshold.
    tau: float = 0.95


DEFAULT_LOSS_WEIGHTS = LossWeights()

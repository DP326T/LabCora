from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_labelled_unlabelled(
    train_df: pd.DataFrame,
    labeled_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < labeled_ratio <= 1.0:
        raise ValueError(f"labeled_ratio must be in (0, 1], got {labeled_ratio}")

    labeled_df, unlabeled_df = train_test_split(
        train_df,
        test_size=1 - labeled_ratio,
        random_state=seed,
    )
    return labeled_df, unlabeled_df

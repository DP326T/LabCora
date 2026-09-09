import os
from typing import Dict

import pandas as pd


def read_txt_file(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_name, right_name = line.split()
            data.append({"left_name": left_name, "right_name": right_name})
    return pd.DataFrame(data)


def load_txt_splits(
    split_dir: str,
) -> Dict[str, pd.DataFrame]:
    train_file = os.path.join(split_dir, "train.txt")
    val_file = os.path.join(split_dir, "val.txt")
    test_file = os.path.join(split_dir, "test.txt")

    if not all(os.path.exists(p) for p in [train_file, val_file, test_file]):
        raise FileNotFoundError(
            f"Split files not found in {split_dir!r}. "
            "Expected train.txt, val.txt, test.txt."
        )

    return {
        "train": read_txt_file(train_file),
        "val": read_txt_file(val_file),
        "test": read_txt_file(test_file),
    }

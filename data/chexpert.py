import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .transforms import (
    strong_view1_transform,
    strong_view2_transform,
    weak_transform,
)


CHEXPERT_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

NUM_CHEXPERT_CLASSES = len(CHEXPERT_CLASSES)


def _resolve_image_path(raw_path: str, data_root: str) -> str:
    cleaned = str(raw_path).replace("\\", "/")
    if os.path.isabs(cleaned):
        return cleaned
    return os.path.join(data_root, cleaned)


class CheXpertLabeledDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.image_paths = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1 : 1 + NUM_CHEXPERT_CLASSES].values.astype("float32")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        raw_path = self.image_paths[idx]
        img_path = _resolve_image_path(raw_path, self.data_root)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"[CheXpertLabeledDataset] Missing image: {img_path} "
                f"(raw path: {raw_path!r})"
            )

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        labels = self.labels[idx]
        return image, labels


class CheXpertUnlabeledDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform_weak=None,
        transform_strong_1=None,
        transform_strong_2=None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform_weak = transform_weak or weak_transform()
        self.transform_strong_1 = transform_strong_1 or strong_view1_transform()
        self.transform_strong_2 = transform_strong_2 or strong_view2_transform()
        self.image_paths = self.df.iloc[:, 0].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        raw_path = self.image_paths[idx]
        img_path = _resolve_image_path(raw_path, self.data_root)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"[CheXpertUnlabeledDataset] Missing image: {img_path} "
                f"(raw path: {raw_path!r})"
            )

        image = Image.open(img_path).convert("RGB")
        view1 = self.transform_strong_1(image)
        view2 = self.transform_strong_2(image)
        weak_view = self.transform_weak(image)
        return view1, view2, weak_view


def apply_uzeros_policy(df: pd.DataFrame, classes: Optional[list] = None) -> pd.DataFrame:
    if classes is None:
        classes = CHEXPERT_CLASSES

    label_block = df[classes].copy()
    label_block = label_block.replace(-1, 0).fillna(0).astype("int64")
    keep = label_block.sum(axis=1) > 0
    out = df.loc[keep].copy()
    out[classes] = label_block.loc[keep].values
    return out.reset_index(drop=True)

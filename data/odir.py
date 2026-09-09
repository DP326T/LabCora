import os
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


ODIR_CLASSES: List[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]

ODIR_KEYWORDS = {
    "N": ["normal"],
    "D": [
        "diabetic",
        "non proliferative retinopathy",
        "nonproliferative retinopathy",
    ],
    "G": ["glaucoma"],
    "C": ["cataract"],
    "A": ["age related macular degeneration"],
    "H": ["hypertensive"],
    "M": ["myopia"],
    "O": [],
}


def _row_to_labels(keywords_raw: str) -> List[int]:
    text = str(keywords_raw).lower().strip()
    labels = [0] * len(ODIR_CLASSES)

    for i, cls in enumerate(ODIR_CLASSES[:-1]):
        for kw in ODIR_KEYWORDS[cls]:
            if kw in text:
                labels[i] = 1

    cleaned = [k.strip() for k in text.replace("，", ",").split(",")]
    for k in cleaned:
        if not k:
            continue
        if not any(kw in k for v in ODIR_KEYWORDS.values() for kw in v):
            labels[-1] = 1
    return labels


def process_odir(input_path: str, image_root_dir: str, output_csv: str) -> pd.DataFrame:
    if input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    rows = []
    for _, row in df.iterrows():
        for eye in ("Left", "Right"):
            image_name = row[f"{eye}-Fundus"]
            image_path = os.path.join(image_root_dir, image_name)
            if not os.path.exists(image_path):
                continue
            keywords_raw = row[f"{eye}-Diagnostic Keywords"]
            labels = _row_to_labels(keywords_raw)
            rows.append([image_name] + labels)

    out = pd.DataFrame(rows, columns=["Image"] + ODIR_CLASSES)
    out.to_csv(output_csv, index=False)
    print(f"[process_odir] Saved {len(out)} images to {output_csv}")
    return out


def _extract_patient_id(path: str) -> str:
    return os.path.basename(path).split("_")[0]


def split_odir_by_patient(
    csv_path: str,
    output_dir: str,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
    copy_images: bool = True,
) -> dict:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    base = Path(output_dir)
    train_dir = base / "train"
    valid_dir = base / "valid"
    test_dir = base / "test"
    split_dir = base / "splits"

    if copy_images:
        for d in (train_dir, valid_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["ID"] = df["Image"].apply(_extract_patient_id)
    ids = df["ID"].unique()

    train_ids, temp_ids = train_test_split(
        ids, test_size=1 - ratios[0], random_state=seed, shuffle=True,
    )
    valid_ids, test_ids = train_test_split(
        temp_ids,
        test_size=ratios[2] / (ratios[1] + ratios[2]),
        random_state=seed,
        shuffle=True,
    )

    print(
        f"[split_odir_by_patient] Train={len(train_ids)} "
        f"Valid={len(valid_ids)} Test={len(test_ids)} patient IDs"
    )

    splits = {}
    for split_name, ids_subset, dest_dir in (
        ("train", train_ids, train_dir),
        ("valid", valid_ids, valid_dir),
        ("test", test_ids, test_dir),
    ):
        sub = df[df["ID"].isin(ids_subset)].copy()
        if copy_images:
            new_paths = []
            for src in sub["Image"]:
                fname = os.path.basename(src)
                src_path = Path(csv_path).parent / src if not Path(src).is_absolute() else Path(src)
                dst_path = dest_dir / fname
                if src_path.exists():
                    shutil.copy(src_path, dst_path)
                new_paths.append(str(dst_path))
            sub["Image"] = new_paths
        sub = sub.drop(columns=["ID"])
        sub.to_csv(split_dir / f"{split_name}.csv", index=False)
        splits[split_name] = sub

    return splits

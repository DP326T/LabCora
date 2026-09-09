from pathlib import Path
from typing import List, Tuple

import pandas as pd


_PATH_COLUMN_CANDIDATES: List[str] = [
    "Path",
    "path",
    "FilePath",
    "file_path",
    "filepath",
    "ImagePath",
    "image_path",
    "Study",
    "StudyPath",
    "Study_path",
    "Image Index",
]


def detect_path_column(df: pd.DataFrame) -> str:
    for name in _PATH_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    return df.columns[0]


def _strip_to_after_token(text: str, token: str) -> str:
    idx = text.find(token)
    if idx == -1:
        return text
    return text[idx + len(token) :]


_KNOWN_PATH_PREFIXES = [
    "data/CheXpert-v1.0/",
    "CheXpert-v1.0/",
]


def generate_candidate_paths(raw_value: str, dataset_root: Path) -> List[Path]:
    if raw_value is None:
        return []

    text = str(raw_value).strip().replace("\\", "/").lstrip("/")
    candidates: List[Path] = []

    candidates.append(Path(text))
    candidates.append(dataset_root / text)

    for prefix in _KNOWN_PATH_PREFIXES:
        if text.startswith(prefix):
            tail = text[len(prefix) :]
            candidates.append(dataset_root / tail)

    if "CheXpert-v1.0/" in text:
        tail = _strip_to_after_token(text, "CheXpert-v1.0/")
        candidates.append(dataset_root / tail)

    for split_root in ("train/", "valid/", "test/"):
        if text.startswith(split_root):
            candidates.append(dataset_root / text)

    unique: List[Path] = []
    seen = set()
    for cand in candidates:
        try:
            key = cand.resolve().as_posix()
        except OSError:
            key = cand.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def resolve_existing_path(raw_value: str, dataset_root: Path) -> Tuple[Path, bool]:
    for cand in generate_candidate_paths(raw_value, dataset_root):
        if cand.exists():
            return cand, True

    text = str(raw_value).strip().replace("\\", "/").lstrip("/")
    for prefix in _KNOWN_PATH_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return dataset_root / text, False


def resolve_csv_paths(csv_path: Path, dataset_root: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    path_col = detect_path_column(df)

    resolved = []
    exists_mask = []
    for raw in df[path_col].tolist():
        path, exists = resolve_existing_path(raw, dataset_root)
        resolved.append(str(path))
        exists_mask.append(bool(exists))

    df[path_col] = resolved
    df.to_csv(output_csv, index=False)

    missing = sum(1 for e in exists_mask if not e)
    if missing:
        print(f"[resolve_csv_paths] Warning: {missing} entries did not resolve on disk.")
    return df

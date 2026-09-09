import argparse
from pathlib import Path

# Make the project root importable.
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import paths  # noqa: E402
from data.chestxray14 import resolve_csv_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve ChestXRay14 CSV paths")
    parser.add_argument("--csv", required=True,
                        help="Path to the input CheXpert-style CSV.")
    parser.add_argument("--root", default=paths.CHESTXRAY14_ROOT,
                        help="Dataset root containing the raw images.")
    parser.add_argument("--output", required=True,
                        help="Destination CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_csv_paths(Path(args.csv), Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()

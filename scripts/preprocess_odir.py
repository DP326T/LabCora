

import argparse
from pathlib import Path

# Make the project root importable.
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.odir import process_odir, split_odir_by_patient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ODIR-5K preprocessing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_process = sub.add_parser("process", help="Convert raw metadata into a flat CSV")
    p_process.add_argument("--input", required=True)
    p_process.add_argument("--image_dir", required=True)
    p_process.add_argument("--output", required=True)

    p_split = sub.add_parser("split", help="Split the flat CSV by patient ID")
    p_split.add_argument("--csv", required=True)
    p_split.add_argument("--output_dir", required=True)
    p_split.add_argument("--no_copy", action="store_true",
                         help="Skip copying images into per-split folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "process":
        process_odir(args.input, args.image_dir, args.output)
    elif args.cmd == "split":
        split_odir_by_patient(
            args.csv,
            args.output_dir,
            copy_images=not args.no_copy,
        )
    else:  # pragma: no cover — argparse already enforces this branch.
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make the project root importable regardless of where the script is
# invoked from.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import paths  # noqa: E402
from data.chexpert import (  # noqa: E402
    NUM_CHEXPERT_CLASSES,
    CheXpertLabeledDataset,
    CheXpertUnlabeledDataset,
)
from data.co_split import split_labelled_unlabelled  # noqa: E402
from data.transforms import strong_transform, val_transform  # noqa: E402
from models.cotraining import CoTrainingModel  # noqa: E402
from training.evaluate import evaluate  # noqa: E402
from training.train import supervised_warmup  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LabCora CheXpert training")
    parser.add_argument("--labeled_ratio", type=float, default=0.1,
                        help="Fraction of the training set treated as labelled.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Supervised warm-up epochs.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Co-training epochs.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_mmd", action="store_true",
                        help="Compute an MMD distance between labelled "
                             "and unlabelled features after each epoch.")
    parser.add_argument("--visualize_tsne", action="store_true",
                        help="Render a 3D t-SNE plot after training.")
    parser.add_argument("--output_dir", type=str, default=paths.OUTPUT_DIR,
                        help="Where to write checkpoints and visualisations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Data -------------------------------------------------------------
    print(f"[main] CheXpert root: {paths.CHEXPERT_ROOT}")
    train_df = pd.read_csv(paths.CHEXPERT_TRAIN_CSV)
    valid_df = pd.read_csv(paths.CHEXPERT_VALID_CSV)
    test_df = pd.read_csv(paths.CHEXPERT_TEST_CSV)

    labeled_df, unlabeled_df = split_labelled_unlabelled(
        train_df, labeled_ratio=args.labeled_ratio, seed=args.seed,
    )
    print(
        f"[main] Labelled: {len(labeled_df)} | "
        f"Unlabelled: {len(unlabeled_df)} | "
        f"Valid: {len(valid_df)} | Test: {len(test_df)}"
    )

    labeled_dataset = CheXpertLabeledDataset(
        labeled_df, data_root=paths.CHEXPERT_ROOT, transform=strong_transform(),
    )
    unlabeled_dataset = CheXpertUnlabeledDataset(
        unlabeled_df, data_root=paths.CHEXPERT_ROOT,
    )
    val_dataset = CheXpertLabeledDataset(
        valid_df, data_root=paths.CHEXPERT_ROOT, transform=val_transform(),
    )
    test_dataset = CheXpertLabeledDataset(
        test_df, data_root=paths.CHEXPERT_ROOT, transform=val_transform(),
    )

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ----- Model ------------------------------------------------------------
    model = CoTrainingModel(
        num_classes=NUM_CHEXPERT_CLASSES,
        device=device,
        lr_backbone=args.lr,
    )

    # ----- Stage 1: supervised warm-up --------------------------------------
    print("[main] Stage 1 — supervised warm-up")
    supervised_warmup(model, labeled_loader, epochs=args.warmup_epochs)

    # ----- Stage 2: Co-training --------------------------------------------
    print("[main] Stage 2 — Co-training")
    global_step = 0
    for epoch in range(args.epochs):
        model.classifier1.train()
        model.classifier2.train()
        for batch_idx, (labeled_batch, unlabeled_batch) in enumerate(
            zip(labeled_loader, unlabeled_loader)
        ):
            stats = model.train_step(labeled_batch, unlabeled_batch, global_step)
            global_step += 1
            if batch_idx % 50 == 0:
                print(
                    f"[Co-train] epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_idx} "
                    f"sup={stats['loss_sup']:.4f} "
                    f"u={stats['loss_u']:.4f} "
                    f"corr={stats['loss_corr']:.4f} "
                    f"d={stats['loss_domain']:.4f} "
                    f"alpha={stats['alpha']:.3f}"
                )

        val_metrics = evaluate(model.classifier1, val_loader, device, NUM_CHEXPERT_CLASSES)
        print(
            f"[Co-train] epoch {epoch + 1}/{args.epochs} "
            f"val F1={val_metrics['f1']:.4f} "
            f"val AUC={val_metrics['auc']:.4f} "
            f"val mAP={val_metrics['mAP']:.4f}"
        )

        if args.use_mmd:
            from models.mmd import evaluate_domain_mmd
            mmd_value = evaluate_domain_mmd(
                model.classifier1, labeled_loader, unlabeled_loader, device,
            )
            print(f"[Co-train] epoch {epoch + 1}/{args.epochs} MMD={mmd_value:.4f}")

    # ----- Final evaluation -------------------------------------------------
    test_metrics = evaluate(model.classifier1, test_loader, device, NUM_CHEXPERT_CLASSES)
    print(
        f"[main] Final test metrics: "
        f"F1={test_metrics['f1']:.4f} "
        f"AUC={test_metrics['auc']:.4f} "
        f"mAP={test_metrics['mAP']:.4f} "
        f"Hamming={test_metrics['hamming']:.4f}"
    )

    if args.visualize_tsne:
        from analysis.tsne import visualize_3d_tsne
        visualize_3d_tsne(
            model.classifier1, labeled_loader, unlabeled_loader, device,
            save_path=os.path.join(args.output_dir, "tsne_alignment.png"),
        )


if __name__ == "__main__":
    main()

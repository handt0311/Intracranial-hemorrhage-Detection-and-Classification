from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Stage2_detection import config2 as cfg


def save_train_loss_curve(history, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Stage 2 Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "stage2_train_loss_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def save_map_curve(history, out_dir):
    eval_df = history.dropna(subset=["map50"]).copy()

    if len(eval_df) == 0:
        print("No validation mAP values found in history.csv")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(eval_df["epoch"], eval_df["map50"], marker="o", label="mAP@0.5")
    plt.plot(eval_df["epoch"], eval_df["map50_95"], marker="s", label="COCO mAP@0.5:0.95")

    best_idx = eval_df["map50"].idxmax()
    best_epoch = int(eval_df.loc[best_idx, "epoch"])
    best_map50 = float(eval_df.loc[best_idx, "map50"])

    plt.axvline(best_epoch, linestyle="--", alpha=0.5)
    plt.text(
        best_epoch,
        best_map50,
        f" best epoch={best_epoch}\n mAP@0.5={best_map50:.3f}",
        fontsize=9,
    )

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Stage 2 Validation mAP")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "stage2_validation_map_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def save_lr_curve(history, out_dir):
    if "lr" not in history.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["lr"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "stage2_lr_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def save_per_class_ap(per_class_df, out_dir):
    if "class_name" not in per_class_df.columns:
        raise ValueError("Missing class_name column in per-class AP CSV")

    x = per_class_df["class_name"].tolist()

    plt.figure(figsize=(10, 5))
    plt.bar(x, per_class_df["ap50"])
    plt.xlabel("Class")
    plt.ylabel("AP@0.5")
    plt.title("Per-class AP@0.5")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "stage2_per_class_ap50.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

    plt.figure(figsize=(10, 5))
    plt.bar(x, per_class_df["ap50_95"])
    plt.xlabel("Class")
    plt.ylabel("COCO AP@0.5:0.95")
    plt.title("Per-class COCO AP@0.5:0.95")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "stage2_per_class_ap50_95.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def main():
    run_dir = cfg.RUN_DIR
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.csv"
    best_map_path = run_dir / "best_val_map_per_class.csv"
    best_coco_map_path = run_dir / "best_coco_val_map_per_class.csv"

    print("Run dir:", run_dir)

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    history = pd.read_csv(history_path)

    print("\nHistory:")
    print(history.tail())

    save_train_loss_curve(history, out_dir)
    save_map_curve(history, out_dir)
    save_lr_curve(history, out_dir)

    if best_map_path.exists():
        per_class_df = pd.read_csv(best_map_path)
        print("\nBest mAP@0.5 per-class result:")
        print(per_class_df[["class_name", "num_gt", "num_pred", "ap50", "ap50_95"]])
        save_per_class_ap(per_class_df, out_dir)
    elif best_coco_map_path.exists():
        per_class_df = pd.read_csv(best_coco_map_path)
        print("\nBest COCO per-class result:")
        print(per_class_df[["class_name", "num_gt", "num_pred", "ap50", "ap50_95"]])
        save_per_class_ap(per_class_df, out_dir)
    else:
        print("No best per-class AP CSV found.")

    print("\nPlots saved to:", out_dir)


if __name__ == "__main__":
    main()

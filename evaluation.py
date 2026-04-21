import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from config import Config
from dataset import RSNADataset, build_train_val_dataframes, set_seed
from model import RSNAClassifier


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_config_from_snapshot(run_dir: str):
    config = Config()

    snapshot_path = os.path.join(run_dir, "config_snapshot.json")
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        for key, value in snapshot.items():
            setattr(config, key, value)

    # Quan trọng: ép lại đúng folder run cũ
    config.OUTPUT_DIR = run_dir
    return config


def compute_multilabel_auc(y_true, y_prob, label_names):
    auc_dict = {}
    auc_values = []

    for i, name in enumerate(label_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")

        auc_dict[name] = auc
        if not np.isnan(auc):
            auc_values.append(auc)

    mean_auc = float(np.mean(auc_values)) if len(auc_values) > 0 else float("nan")
    return mean_auc, auc_dict


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_names, use_amp):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []
    all_image_ids = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_image_ids.extend(image_ids)

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    val_loss = total_loss / len(loader.dataset)

    mean_auc, auc_dict = compute_multilabel_auc(y_true, y_prob, label_names)
    return val_loss, mean_auc, auc_dict, y_true, y_prob, all_image_ids


def compute_metrics_per_class(y_true, y_prob, label_names, threshold=0.5):
    rows = []

    for i, label in enumerate(label_names):
        y_t = y_true[:, i].astype(int)
        y_p = y_prob[:, i]
        y_pred = (y_p >= threshold).astype(int)

        try:
            auc = roc_auc_score(y_t, y_p)
        except ValueError:
            auc = float("nan")

        try:
            pr_auc = average_precision_score(y_t, y_p)
        except ValueError:
            pr_auc = float("nan")

        precision = precision_score(y_t, y_pred, zero_division=0)
        recall = recall_score(y_t, y_pred, zero_division=0)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        acc = accuracy_score(y_t, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append(
            {
                "label": label,
                "auc": auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "f1": f1,
                "accuracy": acc,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "positive_rate_true": float(y_t.mean()),
                "positive_rate_pred": float(y_pred.mean()),
            }
        )

    return pd.DataFrame(rows)


def save_predictions_csv(image_ids, y_true, y_prob, label_names, save_path: str, threshold=0.5):
    data = {"image_id": image_ids}

    for i, label in enumerate(label_names):
        data[f"{label}_true"] = y_true[:, i]
        data[f"{label}_prob"] = y_prob[:, i]
        data[f"{label}_pred"] = (y_prob[:, i] >= threshold).astype(np.int32)

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def plot_roc_curves(y_true, y_prob, label_names, save_path: str):
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(label_names):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        if len(np.unique(y_t)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curves(y_true, y_prob, label_names, save_path: str):
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(label_names):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        if len(np.unique(y_t)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_t, y_p)
        pr_auc = average_precision_score(y_t, y_p)
        plt.plot(recall, precision, label=f"{label} (AP={pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrices(y_true, y_prob, label_names, save_path: str, threshold=0.5):
    num_labels = len(label_names)
    cols = 3
    rows = int(np.ceil(num_labels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, label in enumerate(label_names):
        ax = axes[i]
        y_t = y_true[:, i].astype(int)
        y_pred = (y_prob[:, i] >= threshold).astype(int)

        cm = confusion_matrix(y_t, y_pred, labels=[0, 1])

        im = ax.imshow(cm)
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")

    for j in range(num_labels, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to a specific training run folder")
    parser.add_argument("--model_name", type=str, default="best_model.pth", help="Checkpoint file name inside run_dir")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for y_pred")
    args = parser.parse_args()

    run_dir = args.run_dir
    model_path = os.path.join(run_dir, args.model_name)
    eval_dir = os.path.join(run_dir, "evaluation")
    ensure_dir(eval_dir)

    config = load_config_from_snapshot(run_dir)
    set_seed(config.SEED)

    use_cuda = torch.cuda.is_available() and config.DEVICE == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = bool(getattr(config, "USE_AMP", False) and use_cuda)

    print(f"Using device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print("Rebuilding train/val split...")
    train_df, val_df = build_train_val_dataframes(config)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    val_dataset = RSNADataset(val_df, config)

    persistent_workers = getattr(config, "PERSISTENT_WORKERS", False) if config.NUM_WORKERS > 0 else False
    pin_memory = getattr(config, "PIN_MEMORY", False) if use_cuda else False

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = RSNAClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        head_type=config.HEAD_TYPE,
        mlp_hidden_dim=config.MLP_HIDDEN_DIM,
        dropout=config.DROPOUT,
        kan_num_basis=config.KAN_NUM_BASIS,
        kan_hidden_dim=config.KAN_HIDDEN_DIM,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    criterion = nn.BCEWithLogitsLoss()

    print("Running evaluation...")
    val_loss, mean_auc, auc_dict, y_true, y_prob, image_ids = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        label_names=config.LABEL_COLS,
        use_amp=use_amp,
    )

    print(f"val_loss: {val_loss:.4f}")
    print(f"val_mean_auc: {mean_auc:.4f}")
    for label in config.LABEL_COLS:
        print(f"{label}: {auc_dict[label]:.4f}")

    metrics_df = compute_metrics_per_class(
        y_true=y_true,
        y_prob=y_prob,
        label_names=config.LABEL_COLS,
        threshold=args.threshold,
    )

    summary = {
        "val_loss": float(val_loss),
        "val_mean_auc": float(mean_auc),
        "threshold": float(args.threshold),
    }

    summary_path = os.path.join(eval_dir, "summary.json")
    metrics_csv_path = os.path.join(eval_dir, "per_class_metrics.csv")
    preds_csv_path = os.path.join(eval_dir, "val_predictions_eval.csv")
    roc_png_path = os.path.join(eval_dir, "roc_curves.png")
    pr_png_path = os.path.join(eval_dir, "pr_curves.png")
    cm_png_path = os.path.join(eval_dir, "confusion_matrices.png")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    metrics_df.to_csv(metrics_csv_path, index=False)
    save_predictions_csv(
        image_ids=image_ids,
        y_true=y_true,
        y_prob=y_prob,
        label_names=config.LABEL_COLS,
        save_path=preds_csv_path,
        threshold=args.threshold,
    )

    plot_roc_curves(y_true, y_prob, config.LABEL_COLS, roc_png_path)
    plot_pr_curves(y_true, y_prob, config.LABEL_COLS, pr_png_path)
    plot_confusion_matrices(y_true, y_prob, config.LABEL_COLS, cm_png_path, threshold=args.threshold)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved per-class metrics to: {metrics_csv_path}")
    print(f"Saved predictions to: {preds_csv_path}")
    print(f"Saved ROC plot to: {roc_png_path}")
    print(f"Saved PR plot to: {pr_png_path}")
    print(f"Saved confusion matrices to: {cm_png_path}")


if __name__ == "__main__":
    main()
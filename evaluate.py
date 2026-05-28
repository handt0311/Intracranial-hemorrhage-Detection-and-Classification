import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


LABELS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_metrics(df, labels, threshold):
    rows = []

    for label in labels:
        true_col = f"{label}_true"
        prob_col = f"{label}_prob"

        y_true = df[true_col].values.astype(int)
        y_prob = df[prob_col].values.astype(float)
        y_pred = (y_prob >= threshold).astype(int)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = np.nan

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append({
            "label": label,
            "threshold": threshold,
            "auc": auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "accuracy": accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "positive_rate_true": float(y_true.mean()),
            "positive_rate_pred": float(y_pred.mean()),
        })

    return pd.DataFrame(rows)


def plot_roc_curves(df, labels, save_path):
    plt.figure(figsize=(8, 6))

    for label in labels:
        y_true = df[f"{label}_true"].values.astype(int)
        y_prob = df[f"{label}_prob"].values.astype(float)

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curves(df, labels, save_path):
    plt.figure(figsize=(8, 6))

    for label in labels:
        y_true = df[f"{label}_true"].values.astype(int)
        y_prob = df[f"{label}_prob"].values.astype(float)

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{label} (AP={pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_csv = args.pred_csv

    if args.out_dir is None:
        out_dir = os.path.join(os.path.dirname(pred_csv), "evaluation_from_val_predictions")
    else:
        out_dir = args.out_dir

    ensure_dir(out_dir)

    df = pd.read_csv(pred_csv)

    metrics_df = compute_metrics(df, LABELS, args.threshold)
    metrics_path = os.path.join(out_dir, f"metrics_threshold_{args.threshold}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)]
    sweep_results = []

    for threshold in thresholds:
        sweep_results.append(compute_metrics(df, LABELS, threshold))

    sweep_df = pd.concat(sweep_results, ignore_index=True)
    sweep_path = os.path.join(out_dir, "threshold_sweep_metrics.csv")
    sweep_df.to_csv(sweep_path, index=False)

    best_f1_df = (
        sweep_df
        .sort_values(["label", "f1"], ascending=[True, False])
        .groupby("label")
        .head(1)
        .reset_index(drop=True)
    )
    best_f1_path = os.path.join(out_dir, "best_threshold_by_f1.csv")
    best_f1_df.to_csv(best_f1_path, index=False)

    high_sens_df = sweep_df[sweep_df["recall_sensitivity"] >= 0.95].copy()
    if len(high_sens_df) > 0:
        best_high_sens_df = (
            high_sens_df
            .sort_values(["label", "specificity"], ascending=[True, False])
            .groupby("label")
            .head(1)
            .reset_index(drop=True)
        )
    else:
        best_high_sens_df = pd.DataFrame()

    best_high_sens_path = os.path.join(out_dir, "best_threshold_sensitivity_0_95.csv")
    best_high_sens_df.to_csv(best_high_sens_path, index=False)

    plot_roc_curves(df, LABELS, os.path.join(out_dir, "roc_curves.png"))
    plot_pr_curves(df, LABELS, os.path.join(out_dir, "pr_curves.png"))

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved threshold sweep to: {sweep_path}")
    print(f"Saved best threshold by F1 to: {best_f1_path}")
    print(f"Saved best threshold with sensitivity >= 0.95 to: {best_high_sens_path}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
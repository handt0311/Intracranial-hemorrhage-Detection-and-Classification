from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from Stage2_detection import config2 as cfg
from Stage2_detection.dataset_bhx import BHXDetectionDataset, detection_collate_fn
from Stage2_detection.model_resnet18_fasterrcnn import build_resnet18_fasterrcnn


IOU_THRESH = 0.5
SCORE_THRESH = 0.50

# Set to an integer for quick debugging, e.g. 200.
# Set to None for full validation.
MAX_EVAL_IMAGES = None


def evaluate_one_image(pred, target, num_classes, iou_thresh=0.5, score_thresh=0.05):
    """
    Greedy per-class matching between predicted boxes and ground-truth boxes.

    Returns:
        stats[class_id] = {"tp": ..., "fp": ..., "fn": ...}
    """
    stats = {
        cls_id: {"tp": 0, "fp": 0, "fn": 0}
        for cls_id in range(1, num_classes)
    }

    pred_boxes = pred["boxes"].detach().cpu()
    pred_scores = pred["scores"].detach().cpu()
    pred_labels = pred["labels"].detach().cpu()

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    keep = pred_scores >= score_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    for cls_id in range(1, num_classes):
        cls_pred_mask = pred_labels == cls_id
        cls_gt_mask = gt_labels == cls_id

        cls_pred_boxes = pred_boxes[cls_pred_mask]
        cls_pred_scores = pred_scores[cls_pred_mask]
        cls_gt_boxes = gt_boxes[cls_gt_mask]

        if len(cls_gt_boxes) == 0 and len(cls_pred_boxes) == 0:
            continue

        if len(cls_gt_boxes) == 0:
            stats[cls_id]["fp"] += len(cls_pred_boxes)
            continue

        if len(cls_pred_boxes) == 0:
            stats[cls_id]["fn"] += len(cls_gt_boxes)
            continue

        order = torch.argsort(cls_pred_scores, descending=True)
        cls_pred_boxes = cls_pred_boxes[order]

        matched_gt = set()

        ious = box_iou(cls_pred_boxes, cls_gt_boxes)

        for pred_idx in range(len(cls_pred_boxes)):
            best_iou, best_gt_idx = torch.max(ious[pred_idx], dim=0)
            best_iou = float(best_iou.item())
            best_gt_idx = int(best_gt_idx.item())

            if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
                stats[cls_id]["tp"] += 1
                matched_gt.add(best_gt_idx)
            else:
                stats[cls_id]["fp"] += 1

        stats[cls_id]["fn"] += len(cls_gt_boxes) - len(matched_gt)

    return stats


def merge_stats(total_stats, image_stats):
    for cls_id, s in image_stats.items():
        total_stats[cls_id]["tp"] += s["tp"]
        total_stats[cls_id]["fp"] += s["fp"]
        total_stats[cls_id]["fn"] += s["fn"]


def compute_metrics(total_stats):
    rows = []

    for cls_id, s in total_stats.items():
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "class_id": cls_id,
                "class_name": cfg.ID_TO_CLASS.get(cls_id, str(cls_id)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision@0.5": precision,
                "recall@0.5": recall,
                "f1@0.5": f1,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main():
    device = torch.device(cfg.DEVICE)
    print("Using device:", device)

    val_dataset = BHXDetectionDataset(
        csv_path=cfg.VAL_CSV,
        image_size=cfg.IMAGE_SIZE,
        window_center=cfg.WINDOW_CENTER,
        window_width=cfg.WINDOW_WIDTH,
    )

    if MAX_EVAL_IMAGES is not None:
        val_dataset.sop_uids = val_dataset.sop_uids[:MAX_EVAL_IMAGES]

    print("Validation images:", len(val_dataset))

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_resnet18_fasterrcnn(
        num_classes=cfg.NUM_CLASSES,
        rsna_checkpoint_path=getattr(cfg, "RSNA_BEST_MODEL_PATH", None),
        image_size=cfg.IMAGE_SIZE,
    )

    detector_path = cfg.RUN_DIR / getattr(cfg, "BEST_DETECTOR_NAME", "best_detector.pth")
    print("Loading detector:", detector_path)

    state_dict = torch.load(detector_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    total_stats = {
        cls_id: {"tp": 0, "fp": 0, "fn": 0}
        for cls_id in range(1, cfg.NUM_CLASSES)
    }

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            outputs = model(images)

            pred = outputs[0]
            target = targets[0]

            image_stats = evaluate_one_image(
                pred=pred,
                target=target,
                num_classes=cfg.NUM_CLASSES,
                iou_thresh=IOU_THRESH,
                score_thresh=SCORE_THRESH,
            )

            merge_stats(total_stats, image_stats)

    metrics_df = compute_metrics(total_stats)

    out_csv = cfg.RUN_DIR / "val_detection_metrics_iou0.5.csv"
    metrics_df.to_csv(out_csv, index=False)

    print("\nValidation detection metrics")
    print(metrics_df)

    print("\nMean precision:", metrics_df["precision@0.5"].mean())
    print("Mean recall:", metrics_df["recall@0.5"].mean())
    print("Mean F1:", metrics_df["f1@0.5"].mean())

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()

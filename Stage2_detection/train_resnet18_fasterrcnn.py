import time
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm import tqdm

from Stage2_detection import config2 as cfg
from Stage2_detection.dataset_bhx import BHXDetectionDataset, detection_collate_fn
from Stage2_detection.model_resnet18_fasterrcnn import build_resnet18_fasterrcnn


COCO_IOU_THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
]


def make_subset(dataset, max_images):
    if max_images is None:
        return dataset

    max_images = int(max_images)
    if max_images <= 0 or max_images >= len(dataset):
        return dataset

    return Subset(dataset, list(range(max_images)))


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    total_batches = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())

        if not torch.isfinite(loss):
            print("Non-finite loss detected:", loss.item())
            print(loss_dict)
            raise RuntimeError("Training stopped because loss is not finite.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_idx % 50 == 0 or batch_idx == 1:
            elapsed_min = (time.time() - start_time) / 60.0
            loss_items = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"avg_loss={total_loss / total_batches:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6g} | "
                f"time={elapsed_min:.1f} min | "
                f"{loss_items}"
            )

    return total_loss / max(total_batches, 1)


def voc_ap_101(recalls, precisions):
    """
    Compute AP using 101-point interpolation.
    """
    if len(recalls) == 0:
        return 0.0

    ap = 0.0

    for t in torch.linspace(0, 1, 101):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else torch.tensor(0.0)
        ap += float(p.item()) / 101.0

    return ap


def compute_ap_for_class(preds, gt_dict, iou_thresh):
    """
    Compute AP for one class at one IoU threshold.
    """
    num_gt = sum(len(v) for v in gt_dict.values())

    if num_gt == 0:
        return 0.0

    if len(preds) == 0:
        return 0.0

    matched = {
        image_id: torch.zeros(len(boxes), dtype=torch.bool)
        for image_id, boxes in gt_dict.items()
    }

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    tp = torch.zeros(len(preds), dtype=torch.float32)
    fp = torch.zeros(len(preds), dtype=torch.float32)

    for i, pred in enumerate(preds):
        image_id = pred["image_id"]
        pred_box = pred["box"].unsqueeze(0)

        if image_id not in gt_dict or len(gt_dict[image_id]) == 0:
            fp[i] = 1.0
            continue

        gt_boxes = torch.stack(gt_dict[image_id], dim=0)
        ious = box_iou(pred_box, gt_boxes).squeeze(0)

        best_iou, best_gt_idx = torch.max(ious, dim=0)
        best_iou = float(best_iou.item())
        best_gt_idx = int(best_gt_idx.item())

        if best_iou >= iou_thresh and not matched[image_id][best_gt_idx]:
            tp[i] = 1.0
            matched[image_id][best_gt_idx] = True
        else:
            fp[i] = 1.0

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    recalls = cum_tp / max(num_gt, 1)
    precisions = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)

    return voc_ap_101(recalls, precisions)


def evaluate_map_coco(model, loader, device, num_classes):
    """
    Compute:
        mAP@0.5
        COCO-style mAP@0.5:0.95

    IoU thresholds:
        0.50, 0.55, ..., 0.95
    """
    model.eval()

    score_min = getattr(cfg, "MAP_SCORE_MIN", 0.001)

    gt_by_class = {
        cls_id: defaultdict(list)
        for cls_id in range(1, num_classes)
    }

    pred_by_class = {
        cls_id: []
        for cls_id in range(1, num_classes)
    }

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating mAP"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].item())

                gt_boxes = target["boxes"].detach().cpu()
                gt_labels = target["labels"].detach().cpu()

                for cls_id in range(1, num_classes):
                    mask = gt_labels == cls_id
                    if mask.any():
                        gt_by_class[cls_id][image_id].extend(gt_boxes[mask])

                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()

                keep = pred_scores >= score_min
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    cls_id = int(label.item())

                    if 1 <= cls_id < num_classes:
                        pred_by_class[cls_id].append(
                            {
                                "image_id": image_id,
                                "score": float(score.item()),
                                "box": box,
                            }
                        )

    rows = []

    for cls_id in range(1, num_classes):
        gt_dict = gt_by_class[cls_id]
        preds = pred_by_class[cls_id]

        num_gt = sum(len(v) for v in gt_dict.values())

        row = {
            "class_id": cls_id,
            "class_name": cfg.ID_TO_CLASS.get(cls_id, str(cls_id)),
            "num_gt": num_gt,
            "num_pred": len(preds),
        }

        ap_values = []

        for iou in COCO_IOU_THRESHOLDS:
            ap = compute_ap_for_class(
                preds=preds,
                gt_dict=gt_dict,
                iou_thresh=iou,
            )

            row[f"ap@{iou:.2f}"] = ap
            ap_values.append(ap)

        row["ap50"] = row["ap@0.50"]
        row["ap50_95"] = sum(ap_values) / len(ap_values)

        rows.append(row)

    per_class_df = pd.DataFrame(rows)

    map50 = per_class_df["ap50"].mean()
    map50_95 = per_class_df["ap50_95"].mean()

    summary = {
        "map50": float(map50),
        "map50_95": float(map50_95),
    }

    return summary, per_class_df


def build_scheduler(optimizer):
    scheduler_name = getattr(cfg, "SCHEDULER_NAME", "cosine")

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.EPOCHS,
            eta_min=getattr(cfg, "MIN_LR", 1e-6),
        )

    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5,
        )

    return None


def load_history(history_path):
    if history_path.exists():
        return pd.read_csv(history_path).to_dict("records")

    return []


def main():
    device = torch.device(cfg.DEVICE)
    print("Using device:", device)

    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)

    history_path = cfg.RUN_DIR / getattr(cfg, "HISTORY_CSV_NAME", "history.csv")
    last_checkpoint_path = cfg.RUN_DIR / getattr(cfg, "LAST_CHECKPOINT_NAME", "last_checkpoint.pth")

    best_detector_path = cfg.RUN_DIR / getattr(cfg, "BEST_DETECTOR_NAME", "best_detector.pth")
    best_coco_detector_path = cfg.RUN_DIR / getattr(
        cfg,
        "BEST_COCO_DETECTOR_NAME",
        "best_detector_coco.pth",
    )

    best_map_csv_path = cfg.RUN_DIR / "best_val_map_per_class.csv"
    best_coco_map_csv_path = cfg.RUN_DIR / "best_coco_val_map_per_class.csv"

    train_dataset = BHXDetectionDataset(
        csv_path=cfg.TRAIN_CSV,
        image_size=cfg.IMAGE_SIZE,
        window_center=cfg.WINDOW_CENTER,
        window_width=cfg.WINDOW_WIDTH,
    )

    val_dataset = BHXDetectionDataset(
        csv_path=cfg.VAL_CSV,
        image_size=cfg.IMAGE_SIZE,
        window_center=cfg.WINDOW_CENTER,
        window_width=cfg.WINDOW_WIDTH,
    )

    train_dataset = make_subset(
        train_dataset,
        getattr(cfg, "MAX_TRAIN_IMAGES", None),
    )

    val_dataset = make_subset(
        val_dataset,
        getattr(cfg, "MAX_VAL_IMAGES", None),
    )

    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    print("Run dir:", cfg.RUN_DIR)
    print("RSNA checkpoint:", getattr(cfg, "RSNA_BEST_MODEL_PATH", None))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

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

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer)

    start_epoch = 1

    best_train_loss = float("inf")
    best_map50 = -1.0
    best_map50_95 = -1.0

    no_improve_eval_count = 0
    best_epoch_map50 = 0

    history = load_history(history_path)

    auto_resume = getattr(cfg, "AUTO_RESUME", True)

    if auto_resume and last_checkpoint_path.exists():
        print("Auto-resume from:", last_checkpoint_path)

        checkpoint = torch.load(last_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1

        best_train_loss = float(checkpoint.get("best_train_loss", best_train_loss))
        best_map50 = float(checkpoint.get("best_map50", best_map50))
        best_map50_95 = float(checkpoint.get("best_map50_95", best_map50_95))

        no_improve_eval_count = int(
            checkpoint.get("no_improve_eval_count", no_improve_eval_count)
        )

        best_epoch_map50 = int(
            checkpoint.get("best_epoch_map50", best_epoch_map50)
        )

        print("Resumed from epoch:", checkpoint["epoch"])
        print("Next epoch:", start_epoch)
        print("Best train loss so far:", best_train_loss)
        print("Best mAP@0.5 so far:", best_map50)
        print("Best COCO mAP@0.5:0.95 so far:", best_map50_95)
        print("No-improvement eval count:", no_improve_eval_count)
        print("Best mAP@0.5 epoch:", best_epoch_map50)
    else:
        print("Starting new training run.")

    if start_epoch > cfg.EPOCHS:
        print("Training is already complete.")
        return

    eval_every = getattr(cfg, "EVAL_EVERY", 5)

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        map_summary = {
            "map50": None,
            "map50_95": None,
        }

        should_eval = (
            epoch == 1
            or epoch % eval_every == 0
            or epoch == cfg.EPOCHS
        )

        if should_eval:
            map_summary, per_class_map = evaluate_map_coco(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=cfg.NUM_CLASSES,
            )

            print("\nValidation AP per class")
            print(
                per_class_map[
                    [
                        "class_id",
                        "class_name",
                        "num_gt",
                        "num_pred",
                        "ap50",
                        "ap50_95",
                    ]
                ]
            )

            print(f"Validation mAP@0.5: {map_summary['map50']:.6f}")
            print(f"Validation COCO mAP@0.5:0.95: {map_summary['map50_95']:.6f}")

            map_csv_path = cfg.RUN_DIR / f"val_map_epoch_{epoch:03d}.csv"
            per_class_map.to_csv(map_csv_path, index=False)
            print("Saved per-class AP:", map_csv_path)

        print(
            f"Epoch {epoch} finished | "
            f"train_loss={train_loss:.4f} | "
            f"lr={current_lr:.6g} | "
            f"map50={map_summary['map50']} | "
            f"map50_95={map_summary['map50_95']}"
        )

        early_stop_min_delta = getattr(cfg, "EARLY_STOP_MIN_DELTA", 1e-4)

        is_best_map50 = (
            map_summary["map50"] is not None
            and map_summary["map50"] > best_map50 + early_stop_min_delta
        )

        is_best_map50_95 = (
            map_summary["map50_95"] is not None
            and map_summary["map50_95"] > best_map50_95 + early_stop_min_delta
        )

        if is_best_map50:
            best_map50 = map_summary["map50"]
            torch.save(model.state_dict(), best_detector_path)
            per_class_map.to_csv(best_map_csv_path, index=False)
            print("Saved new best detector by mAP@0.5:", best_detector_path)
            print("Saved best mAP per-class CSV:", best_map_csv_path)

        if is_best_map50_95:
            best_map50_95 = map_summary["map50_95"]
            torch.save(model.state_dict(), best_coco_detector_path)
            per_class_map.to_csv(best_coco_map_csv_path, index=False)
            print("Saved new best detector by COCO mAP@0.5:0.95:", best_coco_detector_path)
            print("Saved best COCO mAP per-class CSV:", best_coco_map_csv_path)

        if map_summary["map50"] is not None:
            if is_best_map50:
                no_improve_eval_count = 0
                best_epoch_map50 = epoch
            else:
                no_improve_eval_count += 1

            print(
                f"Early stopping monitor | "
                f"best_map50={best_map50:.6f} | "
                f"best_epoch={best_epoch_map50} | "
                f"no_improve_eval_count={no_improve_eval_count}/"
                f"{getattr(cfg, 'EARLY_STOP_PATIENCE', 6)}"
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "train_loss": train_loss,
            "best_train_loss": best_train_loss,
            "best_map50": best_map50,
            "best_map50_95": best_map50_95,
            "no_improve_eval_count": no_improve_eval_count,
            "best_epoch_map50": best_epoch_map50,
            "num_classes": cfg.NUM_CLASSES,
            "image_size": cfg.IMAGE_SIZE,
            "rsna_checkpoint_path": getattr(cfg, "RSNA_BEST_MODEL_PATH", None),
        }

        torch.save(checkpoint, last_checkpoint_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": current_lr,
                "map50": map_summary["map50"],
                "map50_95": map_summary["map50_95"],
                "best_map50": best_map50,
                "best_map50_95": best_map50_95,
                "is_best_map50": int(is_best_map50),
                "is_best_map50_95": int(is_best_map50_95),
                "no_improve_eval_count": no_improve_eval_count,
                "best_epoch_map50": best_epoch_map50,
            }
        )

        pd.DataFrame(history).to_csv(history_path, index=False)

        print("Saved last checkpoint:", last_checkpoint_path)
        print("Saved history:", history_path)

        early_stop = getattr(cfg, "EARLY_STOP", True)
        early_stop_patience = getattr(cfg, "EARLY_STOP_PATIENCE", 6)

        if (
            early_stop
            and map_summary["map50"] is not None
            and no_improve_eval_count >= early_stop_patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best mAP@0.5 = {best_map50:.6f} at epoch {best_epoch_map50}."
            )
            break


if __name__ == "__main__":
    main()
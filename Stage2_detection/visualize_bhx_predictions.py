import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from Stage2_detection import config2 as cfg
from Stage2_detection.dataset_bhx import BHXDetectionDataset
from Stage2_detection.model_resnet18_fasterrcnn import build_resnet18_fasterrcnn


CLASS_SHORT = {
    "epidural": "EDH",
    "intraparenchymal": "IPH",
    "intraventricular": "IVH",
    "subarachnoid": "SAH",
    "subdural": "SDH",
}


def short_class_name(cls_name):
    return CLASS_SHORT.get(cls_name.lower(), cls_name)


def draw_box(ax, box, label_text, edgecolor, linestyle="-", linewidth=3):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = x2 - x1
    h = y2 - y1

    rect = Rectangle(
        (x1, y1),
        w,
        h,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)

    ax.text(
        x1,
        max(y1 - 8, 0),
        label_text,
        fontsize=14,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor=edgecolor,
            alpha=0.85,
            pad=2,
            edgecolor="none",
        ),
    )


def get_checkpoint_path(checkpoint_mode):
    if checkpoint_mode == "map50":
        return cfg.RUN_DIR / getattr(cfg, "BEST_DETECTOR_NAME", "best_detector.pth")

    if checkpoint_mode == "coco":
        return cfg.RUN_DIR / getattr(
            cfg,
            "BEST_COCO_DETECTOR_NAME",
            "best_detector_coco.pth",
        )

    return Path(checkpoint_mode)


def visualize_one_sample(
    model,
    dataset,
    idx,
    device,
    output_dir,
    score_thresh=0.5,
    max_predictions=20,
):
    image, target = dataset[idx]

    with torch.no_grad():
        output = model([image.to(device)])[0]

    img_np = image[0].detach().cpu().numpy()

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    pred_boxes = output["boxes"].detach().cpu()
    pred_labels = output["labels"].detach().cpu()
    pred_scores = output["scores"].detach().cpu()

    keep = pred_scores >= score_thresh
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    if len(pred_scores) > max_predictions:
        order = torch.argsort(pred_scores, descending=True)[:max_predictions]
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

    sop_uid = dataset.sop_uids[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    ax.set_title(
        f"BHX validation sample {idx}\n"
        f"GT: green | Prediction: red | score >= {score_thresh}",
        fontsize=16,
        fontweight="bold",
    )

    for box, label in zip(gt_boxes, gt_labels):
        cls_id = int(label.item())
        cls_name = cfg.ID_TO_CLASS.get(cls_id, str(cls_id))
        cls_name = short_class_name(cls_name)

        draw_box(
            ax=ax,
            box=box,
            label_text=f"GT: {cls_name}",
            edgecolor="lime",
            linestyle="-",
            linewidth=3,
        )

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        cls_id = int(label.item())
        cls_name = cfg.ID_TO_CLASS.get(cls_id, str(cls_id))
        cls_name = short_class_name(cls_name)

        draw_box(
            ax=ax,
            box=box,
            label_text=f"Pred: {cls_name} {float(score):.2f}",
            edgecolor="red",
            linestyle="--",
            linewidth=3,
        )

    output_path = output_dir / f"val_{idx:05d}_pred.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "idx": idx,
        "sop_uid": sop_uid,
        "num_gt": len(gt_boxes),
        "num_pred": len(pred_boxes),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="map50",
        help=(
            "Checkpoint to visualize. "
            "Use 'map50' for best_detector.pth, "
            "'coco' for best_detector_coco.pth, "
            "or provide a full checkpoint path."
        ),
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Number of validation images to visualize.",
    )

    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for predicted boxes.",
    )

    parser.add_argument(
        "--max-predictions",
        type=int,
        default=20,
        help="Maximum number of predicted boxes drawn per image.",
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample validation images instead of using the first images.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    device = torch.device(cfg.DEVICE)
    print("Using device:", device)

    checkpoint_path = get_checkpoint_path(args.checkpoint)
    print("Checkpoint:", checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = cfg.RUN_DIR / f"visualizations_{args.checkpoint}_score{args.score_thresh}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BHXDetectionDataset(
        csv_path=cfg.VAL_CSV,
        image_size=cfg.IMAGE_SIZE,
        window_center=cfg.WINDOW_CENTER,
        window_width=cfg.WINDOW_WIDTH,
    )

    print("Validation images:", len(dataset))
    print("Output dir:", output_dir)

    model = build_resnet18_fasterrcnn(
        num_classes=cfg.NUM_CLASSES,
        rsna_checkpoint_path=getattr(cfg, "RSNA_BEST_MODEL_PATH", None),
        image_size=cfg.IMAGE_SIZE,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    indices = list(range(len(dataset)))

    if args.random:
        random.seed(args.seed)
        random.shuffle(indices)

    indices = indices[: args.num_images]

    rows = []

    for idx in indices:
        info = visualize_one_sample(
            model=model,
            dataset=dataset,
            idx=idx,
            device=device,
            output_dir=output_dir,
            score_thresh=args.score_thresh,
            max_predictions=args.max_predictions,
        )
        rows.append(info)
        print(
            f"Saved {info['output_path']} | "
            f"GT boxes={info['num_gt']} | Pred boxes={info['num_pred']}"
        )

    print("\nDone.")
    print("Saved visualizations to:", output_dir)


if __name__ == "__main__":
    main()
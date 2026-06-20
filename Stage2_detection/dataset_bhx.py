from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Stage2_detection.utils_dicom import read_dicom_windowed


class BHXDetectionDataset(Dataset):
    """
    BHX dataset for object detection.

    Each item returns:
        image: Tensor [1, H, W]
        target: dictionary for torchvision detection models
    """

    def __init__(
        self,
        csv_path,
        image_size=512,
        window_center=40,
        window_width=80,
    ):
        self.csv_path = Path(csv_path)
        self.image_size = image_size
        self.window_center = window_center
        self.window_width = window_width

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = {
            "SOPInstanceUID",
            "path",
            "Rows",
            "Columns",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
        }

        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        self.sop_uids = sorted(self.df["SOPInstanceUID"].unique().tolist())

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, idx):
        sop_uid = self.sop_uids[idx]
        rows = self.df[self.df["SOPInstanceUID"] == sop_uid]

        dicom_path = rows.iloc[0]["path"]

        img = read_dicom_windowed(
            dicom_path,
            center=self.window_center,
            width=self.window_width,
        )

        original_h, original_w = img.shape

        # Keep image as 1-channel to match the RSNA ResNet18-KAN backbone.
        # img.copy() avoids the non-writable NumPy array warning.
        image = torch.from_numpy(img.copy()).float().unsqueeze(0)

        if self.image_size is not None:
            image = image.unsqueeze(0)
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            image = image.squeeze(0)

            scale_x = self.image_size / original_w
            scale_y = self.image_size / original_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(
            dtype="float32",
            copy=True,
        )
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        labels = torch.as_tensor(
            rows["class_id"].to_numpy(copy=True),
            dtype=torch.int64,
        )

        # Clip boxes to image boundaries
        _, h, w = image.shape

        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)

        # Remove invalid boxes if any exist after clipping
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        labels = labels[valid]

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        return image, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
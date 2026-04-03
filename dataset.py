
import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from preprocess import preprocess_ct_slice
from config import Config


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dicom_to_hu(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Convert DICOM pixel_array to HU using RescaleSlope and RescaleIntercept.
    """
    image = dcm.pixel_array.astype(np.float32)

    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

    image = image * slope + intercept
    return image


def build_multilabel_df(csv_path: str) -> pd.DataFrame:
    """
    Convert RSNA long-format CSV into wide multi-label format by image.
    Output columns:
      image_id, any, epidural, intraparenchymal, intraventricular,
      subarachnoid, subdural, filepath
    """
    df = pd.read_csv(csv_path)

    # Split "ID" into image_id + subtype
    split_df = df["ID"].str.rsplit("_", n=1, expand=True)
    df["image_id"] = split_df[0]
    df["subtype"] = split_df[1]

    # Handle duplicate (image_id, subtype) safely
    dup_count = df.duplicated(subset=["image_id", "subtype"]).sum()
    print(f"Duplicate (image_id, subtype) rows: {dup_count}")

    pivot_df = df.pivot_table(
        index="image_id",
        columns="subtype",
        values="Label",
        aggfunc="max"
    ).reset_index()

    # Fill missing subtype cols if needed
    expected_cols = [
        "any",
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
    ]
    for col in expected_cols:
        if col not in pivot_df.columns:
            pivot_df[col] = 0.0

    # Ensure float32
    for col in expected_cols:
        pivot_df[col] = pivot_df[col].astype(np.float32)

    # DICOM filename
    pivot_df["filepath"] = pivot_df["image_id"].apply(
        lambda x: os.path.join(Config.TRAIN_DIR, f"{x}.dcm")
    )

    return pivot_df


class RSNADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config: Config):
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.label_cols = config.LABEL_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dcm_path = row["filepath"]

        dcm = pydicom.dcmread(dcm_path)
        image_hu = dicom_to_hu(dcm)

        image = preprocess_ct_slice(
            image_hu=image_hu,
            window_center=self.config.WINDOW_CENTER,
            window_width=self.config.WINDOW_WIDTH,
            image_size=self.config.IMAGE_SIZE,
        )

        # [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)

        target = row[self.label_cols].values.astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "image_id": row["image_id"],
        }


def build_train_val_dataframes(config: Config):
    df = build_multilabel_df(config.CSV_PATH)

    # Keep only files that actually exist
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    if config.DEBUG:
        df = df.sample(
            n=min(config.DEBUG_SAMPLES, len(df)),
            random_state=config.SEED
        ).reset_index(drop=True)

    # For stratify, use "any" only to keep split simple
    train_df, val_df = train_test_split(
        df,
        test_size=config.VAL_RATIO,
        random_state=config.SEED,
        stratify=df["any"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
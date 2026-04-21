import os
import random
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from preprocess import preprocess_ct_slice
from config import Config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dicom_to_hu(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Convert DICOM pixel_array to HU using RescaleSlope and RescaleIntercept.
    """
    image = dcm.pixel_array.astype(np.float32)

    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

    image = image * slope + intercept
    return image


def is_valid_dicom(path: str) -> bool:
    """
    Check whether a DICOM file can be read and converted to HU.
    This helps remove corrupted files before training starts.
    """
    try:
        dcm = pydicom.dcmread(path)
        _ = dicom_to_hu(dcm)
        return True
    except Exception:
        return False


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

    dup_count = df.duplicated(subset=["image_id", "subtype"]).sum()
    print(f"Duplicate (image_id, subtype) rows: {dup_count}")

    pivot_df = df.pivot_table(
        index="image_id",
        columns="subtype",
        values="Label",
        aggfunc="max"
    ).reset_index()

    expected_cols = Config.LABEL_COLS
    for col in expected_cols:
        if col not in pivot_df.columns:
            pivot_df[col] = 0.0

    pivot_df[expected_cols] = pivot_df[expected_cols].fillna(0.0).astype(np.float32)

    pivot_df["filepath"] = pivot_df["image_id"].apply(
        lambda x: os.path.join(Config.TRAIN_DIR, f"{x}.dcm")
    )

    pivot_df = pivot_df.sort_values("image_id").reset_index(drop=True)
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

        try:
            dcm = pydicom.dcmread(dcm_path)
            image_hu = dicom_to_hu(dcm)

            image = preprocess_ct_slice(
                image_hu=image_hu,
                window_center=self.config.WINDOW_CENTER,
                window_width=self.config.WINDOW_WIDTH,
                image_size=self.config.IMAGE_SIZE,
            )

            # [H, W] -> [1, H, W]
            image = np.expand_dims(image, axis=0).astype(np.float32)
            image = np.ascontiguousarray(image)

            target = row[self.label_cols].to_numpy(dtype=np.float32, copy=True)

            return {
                "image": torch.from_numpy(image),
                "target": torch.from_numpy(target),
                "image_id": row["image_id"],
            }

        except Exception as e:
            raise RuntimeError(f"Error reading DICOM: {dcm_path}") from e


def build_train_val_dataframes(config: Config):
    df = build_multilabel_df(config.CSV_PATH)

    # Keep only files that actually exist
    df = df[df["filepath"].map(os.path.exists)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid DICOM files found after filepath filtering.")

    if config.DEBUG and config.DEBUG_SAMPLES is not None:
        df = df.sample(
            n=min(config.DEBUG_SAMPLES, len(df)),
            random_state=config.SEED
        ).reset_index(drop=True)

    # Remove corrupted / unreadable DICOM files before split
    print("Checking DICOM validity...")
    valid_mask = df["filepath"].map(is_valid_dicom)
    num_invalid = int((~valid_mask).sum())

    if num_invalid > 0:
        print(f"Found {num_invalid} invalid/corrupted DICOM files. They will be removed.")

        invalid_df = df.loc[~valid_mask, ["image_id", "filepath"]].copy()

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        invalid_csv_path = os.path.join(config.OUTPUT_DIR, "invalid_dicoms.csv")
        invalid_df.to_csv(invalid_csv_path, index=False)

        print(f"Saved invalid file list to: {invalid_csv_path}")

    df = df.loc[valid_mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("All DICOM files were invalid after validation.")

    # Stratify by 'any' if both classes are present
    stratify_col = df["any"] if df["any"].nunique() > 1 else None

    train_df, val_df = train_test_split(
        df,
        test_size=config.VAL_RATIO,
        random_state=config.SEED,
        shuffle=True,
        stratify=stratify_col
    )

    print(f"Total valid samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Train positive rate (any): {train_df['any'].mean():.4f}")
    print(f"Val positive rate (any): {val_df['any'].mean():.4f}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

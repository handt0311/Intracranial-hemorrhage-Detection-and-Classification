import cv2
import numpy as np


def apply_window(image_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply CT windowing and clip to [lower, upper].
    """
    lower = center - width / 2.0
    upper = center + width / 2.0
    image = np.clip(image_hu, lower, upper)
    return image.astype(np.float32)


def normalize_by_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Normalize using the fixed CT window range, not per-image min/max.
    Output in [0, 1].
    """
    lower = center - width / 2.0
    upper = center + width / 2.0

    image = (image - lower) / (upper - lower)
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize to square size x size.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_ct_slice(
    image_hu: np.ndarray,
    window_center: float,
    window_width: float,
    image_size: int
) -> np.ndarray:
    """
    Full preprocessing:
    HU -> window -> fixed-range normalize -> resize
    """
    image_hu = np.nan_to_num(image_hu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    image = apply_window(image_hu, window_center, window_width)
    image = normalize_by_window(image, window_center, window_width)
    image = resize_image(image, image_size)

    return image.astype(np.float32)
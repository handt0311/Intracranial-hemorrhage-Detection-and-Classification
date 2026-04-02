import cv2
import numpy as np

def apply_window(image_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply CT windowing.
    """
    lower = center - width / 2
    upper = center + width / 2
    image = np.clip(image_hu, lower, upper)
    return image

def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """
    Normalize to [0, 1].
    """
    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)

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
    HU -> window -> normalize -> resize
    """
    image = apply_window(image_hu, window_center, window_width)
    image = normalize_minmax(image)
    image = resize_image(image, image_size)
    return image.astype(np.float32)
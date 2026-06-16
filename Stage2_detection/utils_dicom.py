import numpy as np
import pydicom


def apply_window(img, center=40, width=80):
    """Apply CT windowing and normalize image values to [0, 1]."""
    img = img.astype(np.float32)

    low = center - width / 2
    high = center + width / 2

    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-6)

    return img


def read_dicom_windowed(path, center=40, width=80):
    """Read a DICOM file, convert pixel values to HU, and apply CT windowing."""
    ds = pydicom.dcmread(str(path))

    img = ds.pixel_array.astype(np.float32)

    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))

    img = img * slope + intercept
    img = apply_window(img, center=center, width=width)

    return img
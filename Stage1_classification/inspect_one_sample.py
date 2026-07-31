import matplotlib.pyplot as plt
import pydicom

from config import Config
from dataset import build_binary_labels, dicom_to_hu
from preprocess import preprocess_ct_slice


def main():
    config = Config()
    df = build_binary_labels(config.CSV_PATH, config.POSITIVE_CLASS)

    row = df.iloc[0]
    dcm = pydicom.dcmread(row["filepath"])
    image_hu = dicom_to_hu(dcm)

    processed = preprocess_ct_slice(
        image_hu=image_hu,
        window_center=config.WINDOW_CENTER,
        window_width=config.WINDOW_WIDTH,
        image_size=config.IMAGE_SIZE
    )

    print("Image ID:", row["image_id"])
    print("Target:", row["target"])
    print("Original shape:", image_hu.shape)
    print("Processed shape:", processed.shape)
    print("HU min/max:", image_hu.min(), image_hu.max())
    print("Processed min/max:", processed.min(), processed.max())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_hu, cmap="gray")
    plt.title("Raw HU")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap="gray")
    plt.title("Processed")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
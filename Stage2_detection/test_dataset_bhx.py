from torch.utils.data import DataLoader

from Stage2_detection import config2 as cfg
from Stage2_detection.dataset_bhx import BHXDetectionDataset, detection_collate_fn


def main():
    dataset = BHXDetectionDataset(
        csv_path=cfg.TRAIN_CSV,
        image_size=cfg.IMAGE_SIZE,
        window_center=cfg.WINDOW_CENTER,
        window_width=cfg.WINDOW_WIDTH,
    )

    print("Dataset size:", len(dataset))

    image, target = dataset[0]

    print("\nSingle sample")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image min/max:", image.min().item(), image.max().item())
    print("Boxes shape:", target["boxes"].shape)
    print("Labels:", target["labels"])
    print("Boxes:", target["boxes"][:5])

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    images, targets = next(iter(loader))

    print("\nBatch")
    print("Number of images:", len(images))
    print("Image 0 shape:", images[0].shape)
    print("Target 0 keys:", targets[0].keys())
    print("Target 0 boxes:", targets[0]["boxes"].shape)


if __name__ == "__main__":
    main()
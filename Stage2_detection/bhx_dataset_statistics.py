import os
import pandas as pd

DATA_DIR = "/storage/student5/handt/bhx/bhx_train_val"

TRAIN_CSV = os.path.join(DATA_DIR, "bhx_train_boxes_5class.csv")
VAL_CSV = os.path.join(DATA_DIR, "bhx_val_boxes_5class.csv")

OUT_CSV = os.path.join(DATA_DIR, "bhx_dataset_statistics.csv")


def count_stats(df):
    total_images = df["SOPInstanceUID"].nunique()
    total_boxes = len(df)

    rows = {}

    for cls in sorted(df["class_name"].unique()):
        sub = df[df["class_name"] == cls]
        rows[cls] = {
            "images": sub["SOPInstanceUID"].nunique(),
            "boxes": len(sub),
        }

    return total_images, total_boxes, rows


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_total_images, train_total_boxes, train_stats = count_stats(train_df)
    val_total_images, val_total_boxes, val_stats = count_stats(val_df)

    class_names = sorted(set(train_stats.keys()) | set(val_stats.keys()))

    rows = []

    for cls in class_names:
        train_images = train_stats.get(cls, {}).get("images", 0)
        val_images = val_stats.get(cls, {}).get("images", 0)

        train_boxes = train_stats.get(cls, {}).get("boxes", 0)
        val_boxes = val_stats.get(cls, {}).get("boxes", 0)

        rows.append({
            "Class": cls,
            "Train Images": train_images,
            "Val Images": val_images,
            "Total Images": train_images + val_images,
            "Train Boxes": train_boxes,
            "Val Boxes": val_boxes,
            "Total Boxes": train_boxes + val_boxes,
        })

    rows.append({
        "Class": "Total",
        "Train Images": train_total_images,
        "Val Images": val_total_images,
        "Total Images": train_total_images + val_total_images,
        "Train Boxes": train_total_boxes,
        "Val Boxes": val_total_boxes,
        "Total Boxes": train_total_boxes + val_total_boxes,
    })

    out_df = pd.DataFrame(rows)

    print("\nBHX Dataset Statistics")
    print("=" * 100)
    print(out_df.to_string(index=False))

    out_df.to_csv(OUT_CSV, index=False)
    print("\nSaved to:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()

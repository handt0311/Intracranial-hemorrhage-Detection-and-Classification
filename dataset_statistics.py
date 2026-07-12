import os
import pandas as pd

from config import Config
from dataset import build_train_val_dataframes


def count_split(df, label_cols):
    stats = {}

    stats["Total images"] = len(df)
    stats["Normal"] = int((df["any"] == 0).sum())

    for c in label_cols:
        stats[c] = int((df[c] == 1).sum())

    return stats


def main():
    config = Config()

    train_df, val_df = build_train_val_dataframes(config)

    label_cols = config.LABEL_COLS

    train_stats = count_split(train_df, label_cols)
    val_stats = count_split(val_df, label_cols)

    rows = []

    row_names = ["Total images", "Normal"] + label_cols

    for name in row_names:
        train_count = train_stats[name]
        val_count = val_stats[name]
        total_count = train_count + val_count

        rows.append({
            "Class": name,
            "Train": train_count,
            "Validation": val_count,
            "Total": total_count
        })

    stats_df = pd.DataFrame(rows)

    print("\nRSNA Dataset Statistics")
    print("=" * 60)
    print(stats_df.to_string(index=False))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(config.OUTPUT_DIR, "rsna_dataset_statistics.csv")
    stats_df.to_csv(save_path, index=False)

    print("\nSaved CSV to:")
    print(save_path)


if __name__ == "__main__":
    main()

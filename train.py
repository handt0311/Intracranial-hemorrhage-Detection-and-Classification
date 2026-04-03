
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config import Config
from dataset import RSNADataset, build_train_val_dataframes, set_seed
from model import RSNAClassifier


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_multilabel_auc(y_true, y_prob, label_names):
    auc_dict = {}
    auc_values = []

    for i, name in enumerate(label_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")
        auc_dict[name] = auc
        if not np.isnan(auc):
            auc_values.append(auc)

    mean_auc = float(np.mean(auc_values)) if len(auc_values) > 0 else float("nan")
    return mean_auc, auc_dict


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)   # [B, 6]

        optimizer.zero_grad()
        logits = model(images)                 # [B, 6]
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, label_names):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets, axis=0)   # [N, 6]
    all_probs = np.concatenate(all_probs, axis=0)       # [N, 6]

    mean_auc, auc_dict = compute_multilabel_auc(all_targets, all_probs, label_names)

    return val_loss, mean_auc, auc_dict


def main():
    config = Config()
    ensure_dir(config.OUTPUT_DIR)
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building dataframes...")
    train_df, val_df = build_train_val_dataframes(config)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    train_dataset = RSNADataset(train_df, config)
    val_dataset = RSNADataset(val_df, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    model = RSNAClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    best_auc = -1.0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mean_auc, val_auc_dict = validate(
            model, val_loader, criterion, device, config.LABEL_COLS
        )

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mean_auc={val_mean_auc:.4f}"
        )

        for label_name in config.LABEL_COLS:
            print(f"  {label_name}: {val_auc_dict[label_name]:.4f}")

        if not np.isnan(val_mean_auc) and val_mean_auc > best_auc:
            best_auc = val_mean_auc
            save_path = os.path.join(config.OUTPUT_DIR, "best_model_multilabel.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main()
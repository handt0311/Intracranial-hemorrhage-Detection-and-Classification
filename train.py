import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from config import Config
from dataset import RSNADataset, build_train_val_dataframes, set_seed
from model import RSNAClassifier


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log_message(message: str, log_path: str):
    print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_config_snapshot(config, save_path: str):
    config_dict = {}
    for key in dir(config):
        if key.isupper():
            value = getattr(config, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                config_dict[key] = value
            elif isinstance(value, (list, tuple)):
                config_dict[key] = list(value)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


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


def save_val_predictions_csv(image_ids, y_true, y_prob, label_names, save_path: str):
    data = {"image_id": image_ids}

    for i, label in enumerate(label_names):
        data[f"{label}_true"] = y_true[:, i]
        data[f"{label}_prob"] = y_prob[:, i]
        data[f"{label}_pred"] = (y_prob[:, i] >= 0.5).astype(np.int32)

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def build_scheduler(optimizer, config):
    scheduler_name = getattr(config, "SCHEDULER_NAME", None)

    if scheduler_name is None:
        return None

    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.SCHEDULER_MODE,
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            threshold=config.SCHEDULER_THRESHOLD,
            min_lr=config.MIN_LR,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
    scaler,
    use_amp,
    accumulation_steps,
    clip_grad_norm,
):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{total_epochs}", leave=True)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress_bar):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        loss_for_backward = loss / accumulation_steps
        scaler.scale(loss_for_backward).backward()

        should_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(loader))

        if should_step:
            if clip_grad_norm is not None and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, label_names, epoch, total_epochs, use_amp):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []
    all_image_ids = []

    progress_bar = tqdm(loader, desc=f"Val Epoch {epoch+1}/{total_epochs}", leave=True)

    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_image_ids.extend(image_ids)

        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    val_loss = total_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    mean_auc, auc_dict = compute_multilabel_auc(all_targets, all_probs, label_names)

    return val_loss, mean_auc, auc_dict, all_targets, all_probs, all_image_ids


def main():
    config = Config()
    ensure_dir(config.OUTPUT_DIR)

    log_path = os.path.join(config.OUTPUT_DIR, config.TRAIN_LOG_NAME)
    config_snapshot_path = os.path.join(config.OUTPUT_DIR, "config_snapshot.json")
    history_csv_path = os.path.join(config.OUTPUT_DIR, config.HISTORY_CSV)
    best_model_path = os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_NAME)
    last_model_path = os.path.join(config.OUTPUT_DIR, config.LAST_MODEL_NAME)
    val_pred_path = os.path.join(config.OUTPUT_DIR, config.VAL_PRED_NAME)

    save_config_snapshot(config, config_snapshot_path)
    set_seed(config.SEED)

    use_cuda = torch.cuda.is_available() and config.DEVICE == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = bool(config.USE_AMP and use_cuda)

    log_message(f"Using device: {device}", log_path)
    log_message(f"AMP enabled: {use_amp}", log_path)
    log_message(f"Head type: {config.HEAD_TYPE}", log_path)
    log_message(f"Run name: {config.RUN_NAME}", log_path)
    log_message(f"Saving outputs to: {config.OUTPUT_DIR}", log_path)

    log_message("Building dataframes...", log_path)
    train_df, val_df = build_train_val_dataframes(config)
    log_message(f"Train samples: {len(train_df)}", log_path)
    log_message(f"Val samples: {len(val_df)}", log_path)

    train_dataset = RSNADataset(train_df, config)
    val_dataset = RSNADataset(val_df, config)

    persistent_workers = config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    pin_memory = config.PIN_MEMORY if use_cuda else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = RSNAClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        head_type=config.HEAD_TYPE,
        mlp_hidden_dim=config.MLP_HIDDEN_DIM,
        dropout=config.DROPOUT,
        kan_num_basis=config.KAN_NUM_BASIS,
        kan_hidden_dim=config.KAN_HIDDEN_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_auc = -1.0
    early_stop_counter = 0
    history = []

    for epoch in range(config.EPOCHS):
        current_lr = get_current_lr(optimizer)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=config.EPOCHS,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=config.ACCUMULATION_STEPS,
            clip_grad_norm=config.CLIP_GRAD_NORM,
        )

        val_loss, val_mean_auc, val_auc_dict, y_true, y_prob, image_ids = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            label_names=config.LABEL_COLS,
            epoch=epoch,
            total_epochs=config.EPOCHS,
            use_amp=use_amp,
        )

        if scheduler is not None:
            if config.SCHEDULER_NAME == "ReduceLROnPlateau":
                if not np.isnan(val_mean_auc):
                    scheduler.step(val_mean_auc)
            else:
                scheduler.step()

        log_message(
            (
                f"\nEpoch [{epoch+1}/{config.EPOCHS}] "
                f"lr={current_lr:.6g} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_mean_auc={val_mean_auc:.4f}"
            ),
            log_path,
        )

        for label_name in config.LABEL_COLS:
            label_auc = val_auc_dict[label_name]
            log_message(f"  {label_name}: {label_auc:.4f}", log_path)

        epoch_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auc": val_mean_auc,
        }
        for label_name in config.LABEL_COLS:
            epoch_record[f"auc_{label_name}"] = val_auc_dict[label_name]

        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        improved = (not np.isnan(val_mean_auc)) and (val_mean_auc > best_auc)

        if improved:
            best_auc = val_mean_auc
            early_stop_counter = 0

            torch.save(model.state_dict(), best_model_path)
            save_val_predictions_csv(
                image_ids=image_ids,
                y_true=y_true,
                y_prob=y_prob,
                label_names=config.LABEL_COLS,
                save_path=val_pred_path,
            )

            log_message(f"Saved best model to: {best_model_path}", log_path)
            log_message(f"Saved best val predictions to: {val_pred_path}", log_path)
            log_message(f"Best val_mean_auc so far: {best_auc:.4f}", log_path)
        else:
            early_stop_counter += 1
            log_message(
                f"No improvement. Early stopping counter: "
                f"{early_stop_counter}/{config.EARLY_STOPPING_PATIENCE}",
                log_path,
            )

        if config.SAVE_LAST:
            torch.save(model.state_dict(), last_model_path)

        if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
            log_message("Early stopping triggered.", log_path)
            break

    log_message("\nTraining finished.", log_path)
    log_message(f"Best val_mean_auc: {best_auc:.4f}", log_path)
    log_message(f"History saved to: {history_csv_path}", log_path)
    log_message(f"Last model saved to: {last_model_path}", log_path)


if __name__ == "__main__":
    main()
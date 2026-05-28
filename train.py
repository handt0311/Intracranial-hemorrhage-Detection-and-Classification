import os
import json
import random
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
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError as e:
        print(f"[WARN] Could not write log file {log_path}: {e}")


def save_config_snapshot(config, save_path: str):
    config_dict = {}
    for key in dir(config):
        if key.isupper():
            value = getattr(config, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                config_dict[key] = value
            elif isinstance(value, (list, tuple)):
                config_dict[key] = list(value)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[WARN] Could not save config snapshot to {save_path}: {e}")


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
    tmp_path = save_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, save_path)


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


def save_history_atomic(history, save_path: str):
    tmp_path = save_path + ".tmp"
    pd.DataFrame(history).to_csv(tmp_path, index=False)
    os.replace(tmp_path, save_path)


def save_state_dict_atomic(state_dict, save_path: str):
    tmp_path = save_path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, save_path)


def save_checkpoint_atomic(checkpoint: dict, save_path: str):
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)


def load_history_from_csv(history_csv_path: str):
    if not os.path.exists(history_csv_path):
        return []

    if os.path.getsize(history_csv_path) == 0:
        return []

    try:
        history_df = pd.read_csv(history_csv_path)
    except Exception:
        return []

    if history_df.empty:
        return []

    return history_df.to_dict("records")


def infer_best_auc_from_history(history):
    if len(history) == 0:
        return -1.0

    history_df = pd.DataFrame(history)
    if "val_mean_auc" not in history_df.columns:
        return -1.0

    values = pd.to_numeric(history_df["val_mean_auc"], errors="coerce").dropna()
    if len(values) == 0:
        return -1.0

    return float(values.max())


def infer_early_stop_counter_from_history(history):
    if len(history) == 0:
        return 0

    history_df = pd.DataFrame(history)
    if "val_mean_auc" not in history_df.columns:
        return 0

    auc_series = pd.to_numeric(history_df["val_mean_auc"], errors="coerce")
    if auc_series.dropna().empty:
        return 0

    best_idx = auc_series.idxmax()
    return int(len(history_df) - 1 - best_idx)


def get_rng_state(use_cuda: bool):
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if use_cuda and torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint: dict, use_cuda: bool):
    try:
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if use_cuda and torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    except Exception:
        # Nếu RNG state cũ không restore được thì vẫn tiếp tục train bình thường
        pass


def build_training_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    completed_epoch,
    best_auc,
    early_stop_counter,
    history,
    use_cuda,
):
    checkpoint = {
        "completed_epoch": int(completed_epoch),
        "best_auc": float(best_auc),
        "early_stop_counter": int(early_stop_counter),
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    checkpoint.update(get_rng_state(use_cuda=use_cuda))
    return checkpoint


def load_resume_state(
    config,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    use_cuda,
    history_csv_path,
    last_model_path,
    last_checkpoint_path,
    log_path,
):
    auto_resume = getattr(config, "AUTO_RESUME", True)
    resume_path = getattr(config, "RESUME_PATH", None)

    candidate_paths = []

    if resume_path:
        candidate_paths.append(resume_path)

    if auto_resume:
        candidate_paths.append(last_checkpoint_path)
        candidate_paths.append(last_model_path)

    # bỏ trùng nhưng vẫn giữ thứ tự
    seen = set()
    unique_candidate_paths = []
    for p in candidate_paths:
        if p and p not in seen:
            seen.add(p)
            unique_candidate_paths.append(p)

    default_state = {
        "start_epoch": 0,
        "best_auc": -1.0,
        "early_stop_counter": 0,
        "history": [],
        "resumed": False,
        "resume_source": None,
    }

    for path in unique_candidate_paths:
        if not os.path.isfile(path):
            continue

        log_message(f"Found resume candidate: {path}", log_path)
        checkpoint = torch.load(path, map_location=device)

        # Case 1: full checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

            if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if (
                scheduler is not None
                and "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"] is not None
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception:
                    pass

            restore_rng_state(checkpoint, use_cuda=use_cuda)

            history = checkpoint.get("history", None)
            if not isinstance(history, list):
                history = load_history_from_csv(history_csv_path)

            start_epoch = int(checkpoint.get("completed_epoch", 0))
            best_auc = float(checkpoint.get("best_auc", infer_best_auc_from_history(history)))
            early_stop_counter = int(
                checkpoint.get("early_stop_counter", infer_early_stop_counter_from_history(history))
            )

            return {
                "start_epoch": start_epoch,
                "best_auc": best_auc,
                "early_stop_counter": early_stop_counter,
                "history": history,
                "resumed": True,
                "resume_source": path,
            }

        # Case 2: old raw state_dict only
        if isinstance(checkpoint, dict):
            try:
                model.load_state_dict(checkpoint, strict=True)
                history = load_history_from_csv(history_csv_path)
                start_epoch = 0
                if len(history) > 0 and "epoch" in history[-1]:
                    start_epoch = int(history[-1]["epoch"])

                best_auc = infer_best_auc_from_history(history)
                early_stop_counter = infer_early_stop_counter_from_history(history)

                log_message(
                    "Loaded raw state_dict checkpoint. "
                    "Model weights restored, but optimizer/scheduler/scaler exact state was not available.",
                    log_path,
                )

                return {
                    "start_epoch": start_epoch,
                    "best_auc": best_auc,
                    "early_stop_counter": early_stop_counter,
                    "history": history,
                    "resumed": True,
                    "resume_source": path,
                }
            except Exception:
                pass

    return default_state


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

    # File checkpoint đầy đủ để resume đúng nghĩa
    best_checkpoint_path = os.path.join(config.OUTPUT_DIR, "best_checkpoint.pth")
    last_checkpoint_path = os.path.join(config.OUTPUT_DIR, "last_checkpoint.pth")

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
        kan_hidden_dim=config.KAN_HIDDEN_DIM,
        kan_grid_size=config.KAN_GRID_SIZE,
        kan_grid_min=config.KAN_GRID_MIN,
        kan_grid_max=config.KAN_GRID_MAX,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    resume_state = load_resume_state(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        use_cuda=use_cuda,
        history_csv_path=history_csv_path,
        last_model_path=last_model_path,
        last_checkpoint_path=last_checkpoint_path,
        log_path=log_path,
    )

    start_epoch = resume_state["start_epoch"]
    best_auc = resume_state["best_auc"]
    early_stop_counter = resume_state["early_stop_counter"]
    history = resume_state["history"]

    if resume_state["resumed"]:
        log_message(
            f"Resumed training from: {resume_state['resume_source']}",
            log_path,
        )
        log_message(
            f"Resume state -> start_epoch={start_epoch + 1}, "
            f"best_auc={best_auc:.4f}, "
            f"history_len={len(history)}, "
            f"early_stop_counter={early_stop_counter}",
            log_path,
        )

        # nếu CSV bị rỗng/hỏng nhưng checkpoint còn history thì viết lại CSV
        if len(history) > 0 and (
            (not os.path.exists(history_csv_path)) or os.path.getsize(history_csv_path) == 0
        ):
            try:
                save_history_atomic(history, history_csv_path)
                log_message("Recovered history.csv from checkpoint history.", log_path)
            except OSError as e:
                log_message(f"[WARN] Could not recover history.csv: {e}", log_path)

    if start_epoch >= config.EPOCHS:
        log_message(
            f"Checkpoint already reached epoch {start_epoch}. "
            f"config.EPOCHS={config.EPOCHS}, nothing to do.",
            log_path,
        )
        return

    interrupted_epoch = None

    try:
        for epoch in range(start_epoch, config.EPOCHS):
            interrupted_epoch = epoch
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

            improved = (not np.isnan(val_mean_auc)) and (val_mean_auc > best_auc)

            if improved:
                best_auc = val_mean_auc
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Luôn lưu checkpoint đầy đủ trước để nếu history.csv lỗi vẫn resume được
            last_checkpoint = build_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                completed_epoch=epoch + 1,
                best_auc=best_auc,
                early_stop_counter=early_stop_counter,
                history=history,
                use_cuda=use_cuda,
            )
            save_checkpoint_atomic(last_checkpoint, last_checkpoint_path)

            # Giữ file weights-only cũ để tương thích với code đánh giá cũ
            if config.SAVE_LAST:
                save_state_dict_atomic(model.state_dict(), last_model_path)

            # Save best
            if improved:
                best_checkpoint = build_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    completed_epoch=epoch + 1,
                    best_auc=best_auc,
                    early_stop_counter=early_stop_counter,
                    history=history,
                    use_cuda=use_cuda,
                )
                save_checkpoint_atomic(best_checkpoint, best_checkpoint_path)
                save_state_dict_atomic(model.state_dict(), best_model_path)

                try:
                    save_val_predictions_csv(
                        image_ids=image_ids,
                        y_true=y_true,
                        y_prob=y_prob,
                        label_names=config.LABEL_COLS,
                        save_path=val_pred_path,
                    )
                except OSError as e:
                    log_message(f"[WARN] Could not save val predictions CSV: {e}", log_path)

                log_message(f"Saved best model to: {best_model_path}", log_path)
                log_message(f"Saved best checkpoint to: {best_checkpoint_path}", log_path)
                log_message(f"Saved best val predictions to: {val_pred_path}", log_path)
                log_message(f"Best val_mean_auc so far: {best_auc:.4f}", log_path)
            else:
                log_message(
                    f"No improvement. Early stopping counter: "
                    f"{early_stop_counter}/{config.EARLY_STOPPING_PATIENCE}",
                    log_path,
                )

            # Ghi history.csv kiểu an toàn; nếu lỗi thì vẫn không làm mất progress đã checkpoint
            try:
                save_history_atomic(history, history_csv_path)
            except OSError as e:
                log_message(
                    f"[WARN] Could not save history CSV to {history_csv_path}: {e}. "
                    f"Training progress is still recoverable from {last_checkpoint_path}.",
                    log_path,
                )

            if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                log_message("Early stopping triggered.", log_path)
                break

    except KeyboardInterrupt:
        log_message("\nTraining interrupted by user.", log_path)

        # Nếu bị ngắt giữa epoch, resume sẽ quay lại từ đầu epoch đang dang dở
        completed_epoch = interrupted_epoch if interrupted_epoch is not None else start_epoch

        interrupted_checkpoint = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=completed_epoch,
            best_auc=best_auc,
            early_stop_counter=early_stop_counter,
            history=history,
            use_cuda=use_cuda,
        )
        try:
            save_checkpoint_atomic(interrupted_checkpoint, last_checkpoint_path)
            if config.SAVE_LAST:
                save_state_dict_atomic(model.state_dict(), last_model_path)
            log_message(
                f"Saved interrupt checkpoint to: {last_checkpoint_path}. "
                f"Next run will resume from epoch {completed_epoch + 1}.",
                log_path,
            )
        except OSError as e:
            log_message(f"[WARN] Could not save interrupt checkpoint: {e}", log_path)

        raise

    log_message("\nTraining finished.", log_path)
    log_message(f"Best val_mean_auc: {best_auc:.4f}", log_path)
    log_message(f"History saved to: {history_csv_path}", log_path)
    log_message(f"Last model saved to: {last_model_path}", log_path)
    log_message(f"Last checkpoint saved to: {last_checkpoint_path}", log_path)


if __name__ == "__main__":
    main()
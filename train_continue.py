import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config as BaseConfig
from dataset import RSNADataset, build_train_val_dataframes, set_seed
from model import RSNAClassifier
from train import (
    ensure_dir,
    log_message,
    save_config_snapshot,
    save_val_predictions_csv,
    build_scheduler,
    get_current_lr,
    train_one_epoch,
    validate,
)


def copy_existing_config(base_cls):
    class ContinueConfig:
        pass

    for key in dir(base_cls):
        if key.isupper():
            setattr(ContinueConfig, key, getattr(base_cls, key))
    return ContinueConfig


def backup_file(path: str):
    if not os.path.exists(path):
        return
    backup_path = path + ".before_continue_backup"
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)


def load_history(history_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(history_csv_path):
        raise FileNotFoundError(f"Khong tim thay history.csv: {history_csv_path}")
    df = pd.read_csv(history_csv_path)
    if "epoch" not in df.columns:
        raise ValueError("history.csv khong co cot 'epoch'")
    return df


def infer_best_and_early_stop_counter(history_df: pd.DataFrame):
    best_auc = -1.0
    early_stop_counter = 0

    if "val_mean_auc" not in history_df.columns:
        return best_auc, early_stop_counter

    for metric in history_df["val_mean_auc"].tolist():
        if pd.isna(metric):
            early_stop_counter += 1
            continue
        metric = float(metric)
        if metric > best_auc:
            best_auc = metric
            early_stop_counter = 0
        else:
            early_stop_counter += 1

    return best_auc, early_stop_counter


def replay_scheduler_from_history(optimizer, scheduler, history_df: pd.DataFrame):
    if scheduler is None:
        return
    if "val_mean_auc" not in history_df.columns:
        return

    for metric in history_df["val_mean_auc"].tolist():
        if pd.isna(metric):
            continue
        scheduler.step(float(metric))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/storage/student5/handt/outputforclassification/linear/resnet18_linear_scratch_ep100_20260411_001232",
        help="Folder run cu can train tiep",
    )
    parser.add_argument(
        "--target-total-epochs",
        type=int,
        default=100,
        help="Tong so epoch mong muon sau khi train tiep",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="last_model.pth",
        choices=["last_model.pth", "best_model.pth"],
        help="Mac dinh dung last_model.pth de bam sat run dang do",
    )
    parser.add_argument(
        "--lr-override",
        type=float,
        default=None,
        help="Neu muon ep LR moi, vi du 1e-4",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    history_csv_path = os.path.join(run_dir, BaseConfig.HISTORY_CSV)
    log_path = os.path.join(run_dir, BaseConfig.TRAIN_LOG_NAME)
    best_model_path = os.path.join(run_dir, BaseConfig.BEST_MODEL_NAME)
    last_model_path = os.path.join(run_dir, BaseConfig.LAST_MODEL_NAME)
    val_pred_path = os.path.join(run_dir, BaseConfig.VAL_PRED_NAME)
    config_snapshot_path = os.path.join(run_dir, "config_snapshot_continue.json")
    checkpoint_path = os.path.join(run_dir, args.checkpoint_name)

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Khong tim thay run_dir: {run_dir}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {checkpoint_path}")

    history_df = load_history(history_csv_path)
    last_completed_epoch = int(history_df["epoch"].iloc[-1])

    if last_completed_epoch >= args.target_total_epochs:
        print(
            f"Run nay da co epoch cuoi = {last_completed_epoch}, "
            f"khong can train tiep den {args.target_total_epochs}."
        )
        return

    # backup 1 lan truoc khi ghi tiep
    backup_file(history_csv_path)
    backup_file(log_path)
    backup_file(best_model_path)
    backup_file(last_model_path)
    backup_file(val_pred_path)

    ContinueConfig = copy_existing_config(BaseConfig)

    # giu logic cu, nhung ep ve linear va ghi vao chinh run cu
    ContinueConfig.HEAD_TYPE = "linear"
    ContinueConfig.PRETRAINED = False
    ContinueConfig.OUTPUT_DIR = run_dir
    ContinueConfig.RUN_NAME = os.path.basename(run_dir)
    ContinueConfig.EPOCHS = int(args.target_total_epochs)

    if args.lr_override is not None:
        ContinueConfig.LR = float(args.lr_override)

    ensure_dir(ContinueConfig.OUTPUT_DIR)
    save_config_snapshot(ContinueConfig, config_snapshot_path)

    set_seed(ContinueConfig.SEED)

    use_cuda = torch.cuda.is_available() and ContinueConfig.DEVICE == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = bool(ContinueConfig.USE_AMP and use_cuda)

    log_message("=" * 80, log_path)
    log_message("CONTINUE TRAINING LINEAR HEAD", log_path)
    log_message(f"Using device: {device}", log_path)
    log_message(f"AMP enabled: {use_amp}", log_path)
    log_message(f"Run dir: {run_dir}", log_path)
    log_message(f"Checkpoint: {checkpoint_path}", log_path)
    log_message(f"Last completed epoch in history: {last_completed_epoch}", log_path)
    log_message(f"Target total epochs: {ContinueConfig.EPOCHS}", log_path)

    train_df, val_df = build_train_val_dataframes(ContinueConfig)
    log_message(f"Train samples: {len(train_df)}", log_path)
    log_message(f"Val samples: {len(val_df)}", log_path)

    train_dataset = RSNADataset(train_df, ContinueConfig)
    val_dataset = RSNADataset(val_df, ContinueConfig)

    persistent_workers = (
        ContinueConfig.PERSISTENT_WORKERS if ContinueConfig.NUM_WORKERS > 0 else False
    )
    pin_memory = ContinueConfig.PIN_MEMORY if use_cuda else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=ContinueConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=ContinueConfig.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ContinueConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=ContinueConfig.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = RSNAClassifier(
        num_classes=ContinueConfig.NUM_CLASSES,
        pretrained=ContinueConfig.PRETRAINED,
        head_type=ContinueConfig.HEAD_TYPE,
        mlp_hidden_dim=ContinueConfig.MLP_HIDDEN_DIM,
        dropout=ContinueConfig.DROPOUT,
        kan_num_basis=ContinueConfig.KAN_NUM_BASIS,
        kan_hidden_dim=ContinueConfig.KAN_HIDDEN_DIM,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    log_message(f"Loaded model weights from: {checkpoint_path}", log_path)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ContinueConfig.LR,
        weight_decay=ContinueConfig.WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer, ContinueConfig)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Khôi phục trạng thái logic từ history cũ
    best_auc, early_stop_counter = infer_best_and_early_stop_counter(history_df)
    log_message(f"Recovered best val_mean_auc from history: {best_auc:.6f}", log_path)
    log_message(
        f"Recovered early stopping counter from history: "
        f"{early_stop_counter}/{ContinueConfig.EARLY_STOPPING_PATIENCE}",
        log_path,
    )

    # Replay scheduler để LR hiện tại khớp với lịch sử cũ nhiều nhất có thể
    if args.lr_override is None:
        replay_scheduler_from_history(optimizer, scheduler, history_df)
        log_message(
            f"Recovered optimizer LR for next epoch: {get_current_lr(optimizer):.6g}",
            log_path,
        )
    else:
        log_message(
            f"Using manual LR override for next epoch: {get_current_lr(optimizer):.6g}",
            log_path,
        )

    existing_history = history_df.copy()
    new_records = []

    start_epoch_idx = last_completed_epoch  # vi epoch trong code goc la 0-based, log la +1
    for epoch in range(start_epoch_idx, ContinueConfig.EPOCHS):
        current_lr = get_current_lr(optimizer)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=ContinueConfig.EPOCHS,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=ContinueConfig.ACCUMULATION_STEPS,
            clip_grad_norm=ContinueConfig.CLIP_GRAD_NORM,
        )

        val_loss, val_mean_auc, val_auc_dict, y_true, y_prob, image_ids = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            label_names=ContinueConfig.LABEL_COLS,
            epoch=epoch,
            total_epochs=ContinueConfig.EPOCHS,
            use_amp=use_amp,
        )

        if scheduler is not None:
            if not np.isnan(val_mean_auc):
                scheduler.step(val_mean_auc)

        log_message(
            (
                f"\nEpoch [{epoch+1}/{ContinueConfig.EPOCHS}] "
                f"lr={current_lr:.6g} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_mean_auc={val_mean_auc:.4f}"
            ),
            log_path,
        )

        for label_name in ContinueConfig.LABEL_COLS:
            label_auc = val_auc_dict[label_name]
            log_message(f" {label_name}: {label_auc:.4f}", log_path)

        epoch_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auc": val_mean_auc,
        }
        for label_name in ContinueConfig.LABEL_COLS:
            epoch_record[f"auc_{label_name}"] = val_auc_dict[label_name]

        new_records.append(epoch_record)

        combined_history = pd.concat(
            [existing_history, pd.DataFrame(new_records)],
            ignore_index=True,
        )
        combined_history.to_csv(history_csv_path, index=False)

        improved = (not np.isnan(val_mean_auc)) and (val_mean_auc > best_auc)

        if improved:
            best_auc = float(val_mean_auc)
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            save_val_predictions_csv(
                image_ids=image_ids,
                y_true=y_true,
                y_prob=y_prob,
                label_names=ContinueConfig.LABEL_COLS,
                save_path=val_pred_path,
            )
            log_message(f"Saved best model to: {best_model_path}", log_path)
            log_message(f"Saved best val predictions to: {val_pred_path}", log_path)
            log_message(f"Best val_mean_auc so far: {best_auc:.4f}", log_path)
        else:
            early_stop_counter += 1
            log_message(
                f"No improvement. Early stopping counter: "
                f"{early_stop_counter}/{ContinueConfig.EARLY_STOPPING_PATIENCE}",
                log_path,
            )

        if ContinueConfig.SAVE_LAST:
            torch.save(model.state_dict(), last_model_path)

        if early_stop_counter >= ContinueConfig.EARLY_STOPPING_PATIENCE:
            log_message("Early stopping triggered during continue training.", log_path)
            break

    log_message("\nContinue training finished.", log_path)
    log_message(f"Best val_mean_auc: {best_auc:.4f}", log_path)
    log_message(f"History saved to: {history_csv_path}", log_path)
    log_message(f"Last model saved to: {last_model_path}", log_path)


if __name__ == "__main__":
    main()
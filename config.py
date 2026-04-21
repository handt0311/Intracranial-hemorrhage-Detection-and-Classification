import os
from datetime import datetime


class Config:
    # =========================
    # Paths for Dataset
    # =========================
    DATA_ROOT = "/storage/student5/handt/rsna-intracranial-hemorrhage-detection"
    TRAIN_DIR = os.path.join(DATA_ROOT, "stage_2_train")
    CSV_PATH = os.path.join(DATA_ROOT, "stage_2_train.csv")

    # =========================
    # Output
    # =========================
    OUTPUT_ROOT = "/storage/student5/handt/outputforclassification"

    # =========================
    # Data
    # =========================
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 1

    # CT window
    WINDOW_CENTER = 40
    WINDOW_WIDTH = 80

    # Multi-label setup
    LABEL_COLS = [
        "any",
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
    ]
    NUM_CLASSES = 6

    # =========================
    # Model
    # =========================
    BACKBONE = "resnet18"
    PRETRAINED = False

    # Model head
    # Options: "linear", "mlp", "kan"
    HEAD_TYPE = "linear"

    # For MLP
    MLP_HIDDEN_DIM = 256
    DROPOUT = 0.2

    # For KAN-style head
    KAN_NUM_BASIS = 8
    KAN_HIDDEN_DIM = 256

    # =========================
    # Train
    # =========================
    EPOCHS = 100
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42
    VAL_RATIO = 0.1
    DEVICE = "cuda"

    # Mixed precision
    USE_AMP = True

    # Gradient clipping
    CLIP_GRAD_NORM = 1.0

    # =========================
    # LR Scheduler
    # =========================
    SCHEDULER_NAME = "ReduceLROnPlateau"
    SCHEDULER_MODE = "max"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_THRESHOLD = 1e-4
    MIN_LR = 1e-6

    # =========================
    # Early stopping
    # =========================
    MONITOR_METRIC = "val_auc"
    MONITOR_MODE = "max"
    EARLY_STOPPING_PATIENCE = 12

    # =========================
    # Debug
    # =========================
    DEBUG = False
    DEBUG_SAMPLES = None

    # =========================
    # Run name / folder
    # =========================
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"{BACKBONE}_{HEAD_TYPE}_scratch_ep{EPOCHS}_{TIMESTAMP}"

    # folder theo head rồi tới từng lần train
    HEAD_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, HEAD_TYPE)
    OUTPUT_DIR = os.path.join(HEAD_OUTPUT_DIR, RUN_NAME)

    # =========================
    # Saving / Logging
    # =========================
    SAVE_BEST_ONLY = True
    SAVE_LAST = True

    BEST_MODEL_NAME = "best_model.pth"
    LAST_MODEL_NAME = "last_model.pth"
    HISTORY_CSV = "history.csv"
    VAL_PRED_NAME = "val_predictions.csv"
    TRAIN_LOG_NAME = "train_log.txt"
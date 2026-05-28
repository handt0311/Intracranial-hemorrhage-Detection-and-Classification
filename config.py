import os


class Config:
    # =========================
    # Paths for dataset
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
    HEAD_TYPE = "mlp"

    # MLP head settings
    MLP_HIDDEN_DIM = 256
    DROPOUT = 0.2

    # True KAN head settings
    KAN_HIDDEN_DIM = 64
    KAN_GRID_SIZE = 16
    KAN_GRID_MIN = -2.0
    KAN_GRID_MAX = 2.0

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
    # LR scheduler
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
    # Resume MLP run cu
    # =========================
    RUN_NAME = "resnet18_mlp_scratch_ep100_20260429_031344"
    HEAD_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "mlp")
    OUTPUT_DIR = os.path.join(HEAD_OUTPUT_DIR, RUN_NAME)

    AUTO_RESUME = True
    RESUME_PATH = os.path.join(OUTPUT_DIR, "last_checkpoint.pth")

    # =========================
    # Saving / logging
    # =========================
    SAVE_BEST_ONLY = True
    SAVE_LAST = True

    BEST_MODEL_NAME = "best_model.pth"
    LAST_MODEL_NAME = "last_model.pth"
    HISTORY_CSV = "history.csv"
    VAL_PRED_NAME = "val_predictions.csv"
    TRAIN_LOG_NAME = "train_log.txt"

import os

class Config:
    # =========================
    # Paths for Kaggle
    # =========================
    DATA_ROOT = "/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
    TRAIN_DIR = os.path.join(DATA_ROOT, "stage_2_train")
    CSV_PATH = os.path.join(DATA_ROOT, "stage_2_train.csv")

    # =========================
    # Output
    # =========================
    OUTPUT_DIR = "/kaggle/working/output"

    # =========================
    # Data
    # =========================
    IMAGE_SIZE = 224
    NUM_WORKERS = 2
    BATCH_SIZE = 8

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
    # Model head
    # Options: "linear", "mlp", "kan"
    # =========================
    HEAD_TYPE = "kan"

    # For MLP
    MLP_HIDDEN_DIM = 256
    DROPOUT = 0.2

    # For KAN-style head
    KAN_NUM_BASIS = 8
    KAN_HIDDEN_DIM = 256

    # =========================
    # Train
    # =========================
    EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    VAL_RATIO = 0.1

    # Use subset first
    DEBUG = True
    DEBUG_SAMPLES = 10000

    DEVICE = "cuda"

    RUN_NAME = f"resnet18_{HEAD_TYPE}_multilabel"
import os

class Config:
    # =========================
    # Paths for Kaggle
    # =========================
    DATA_ROOT = "/kaggle/input/rsna-intracranial-hemorrhage-detection"
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
    BATCH_SIZE = 16

    # CT window
    WINDOW_CENTER = 40
    WINDOW_WIDTH = 80

    # Classification setup
    # Binary first: any hemorrhage vs no hemorrhage
    POSITIVE_CLASS = "any"

    # =========================
    # Train
    # =========================
    EPOCHS = 3
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    VAL_RATIO = 0.1

    # Use only a subset first for debugging
    DEBUG = True
    DEBUG_SAMPLES = 5000

    DEVICE = "cuda"
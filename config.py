
import os

class Config:
    DATA_ROOT = "/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
    TRAIN_DIR = os.path.join(DATA_ROOT, "stage_2_train")
    CSV_PATH = os.path.join(DATA_ROOT, "stage_2_train.csv")

    OUTPUT_DIR = "/kaggle/working/output"

    IMAGE_SIZE = 224
    NUM_WORKERS = 2
    BATCH_SIZE = 8

    WINDOW_CENTER = 40
    WINDOW_WIDTH = 80

    LABEL_COLS = [
        "any",
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
    ]
    NUM_CLASSES = 6

    EPOCHS = 3
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    VAL_RATIO = 0.1

    DEBUG = True
    DEBUG_SAMPLES = 10000

    DEVICE = "cuda"
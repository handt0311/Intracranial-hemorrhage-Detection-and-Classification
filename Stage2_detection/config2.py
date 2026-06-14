from pathlib import Path
import torch


# =========================
# Root paths
# =========================
STORAGE_ROOT = Path("/storage/student5/handt")

# CQ500 DICOM root
CQ500_ROOT = STORAGE_ROOT / "cq500"

# BHX parent folder
BHX_ROOT = STORAGE_ROOT / "bhx"


# =========================
# BHX raw annotation
# =========================
# This path is only needed if you rerun BHX preprocessing scripts.
# Your BHX folder was previously extracted with one nested folder level.
BHX_RAW_ROOT = (
    BHX_ROOT
    / "brain-hemorrhage-extended-bhx-bounding-box-extrapolation-from-thick-to-thin-slice-ct-images-1.1"
)

BHX_CSV = BHX_RAW_ROOT / "3_Extrapolation_to_Selected_Series.csv"


# =========================
# Processed BHX files
# =========================
CQ500_INDEX_CSV = BHX_ROOT / "cq500_sop_index.csv"
BHX_BOX_CSV = BHX_ROOT / "bhx_selected_boxes_5class.csv"

SPLIT_CSV = BHX_ROOT / "bhx_train_val" / "bhx_boxes_5class_with_split.csv"
TRAIN_CSV = BHX_ROOT / "bhx_train_val" / "bhx_train_boxes_5class.csv"
VAL_CSV = BHX_ROOT / "bhx_train_val" / "bhx_val_boxes_5class.csv"


# =========================
# Output
# =========================
OUTPUT_ROOT = STORAGE_ROOT / "output_bhx_detection"

# Official Stage 2 run:
# ResNet18 backbone transferred from KAN-based RSNA classifier
# + Faster R-CNN detection head
RUN_NAME = "resnet18_kan_to_fasterrcnn_bhx_full_ep100_map_coco"

RUN_DIR = OUTPUT_ROOT / RUN_NAME

BEST_DETECTOR_NAME = "best_detector.pth"               # best by mAP@0.5
BEST_COCO_DETECTOR_NAME = "best_detector_coco.pth"     # best by COCO mAP@0.5:0.95
LAST_CHECKPOINT_NAME = "last_checkpoint.pth"
HISTORY_CSV_NAME = "history.csv"


# =========================
# Stage 1 RSNA classification checkpoint
# =========================
RSNA_BEST_MODEL_PATH = (
    "/storage/student5/handt/outputforclassification/"
    "kan_official/resnet18_kan_official_scratch_ep100/best_model.pth"
)


# =========================
# Classes
# =========================
CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

CLASS_TO_ID = {
    "epidural": 1,
    "intraparenchymal": 2,
    "intraventricular": 3,
    "subarachnoid": 4,
    "subdural": 5,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

NUM_CLASSES = len(CLASS_NAMES)


# =========================
# Image preprocessing
# =========================
IMAGE_SIZE = 512

WINDOW_CENTER = 40
WINDOW_WIDTH = 80


# =========================
# Training
# =========================
SEED = 42

BATCH_SIZE = 1
NUM_WORKERS = 2

EPOCHS = 100

LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUTO_RESUME = True
SCHEDULER_NAME = "cosine"

# Full dataset training
MAX_TRAIN_IMAGES = None
MAX_VAL_IMAGES = None


# =========================
# Validation mAP settings
# =========================
# Evaluate validation every N epochs.
EVAL_EVERY = 5

# For AP computation, keep predictions above this score.
# AP ranks predictions by confidence, so this should be low.
MAP_SCORE_MIN = 0.001

# Main checkpoint:
# best_detector.pth is selected by mAP@0.5.
# best_detector_coco.pth is selected by COCO mAP@0.5:0.95.
BEST_MAP_KEY = "map50"


# =========================
# Early stopping
# =========================
EARLY_STOP = True

# Patience is counted by validation evaluations, not epochs.
# With EVAL_EVERY = 5 and EARLY_STOP_PATIENCE = 6,
# training stops after 30 epochs without mAP@0.5 improvement.
EARLY_STOP_PATIENCE = 6

# Minimum mAP@0.5 improvement required to reset patience.
EARLY_STOP_MIN_DELTA = 1e-4
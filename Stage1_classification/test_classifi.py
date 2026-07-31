import os
import random
import argparse

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch

from model import RSNAClassifier


# ============================================================
# 1. TEST CONFIG - SỬA THÔNG SỐ Ở ĐÂY
# ============================================================

DATA_ROOT = "/storage/student5/handt/rsna-intracranial-hemorrhage-detection"
TRAIN_DIR = os.path.join(DATA_ROOT, "stage_2_train")
CSV_PATH = os.path.join(DATA_ROOT, "stage_2_train.csv")

OUTPUT_ROOT = "/storage/student5/handt/outputforclassification"

# Sửa head muốn test ở đây: "linear", "mlp", hoặc "kan"
HEAD_TYPE = "linear"

# Sửa đúng tên folder run chứa best_model.pth
RUN_NAME = "resnet18_linear_scratch_ep100_20260411_001232"

RUN_DIR = os.path.join(OUTPUT_ROOT, HEAD_TYPE, RUN_NAME)

MODEL_PATH = os.path.join(RUN_DIR, "best_model.pth")

# File threshold của bạn nằm trong folder con này:
THRESHOLD_CSV = os.path.join(
    RUN_DIR,
    "evaluation_from_val_predictions",
    "best_threshold_by_f1.csv"
)

# Kết quả test sẽ được lưu ở đây
TEST_OUTPUT_DIR = os.path.join(RUN_DIR, "test_10_images")

LABEL_COLS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

NUM_CLASSES = len(LABEL_COLS)

IMAGE_SIZE = 224
WINDOW_CENTER = 40
WINDOW_WIDTH = 80

# Các thông số này phải giống lúc train
PRETRAINED = False
MLP_HIDDEN_DIM = 512
DROPOUT = 0.3

KAN_HIDDEN_DIM = 64
KAN_GRID_SIZE = 5
KAN_GRID_MIN = -1.0
KAN_GRID_MAX = 1.0

DEVICE = "cuda"
SEED = 42


# ============================================================
# 2. UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def dicom_to_hu(ds):
    image = ds.pixel_array.astype(np.int16)

    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    slope = float(getattr(ds, "RescaleSlope", 1.0))

    image = image.astype(np.float32)
    image = image * slope + intercept

    return image


def preprocess_ct_slice(image_hu, window_center=40, window_width=80, image_size=224):
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    image = np.clip(image_hu, lower, upper)
    image = (image - lower) / (upper - lower)
    image = image.astype(np.float32)

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    return image


def build_multilabel_df(csv_path, train_dir):
    df = pd.read_csv(csv_path)

    df["image_id"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
    df["subtype"] = df["ID"].apply(lambda x: x.split("_")[-1])

    df = df.pivot_table(
        index="image_id",
        columns="subtype",
        values="Label",
        aggfunc="first",
    ).reset_index()

    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[["image_id"] + LABEL_COLS]

    df["filepath"] = df["image_id"].apply(
        lambda x: os.path.join(train_dir, f"{x}.dcm")
    )

    return df


def clean_state_dict(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value

    return new_state_dict


def load_model(device):
    model = RSNAClassifier(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        head_type=HEAD_TYPE,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        dropout=DROPOUT,
        kan_hidden_dim=KAN_HIDDEN_DIM,
        kan_grid_size=KAN_GRID_SIZE,
        kan_grid_min=KAN_GRID_MIN,
        kan_grid_max=KAN_GRID_MAX,
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded model: {MODEL_PATH}")

    return model


def load_thresholds(default_threshold=0.5):
    if not os.path.exists(THRESHOLD_CSV):
        raise FileNotFoundError(f"Threshold CSV not found: {THRESHOLD_CSV}")

    df = pd.read_csv(THRESHOLD_CSV)

    if "label" not in df.columns or "threshold" not in df.columns:
        raise ValueError(
            f"Threshold CSV must have columns: label, threshold. "
            f"Current columns: {list(df.columns)}"
        )

    threshold_dict = dict(zip(df["label"], df["threshold"]))

    thresholds = {
        label: float(threshold_dict.get(label, default_threshold))
        for label in LABEL_COLS
    }

    print(f"Loaded thresholds: {THRESHOLD_CSV}")
    for label in LABEL_COLS:
        print(f"  {label}: {thresholds[label]}")

    return thresholds


def save_gray_image(path, image_01):
    image_uint8 = (image_01 * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, image_uint8)


def make_prediction_image(image_01, image_id, probs, thresholds, true_labels=None):
    image_uint8 = (image_01 * 255).clip(0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

    pred_labels = []

    for label, prob in zip(LABEL_COLS, probs):
        th = thresholds.get(label, 0.5)
        if prob >= th:
            pred_labels.append(label)

    if len(pred_labels) == 0:
        summary = "Prediction: negative"
    else:
        summary = "Prediction: " + ", ".join(pred_labels)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 170), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    y = 18

    cv2.putText(
        canvas,
        f"Image: {image_id}",
        (5, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 22

    cv2.putText(
        canvas,
        summary,
        (5, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 24

    for label, prob in zip(LABEL_COLS, probs):
        th = thresholds.get(label, 0.5)
        pred = int(prob >= th)

        if true_labels is not None:
            true_value = int(true_labels[label])
            text = f"{label}: prob={prob:.3f}, th={th:.2f}, pred={pred}, true={true_value}"
        else:
            text = f"{label}: prob={prob:.3f}, th={th:.2f}, pred={pred}"

        cv2.putText(
            canvas,
            text,
            (5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 18

    return canvas


@torch.no_grad()
def predict_one_image(model, dcm_path, device):
    ds = pydicom.dcmread(dcm_path)

    image_hu = dicom_to_hu(ds)

    image = preprocess_ct_slice(
        image_hu=image_hu,
        window_center=WINDOW_CENTER,
        window_width=WINDOW_WIDTH,
        image_size=IMAGE_SIZE,
    )

    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

    return image, probs


# ============================================================
# 3. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to test.",
    )

    parser.add_argument(
        "--only-positive-any",
        action="store_true",
        help="Only sample images with ground truth any=1.",
    )

    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold if label is missing in CSV.",
    )

    args = parser.parse_args()

    set_seed(SEED)

    if torch.cuda.is_available() and DEVICE == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"HEAD_TYPE: {HEAD_TYPE}")
    print(f"RUN_NAME: {RUN_NAME}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Threshold CSV: {THRESHOLD_CSV}")
    print(f"Output dir: {TEST_OUTPUT_DIR}")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    model = load_model(device)
    thresholds = load_thresholds(default_threshold=args.default_threshold)

    print("Building dataframe...")
    df = build_multilabel_df(CSV_PATH, TRAIN_DIR)

    df = df[df["filepath"].map(os.path.exists)].reset_index(drop=True)

    if args.only_positive_any:
        df = df[df["any"] == 1].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid DICOM images found.")

    num_images = min(args.num_images, len(df))

    sample_df = df.sample(
        n=num_images,
        random_state=SEED,
    ).reset_index(drop=True)

    records = []

    for idx, row in sample_df.iterrows():
        image_id = row["image_id"]
        dcm_path = row["filepath"]

        print(f"[{idx + 1}/{num_images}] Testing {image_id}")

        try:
            image, probs = predict_one_image(
                model=model,
                dcm_path=dcm_path,
                device=device,
            )

            true_labels = {
                label: int(row[label])
                for label in LABEL_COLS
            }

            original_path = os.path.join(TEST_OUTPUT_DIR, f"{image_id}_original.png")
            prediction_path = os.path.join(TEST_OUTPUT_DIR, f"{image_id}_prediction.png")

            save_gray_image(original_path, image)

            pred_img = make_prediction_image(
                image_01=image,
                image_id=image_id,
                probs=probs,
                thresholds=thresholds,
                true_labels=true_labels,
            )

            cv2.imwrite(prediction_path, pred_img)

            record = {
                "image_id": image_id,
                "dicom_path": dcm_path,
                "original_png": original_path,
                "prediction_png": prediction_path,
            }

            for label, prob in zip(LABEL_COLS, probs):
                th = thresholds.get(label, args.default_threshold)
                pred = int(prob >= th)

                record[f"{label}_true"] = int(row[label])
                record[f"{label}_prob"] = float(prob)
                record[f"{label}_threshold"] = float(th)
                record[f"{label}_pred"] = pred

            records.append(record)

        except Exception as e:
            print(f"Error with {image_id}: {e}")

    result_csv = os.path.join(TEST_OUTPUT_DIR, "test_10_predictions.csv")
    pd.DataFrame(records).to_csv(result_csv, index=False)

    print("\nDone.")
    print(f"Saved images to: {TEST_OUTPUT_DIR}")
    print(f"Saved CSV to: {result_csv}")


if __name__ == "__main__":
    main()
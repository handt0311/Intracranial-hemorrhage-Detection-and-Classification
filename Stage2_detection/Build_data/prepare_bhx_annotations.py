from pathlib import Path
import ast

import pandas as pd


BHX_CSV = Path(r"/storage/student5/handt/bhx/"
    "brain-hemorrhage-extended-bhx-bounding-box-extrapolation-from-thick-to-thin-slice-ct-images-1.1/"
    "3_Extrapolation_to_Selected_Series.csv")
CQ500_INDEX = Path(r"/storage/student5/handt/bhx/cq500_sop_index.csv")

OUTPUT_BOXES = Path(r"/storage/student5/handt/bhx/bhx_selected_boxes_5class.csv")


LABEL_MAP = {
    "Epidural": "epidural",
    "Intraparenchymal": "intraparenchymal",
    "Intraventricular": "intraventricular",
    "Subarachnoid": "subarachnoid",
    "Subdural": "subdural",
    # "Chronic": tạm bỏ để đồng bộ với RSNA
}


CLASS_TO_ID = {
    "epidural": 1,
    "intraparenchymal": 2,
    "intraventricular": 3,
    "subarachnoid": 4,
    "subdural": 5,
}


def parse_box(data_str):
    """
    BHX data column thường có dạng:
    {'x': 320.95, 'y': 235.81, 'width': 30.68, 'height': 52.49}
    """
    d = ast.literal_eval(data_str)

    x = float(d["x"])
    y = float(d["y"])
    w = float(d["width"])
    h = float(d["height"])

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return x1, y1, x2, y2


def main():
    bhx = pd.read_csv(BHX_CSV)
    idx = pd.read_csv(CQ500_INDEX)

    print("Original BHX rows:", len(bhx))
    print("Original BHX labels:")
    print(bhx["labelName"].value_counts())

    # Chỉ giữ 5 class acute tương ứng với RSNA
    bhx = bhx[bhx["labelName"].isin(LABEL_MAP.keys())].copy()

    bhx["class_name"] = bhx["labelName"].map(LABEL_MAP)
    bhx["class_id"] = bhx["class_name"].map(CLASS_TO_ID)

    # Parse bbox
    boxes = bhx["data"].apply(parse_box)
    bhx["x1"] = boxes.apply(lambda b: b[0])
    bhx["y1"] = boxes.apply(lambda b: b[1])
    bhx["x2"] = boxes.apply(lambda b: b[2])
    bhx["y2"] = boxes.apply(lambda b: b[3])

    # Join với path DICOM
    merged = bhx.merge(
        idx[["SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "Rows", "Columns", "path"]],
        on="SOPInstanceUID",
        how="inner",
        suffixes=("_bhx", "_dicom")
    )

    print("\nAfter removing Chronic:", len(bhx))
    print("After joining with CQ500:", len(merged))
    print("Lost rows after join:", len(bhx) - len(merged))

    # Lấy Study/Series từ DICOM index nếu có
    if "StudyInstanceUID_dicom" in merged.columns:
        merged["StudyInstanceUID_final"] = merged["StudyInstanceUID_dicom"]
    else:
        merged["StudyInstanceUID_final"] = merged["StudyInstanceUID"]

    if "SeriesInstanceUID_dicom" in merged.columns:
        merged["SeriesInstanceUID_final"] = merged["SeriesInstanceUID_dicom"]
    else:
        merged["SeriesInstanceUID_final"] = merged["SeriesInstanceUID"]

    # Clip bbox về trong ảnh
    merged["x1"] = merged["x1"].clip(lower=0)
    merged["y1"] = merged["y1"].clip(lower=0)

    merged["x2"] = merged.apply(
        lambda r: min(r["x2"], r["Columns"]) if pd.notnull(r["Columns"]) else r["x2"],
        axis=1
    )
    merged["y2"] = merged.apply(
        lambda r: min(r["y2"], r["Rows"]) if pd.notnull(r["Rows"]) else r["y2"],
        axis=1
    )

    # Bỏ bbox lỗi
    before = len(merged)
    merged = merged[(merged["x2"] > merged["x1"]) & (merged["y2"] > merged["y1"])].copy()
    print("Invalid boxes removed:", before - len(merged))

    output = merged[
        [
            "SOPInstanceUID",
            "StudyInstanceUID_final",
            "SeriesInstanceUID_final",
            "path",
            "Rows",
            "Columns",
            "class_name",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    ].copy()

    output = output.rename(
        columns={
            "StudyInstanceUID_final": "StudyInstanceUID",
            "SeriesInstanceUID_final": "SeriesInstanceUID",
        }
    )

    output.to_csv(OUTPUT_BOXES, index=False)

    print("\nSaved:", OUTPUT_BOXES)
    print("Final box rows:", len(output))
    print("Unique images:", output["SOPInstanceUID"].nunique())
    print("Unique studies:", output["StudyInstanceUID"].nunique())

    print("\nFinal labels:")
    print(output["class_name"].value_counts())

    print("\nExample:")
    print(output.head())


if __name__ == "__main__":
    main()
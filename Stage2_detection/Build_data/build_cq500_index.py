from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm


# Root folder that contains qct01, qct02, ..., qctXX on the server
CQ500_ROOT = Path("/storage/student5/handt/cq500")

# Output CSV used to map SOPInstanceUID to the corresponding DICOM file path
OUTPUT_CSV = Path("/storage/student5/handt/bhx/cq500_sop_index.csv")


def main():
    if not CQ500_ROOT.exists():
        raise FileNotFoundError(f"CQ500_ROOT does not exist: {CQ500_ROOT}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Recursively collect all files under the CQ500 root folder
    files = [p for p in CQ500_ROOT.rglob("*") if p.is_file()]

    print(f"Total files found: {len(files)}")
    print(f"CQ500_ROOT: {CQ500_ROOT}")
    print(f"OUTPUT_CSV: {OUTPUT_CSV}")

    rows = []
    failed = 0

    for path in tqdm(files, desc="Indexing CQ500 DICOM files"):
        try:
            # Read only DICOM metadata to make indexing faster
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

            sop_uid = str(ds.get("SOPInstanceUID", ""))
            study_uid = str(ds.get("StudyInstanceUID", ""))
            series_uid = str(ds.get("SeriesInstanceUID", ""))
            modality = str(ds.get("Modality", ""))

            # Skip files without SOPInstanceUID because they cannot be matched with BHX
            if sop_uid == "":
                failed += 1
                continue

            rows.append(
                {
                    "SOPInstanceUID": sop_uid,
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Modality": modality,
                    "Rows": ds.get("Rows", None),
                    "Columns": ds.get("Columns", None),
                    "path": str(path),
                }
            )

        except Exception:
            # Non-DICOM or unreadable files are skipped
            failed += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV)
    print("Readable DICOM files:", len(df))
    print("Failed files:", failed)

    if len(df) > 0:
        print("Unique SOPInstanceUID:", df["SOPInstanceUID"].nunique())
        print("Duplicated SOPInstanceUID:", df["SOPInstanceUID"].duplicated().sum())
        print("\nExample rows:")
        print(df.head())


if __name__ == "__main__":
    main()
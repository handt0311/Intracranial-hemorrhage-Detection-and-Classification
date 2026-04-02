from pathlib import Path
from collections import Counter

import nibabel as nib
import pandas as pd


def inspect_nifti_dataset(ct_scans_dir, output_csv="ct_ich_scan_info.csv"):
    ct_scans_dir = Path(ct_scans_dir)

    if not ct_scans_dir.exists():
        print(f"[ERROR] Folder not found: {ct_scans_dir}")
        return

    nifti_files = sorted(
        [
            p for p in ct_scans_dir.iterdir()
            if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        ]
    )

    if not nifti_files:
        print(f"[ERROR] No .nii or .nii.gz files found in: {ct_scans_dir}")
        return

    rows = []
    shape_counter = Counter()
    xy_spacing_counter = Counter()
    xyz_spacing_counter = Counter()

    print(f"Found {len(nifti_files)} NIfTI files\n")

    for i, file_path in enumerate(nifti_files, 1):
        try:
            nii = nib.load(str(file_path))
            shape = nii.shape
            zooms = nii.header.get_zooms()

            # CT volume shape is usually (H, W, S)
            dim_x = shape[0] if len(shape) > 0 else None
            dim_y = shape[1] if len(shape) > 1 else None
            num_slices = shape[2] if len(shape) > 2 else None

            spacing_x = zooms[0] if len(zooms) > 0 else None
            spacing_y = zooms[1] if len(zooms) > 1 else None
            spacing_z = zooms[2] if len(zooms) > 2 else None

            rows.append({
                "file_name": file_path.name,
                "full_path": str(file_path.resolve()),
                "shape": str(shape),
                "resolution_xy": f"{dim_x}x{dim_y}" if dim_x is not None and dim_y is not None else None,
                "num_slices": num_slices,
                "spacing_x_mm": spacing_x,
                "spacing_y_mm": spacing_y,
                "spacing_z_mm": spacing_z,
                "xy_spacing_mm": f"{spacing_x}x{spacing_y}" if spacing_x is not None and spacing_y is not None else None,
            })

            shape_counter[shape] += 1

            if spacing_x is not None and spacing_y is not None:
                xy_spacing_counter[(round(spacing_x, 6), round(spacing_y, 6))] += 1

            if spacing_x is not None and spacing_y is not None and spacing_z is not None:
                xyz_spacing_counter[(round(spacing_x, 6), round(spacing_y, 6), round(spacing_z, 6))] += 1

            print(f"[{i}/{len(nifti_files)}] {file_path.name}")
            print(f"  shape = {shape}")
            print(f"  zooms = {zooms}\n")

        except Exception as e:
            print(f"[ERROR] Failed to read {file_path.name}: {e}\n")

    if not rows:
        print("[ERROR] No readable files found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Readable files: {len(df)}")
    print()

    print("Most common shapes:")
    for shape, count in shape_counter.most_common(10):
        print(f"  {shape}: {count}")

    print("\nMost common pixel spacings (x, y):")
    for spacing, count in xy_spacing_counter.most_common(10):
        print(f"  {spacing} mm: {count}")

    print("\nMost common voxel spacings (x, y, z):")
    for spacing, count in xyz_spacing_counter.most_common(10):
        print(f"  {spacing} mm: {count}")

    print("\nDetailed table saved to:")
    print(f"  {Path(output_csv).resolve()}")

    most_common_shape = shape_counter.most_common(1)[0][0] if shape_counter else None
    most_common_xy = xy_spacing_counter.most_common(1)[0][0] if xy_spacing_counter else None
    most_common_xyz = xyz_spacing_counter.most_common(1)[0][0] if xyz_spacing_counter else None

    print("\n" + "=" * 60)
    print("SUGGESTED DATASET TABLE ENTRIES")
    print("=" * 60)

    if most_common_shape:
        print(f"Resolution: most common = {most_common_shape[0]} x {most_common_shape[1]}")
    else:
        print("Resolution: could not be determined")

    if most_common_xy:
        print(f"Pixel spacing: most common = {most_common_xy[0]} x {most_common_xy[1]} mm")
    else:
        print("Pixel spacing: could not be determined")

    if most_common_xyz:
        print(f"Slice spacing / z spacing: most common = {most_common_xyz[2]} mm")
    else:
        print("Slice spacing / z spacing: could not be determined")


if __name__ == "__main__":
    ct_scans_folder = r"D:\Hieu\B3\Intership\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\ct_scans"

    inspect_nifti_dataset(
        ct_scans_dir=ct_scans_folder,
        output_csv="ct_ich_scan_info.csv"
    )
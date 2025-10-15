import os
import re
import argparse
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np


def extract_id(fname):
    match = re.search(r'\d{4}', fname)
    return match.group(0) if match else None


def find_matching_pairs(pred_dir, label_dir):
    preds = sorted(glob(os.path.join(pred_dir, "*.nii*")))
    labels = sorted(glob(os.path.join(label_dir, "*.nii*")))
    pred_map = {extract_id(p): p for p in preds if extract_id(p)}
    label_map = {extract_id(l): l for l in labels if extract_id(l)}
    common = sorted(set(pred_map.keys()) & set(label_map.keys()))
    return [(pred_map[i], label_map[i], i) for i in common]


def build_probabilistic_atlas(label_paths, label_value):
    print(f"Building probabilistic atlas for label {label_value}...")

    ref_img = sitk.ReadImage(label_paths[0])  # use first GT as reference

    atlas = sitk.Image(ref_img.GetSize(), sitk.sitkFloat32)
    atlas.CopyInformation(ref_img)

    for p in tqdm(label_paths):
        img = sitk.ReadImage(p)
        mask = sitk.Equal(img, label_value)
        mask = sitk.Cast(mask, sitk.sitkFloat32)
        mask = sitk.Resample(mask, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
        atlas = atlas + mask

    atlas /= len(label_paths)
    return atlas



def get_largest_n_components(mask, n=2):
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    sizes = [(l, stats.GetPhysicalSize(l)) for l in stats.GetLabels()]
    largest = sorted(sizes, key=lambda x: -x[1])[:n]
    output = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    output.CopyInformation(mask)
    for l, _ in largest:
        output = output | sitk.Cast(cc == l, sitk.sitkUInt8)
    return output


def process_label(pred_img, atlas, label_value, threshold=0.5):
    # Create binary mask from prediction
    pred_mask = sitk.Equal(pred_img, label_value)
    pred_mask = sitk.Cast(pred_mask, sitk.sitkUInt8)

    # Threshold atlas to make binary mask
    atlas_mask = sitk.Greater(atlas, threshold)
    atlas_mask = sitk.Cast(atlas_mask, sitk.sitkUInt8)

    # 🚨 Resample atlas_mask to pred_img space
    atlas_mask = sitk.Resample(atlas_mask, pred_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # Apply filtering
    filtered_mask = pred_mask & atlas_mask
    return sitk.Cast(filtered_mask, sitk.sitkUInt8) * label_value



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Prediction directory")
    parser.add_argument("--label_dir", required=True, help="Ground truth label directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--min_volume", type=float, default=1000, help="Minimum component volume in mm³")
    parser.add_argument("--max_volume", type=float, default=7000, help="Maximum component volume in mm³")
    parser.add_argument("--atlas_thresh", type=float, default=0.2, help="Threshold for atlas masking")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = find_matching_pairs(args.pred_dir, args.label_dir)
    if not pairs:
        print("No matching files found.")
        return

    voxel_volume = 1.0  # Optional: Read from metadata

    label_paths = [gt for _, gt, _ in pairs]
    atlas1 = build_probabilistic_atlas(label_paths, label_value=1)
    atlas2 = build_probabilistic_atlas(label_paths, label_value=2)

    print("Postprocessing predictions...")
    for pred_path, _, case_id in tqdm(pairs):
        pred_img = sitk.ReadImage(pred_path)
        output_img = sitk.Image(pred_img.GetSize(), sitk.sitkUInt8)
        output_img.CopyInformation(pred_img)

        proc1 = process_label(pred_img, atlas1, label_value=1)

        proc2 = process_label(pred_img, atlas2, label_value=2)

        output_img = sitk.Maximum(proc1, proc2)

        output_path = os.path.join(args.out_dir, os.path.basename(pred_path))
        sitk.WriteImage(output_img, output_path)


if __name__ == "__main__":
    main()


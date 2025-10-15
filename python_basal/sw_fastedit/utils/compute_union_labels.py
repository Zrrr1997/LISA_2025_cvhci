import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob

def pad_to_shape(array, target_shape):
    """Pads array to match the target shape with zeros."""
    pad_width = [(0, max(ts - s, 0)) for s, ts in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=0)

def compute_union_and_high_confidence(input_dir, union_output_path, highconf_output_path, class1=1, class2=2, threshold=0.9):
    nifti_files = sorted(glob(os.path.join(input_dir, "*.nii*")))
    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")

    print(f"Found {len(nifti_files)} NIfTI files.")

    all_shapes = []
    masks = []
    affines = []
    headers = []

    # Step 1: Load masks and record shapes
    for file in nifti_files:
        img = nib.load(file)
        data = img.get_fdata().astype(np.uint8)
        mask = ((data == class1) | (data == class2)).astype(np.uint8)

        all_shapes.append(mask.shape)
        masks.append(mask)
        affines.append(img.affine)
        headers.append(img.header)
        print(f"Loaded: {os.path.basename(file)} with shape {mask.shape}")

    # Step 2: Find maximum shape
    max_shape = np.max(np.array(all_shapes), axis=0)
    print(f"Max shape across all volumes: {tuple(max_shape)}")

    # Step 3: Pad all masks to max shape
    padded_masks = [pad_to_shape(mask, max_shape) for mask in masks]

    # Stack all masks: shape (N, H, W, D)
    stacked_masks = np.stack(padded_masks, axis=0)

    # Step 4: Compute union mask (any positive)
    union_mask = np.any(stacked_masks, axis=0).astype(np.uint8)

    # Step 5: Compute probability mask (mean over masks)
    prob_map = np.mean(stacked_masks, axis=0)

    # Step 6: Threshold probability map to get high confidence mask
    highconf_mask = (prob_map > threshold).astype(np.uint8)

    # Step 7: Save union mask
    reference_affine = affines[0] if affines else np.eye(4)
    reference_header = headers[0] if headers else None

    union_img = nib.Nifti1Image(union_mask, affine=reference_affine, header=reference_header)
    nib.save(union_img, union_output_path)
    print(f"Saved union mask to {union_output_path}")

    # Step 8: Save high confidence mask
    highconf_img = nib.Nifti1Image(highconf_mask, affine=reference_affine, header=reference_header)
    nib.save(highconf_img, highconf_output_path)
    print(f"Saved high confidence mask (>{threshold*100:.0f}%) to {highconf_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute union and high-confidence masks across NIfTI files with possibly different shapes.")
    parser.add_argument("--input_dir", required=True, help="Directory with NIfTI label files")
    parser.add_argument("--union_nifti", required=True, help="Output NIfTI file path for the union mask")
    parser.add_argument("--highconf_nifti", required=True, help="Output NIfTI file path for the high-confidence mask")
    parser.add_argument("--class1", type=int, default=1, help="Class label 1 to union")
    parser.add_argument("--class2", type=int, default=2, help="Class label 2 to union")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold for high confidence mask (0-1)")
    args = parser.parse_args()

    compute_union_and_high_confidence(args.input_dir, args.union_nifti, args.highconf_nifti,
                                      args.class1, args.class2, args.threshold)

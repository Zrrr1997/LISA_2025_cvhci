import os
import argparse
import nibabel as nib
import numpy as np
from glob import glob

def pad_to_match(array, target_shape):
    pad_width = []
    for s, t in zip(array.shape, target_shape):
        before = 0
        after = max(t - s, 0)
        pad_width.append((before, after))
    return np.pad(array, pad_width, mode='constant', constant_values=0)

def crop_to_shape(array, target_shape):
    slices = tuple(slice(0, s) for s in target_shape)
    return array[slices]

def main():
    parser = argparse.ArgumentParser(description="Filter prediction NIfTI files by a binary mask with shape handling.")
    parser.add_argument("--binary_mask", required=True, help="Path to binary NIfTI mask (1 = keep, 0 = discard)")
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction NIfTI files (labels 1 and 2)")
    parser.add_argument("--output_dir", required=True, help="Directory to save filtered prediction NIfTI files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    binary_img = nib.load(args.binary_mask)
    binary_data = binary_img.get_fdata() > 0
    binary_shape = binary_data.shape

    pred_files = sorted(glob(os.path.join(args.pred_dir, "*.nii*")))
    if not pred_files:
        print(f"No prediction NIfTI files found in {args.pred_dir}")
        return

    for pred_path in pred_files:
        pred_img = nib.load(pred_path)
        pred_data = pred_img.get_fdata().astype(np.uint8)
        pred_shape = pred_data.shape

        # Determine target shape (max of both)
        target_shape = tuple(max(ps, bs) for ps, bs in zip(pred_shape, binary_shape))

        # Pad both to target shape if needed
        padded_pred = pad_to_match(pred_data, target_shape)
        padded_binary = pad_to_match(binary_data, target_shape)

        # Apply mask
        filtered_data = np.where(padded_binary, padded_pred, 0)

        # Crop back to original prediction shape
        cropped_filtered = crop_to_shape(filtered_data, pred_shape)

        output_path = os.path.join(args.output_dir, os.path.basename(pred_path))
        nib.save(nib.Nifti1Image(cropped_filtered.astype(np.uint8), pred_img.affine, header=pred_img.header), output_path)

        print(f"Filtered and saved: {os.path.basename(output_path)}")

if __name__ == "__main__":
    main()

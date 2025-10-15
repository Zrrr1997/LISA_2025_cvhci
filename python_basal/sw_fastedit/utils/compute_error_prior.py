import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob

def pad_to_shape(array, target_shape):
    pad_width = [(0, max(ts - s, 0)) for s, ts in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=0)

def compute_error_priors(input_dirs, output_overseg, output_underseg, class1=1, class2=2, threshold=None):
    all_files = []
    for dir_path in input_dirs:
        dir_files = sorted(glob(os.path.join(dir_path, "*.nii*")))
        all_files.extend(dir_files)

    if not all_files:
        raise FileNotFoundError("No NIfTI files found in the provided directories.")

    print(f"Found {len(all_files)} NIfTI error files.")

    all_shapes = []
    overseg_masks = []
    underseg_masks = []
    affines = []
    headers = []

    for file in all_files:
        img = nib.load(file)
        data = img.get_fdata().astype(np.uint8)

        overseg = (data == class1).astype(np.uint8)
        underseg = (data == class2).astype(np.uint8)

        all_shapes.append(data.shape)
        overseg_masks.append(overseg)
        underseg_masks.append(underseg)

        affines.append(img.affine)
        headers.append(img.header)
        print(f"Loaded: {os.path.basename(file)} with shape {data.shape}")

    max_shape = np.max(np.array(all_shapes), axis=0)
    print(f"Max shape across all volumes: {tuple(max_shape)}")

    padded_oversegs = [pad_to_shape(mask, max_shape) for mask in overseg_masks]
    padded_undersegs = [pad_to_shape(mask, max_shape) for mask in underseg_masks]

    overseg_prior = np.mean(np.stack(padded_oversegs, axis=0), axis=0)
    underseg_prior = np.mean(np.stack(padded_undersegs, axis=0), axis=0)

    if threshold is not None:
        print(f"Applying threshold: {threshold}")
        overseg_prior = (overseg_prior >= threshold).astype(np.uint8)
        underseg_prior = (underseg_prior >= threshold).astype(np.uint8)

    reference_affine = affines[0] if affines else np.eye(4)
    reference_header = headers[0] if headers else None

    nib.save(nib.Nifti1Image(overseg_prior.astype(np.float32), reference_affine, header=reference_header), output_overseg)
    nib.save(nib.Nifti1Image(underseg_prior.astype(np.float32), reference_affine, header=reference_header), output_underseg)

    print(f"Saved oversegmentation prior to {output_overseg}")
    print(f"Saved undersegmentation prior to {output_underseg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute separate voxel-wise priors for oversegmentation and undersegmentation from multiple directories.")
    parser.add_argument("--input_dirs", nargs='+', required=True, help="List of directories with NIfTI error masks (labels 1 and 2)")
    parser.add_argument("--output_overseg", required=True, help="Output path for the oversegmentation prior NIfTI file")
    parser.add_argument("--output_underseg", required=True, help="Output path for the undersegmentation prior NIfTI file")
    parser.add_argument("--class1", type=int, default=1, help="Oversegmentation label (default=1)")
    parser.add_argument("--class2", type=int, default=2, help="Undersegmentation label (default=2)")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold for binarizing the prior maps (0.0 - 1.0)")
    args = parser.parse_args()

    compute_error_priors(args.input_dirs, args.output_overseg, args.output_underseg, args.class1, args.class2, args.threshold)

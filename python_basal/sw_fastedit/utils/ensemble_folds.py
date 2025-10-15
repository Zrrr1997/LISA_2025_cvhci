import argparse
import os
import numpy as np
import nibabel as nib
from collections import Counter
from glob import glob
from scipy.ndimage import label

def collect_all_filenames(input_dirs):
    filenames = set()
    for d in input_dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        filenames.update(os.path.basename(f) for f in files)
    return sorted(filenames)

def keep_largest_component_per_class(segmentation, num_classes=9):
    output = np.zeros_like(segmentation)
    for cls in range(1, num_classes):  # Skip background 0
        mask = (segmentation == cls)
        if np.any(mask):
            labeled, num_features = label(mask)
            if num_features > 0:
                sizes = np.bincount(labeled.flatten())
                sizes[0] = 0  # Background label
                largest_label = sizes.argmax()
                output[labeled == largest_label] = cls
    return output

def load_existing_nifti_files(directories, filename, harmonize):
    arrays = []
    ref_img = None
    for d in directories:
        filepath = os.path.join(d, filename)
        if os.path.isfile(filepath):
            img = nib.load(filepath)
            data = img.get_fdata().astype(np.int16)
            if harmonize:
                data = keep_largest_component_per_class(data)
            arrays.append(data)
            ref_img = img  # Use the last found image for reference
    if not arrays:
        raise FileNotFoundError(f"File {filename} not found in any input directory.")
    return arrays, ref_img

def majority_vote(arrays):
    stacked = np.stack(arrays, axis=-1)
    vote_result = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], -1, stacked)
    return vote_result

def intersection_vote(arrays):
    intersect = arrays[0].copy()
    for arr in arrays[1:]:
        intersect = np.where(intersect == arr, intersect, 0)
    return intersect

def union_vote(arrays):
    stacked = np.stack(arrays, axis=-1)
    return np.max(stacked, axis=-1)

def save_nifti(array, ref_img, output_path):
    new_img = nib.Nifti1Image(array.astype(np.int16), ref_img.affine, ref_img.header)
    nib.save(new_img, output_path)

def process_file(filename, input_dirs, output_dir, mode, harmonize):
    arrays, ref_img = load_existing_nifti_files(input_dirs, filename, harmonize)

    if mode == 'majority':
        ensembled = majority_vote(arrays)
    elif mode == 'intersection':
        ensembled = intersection_vote(arrays)
    elif mode == 'union':
        ensembled = union_vote(arrays)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    output_path = os.path.join(output_dir, f"{filename}")
    save_nifti(ensembled, ref_img, output_path)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Ensemble NIfTI predictions with optional largest component filtering.")
    parser.add_argument('--input_dirs', nargs='+', required=True, help='List of directories with prediction NIfTI files.')
    parser.add_argument('--output_dir', required=True, help='Directory to save ensemble outputs.')
    parser.add_argument('--mode', choices=['majority', 'intersection', 'union'], required=True, help='Ensembling mode.')
    parser.add_argument('--no_harmonize', dest='harmonize', action='store_false', help='Disable largest component filtering (default is enabled).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_filenames = collect_all_filenames(args.input_dirs)
    if not all_filenames:
        raise ValueError("No .nii.gz files found in the provided directories.")

    print(f"Processing {len(all_filenames)} files with mode '{args.mode}' and harmonize={args.harmonize}")

    # Sequential processing (removed ProcessPoolExecutor)
    for filename in all_filenames:
        try:
            result = process_file(filename, args.input_dirs, args.output_dir, args.mode, args.harmonize)
            print(f"Saved {result}")
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")

if __name__ == '__main__':
    main()

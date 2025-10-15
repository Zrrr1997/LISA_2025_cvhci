import os
import argparse
import nibabel as nib
import numpy as np

def compute_label_volumes(nifti_path, labels=(1, 2)):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    volumes = {}
    for label in labels:
        volumes[label] = np.sum(data == label)
    return volumes

def main():
    parser = argparse.ArgumentParser(description="Compute min and max volumes for labels 1 and 2 in NIfTI files.")
    parser.add_argument('--input_dir', required=True, help='Directory containing NIfTI label files (.nii or .nii.gz)')
    parser.add_argument('--output_file', required=True, help='Output text file for min and max volumes')
    args = parser.parse_args()

    min_volumes = {1: float('inf'), 2: float('inf')}
    max_volumes = {1: 0, 2: 0}

    # Process files
    for fname in os.listdir(args.input_dir):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            path = os.path.join(args.input_dir, fname)
            volumes = compute_label_volumes(path)
            for label in (1, 2):
                vol = volumes[label]
                if vol < min_volumes[label]:
                    min_volumes[label] = vol
                if vol > max_volumes[label]:
                    max_volumes[label] = vol

    # Handle case no files or no labels found
    for label in (1, 2):
        if min_volumes[label] == float('inf'):
            min_volumes[label] = 0

    # Write results
    with open(args.output_file, 'w') as f:
        f.write("Label Min_Volume Max_Volume\n")
        for label in (1, 2):
            f.write(f"{label} {min_volumes[label]} {max_volumes[label]}\n")

    print(f"Results written to {args.output_file}")

if __name__ == "__main__":
    main()


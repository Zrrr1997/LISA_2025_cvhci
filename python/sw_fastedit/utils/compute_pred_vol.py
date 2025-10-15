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
    parser = argparse.ArgumentParser(description="Compute volumes of labels 1 and 2 in prediction files.")
    parser.add_argument('--input_dir', required=True, help='Directory with prediction NIfTI files')
    args = parser.parse_args()

    for fname in sorted(os.listdir(args.input_dir)):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            path = os.path.join(args.input_dir, fname)
            volumes = compute_label_volumes(path)
            print(f"{fname}: Label 1 volume = {volumes[1]}, Label 2 volume = {volumes[2]}")

if __name__ == "__main__":
    main()


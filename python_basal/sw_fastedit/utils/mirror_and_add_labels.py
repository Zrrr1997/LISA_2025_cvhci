import numpy as np
import nibabel as nib
import argparse
import os
import shutil

def mirror_and_add_labels(nifti_path, output_path, class1=1, class2=2):
    # Load NIfTI image
    img = nib.load(nifti_path)
    label_data = img.get_fdata().astype(np.uint8)

    # Extract binary masks for class 1 and class 2
    mask1 = (label_data == class1).astype(np.uint8)
    mask2 = (label_data == class2).astype(np.uint8)
    fraction = 0.5
    if (np.sum(mask1) / np.sum(mask2)) < fraction or (np.sum(mask1) / np.sum(mask2)) > (1 / fraction):

        # Flip along the first axis (x-axis in (x, y, z))
        mirrored_mask1 = np.flip(mask1, axis=0)
        mirrored_mask2 = np.flip(mask2, axis=0)

        # Add mirrored class 1 to class 2 and vice versa
        # Flip along first axis (x-axis in (x, y, z))
        mirrored_mask1 = np.flip(mask1, axis=0)
        mirrored_mask2 = np.flip(mask2, axis=0)

        # Add mirrored masks to the opposite class (preserving highest label)
        label = ((mask1 + mirrored_mask2) > 0) * class1
        label += ((mask2 + mirrored_mask1) > 0) * class2

        # Save to output NIfTI file
        new_img = nib.Nifti1Image(label, affine=img.affine, header=img.header)
        nib.save(new_img, output_path)
        print(f"Saved mirrored and modified label to {output_path}")
    else:
        shutil.copy(nifti_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirror class 1 and 2 in a NIfTI label and add to each other.")
    parser.add_argument("--input", help="Path to the input NIfTI file")
    parser.add_argument("--output", help="Path to save the modified NIfTI file")
    parser.add_argument("--class1", type=int, default=1, help="Label value for class 1 (default: 1)")
    parser.add_argument("--class2", type=int, default=2, help="Label value for class 2 (default: 2)")
    args = parser.parse_args()

    mirror_and_add_labels(args.input, args.output, args.class1, args.class2)

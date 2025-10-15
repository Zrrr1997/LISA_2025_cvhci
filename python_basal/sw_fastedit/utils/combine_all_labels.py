import argparse
import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

def combine_labels(hipp_dir, baga_dir, vent_dir, output_dir):
    """
    Combine multiple label sets into a single unified labeling scheme:
    0: Background
    1-2: Hippocampus (original labels 1-2)
    3-6: Basal ganglia (original labels 1-4)
    7-8: Ventricles (original labels 1-2)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all case IDs by looking at hippocampus files
    hipp_files = sorted(glob.glob(os.path.join(hipp_dir, 'LISA_*_HF_hipp.nii.gz')))
    case_ids = [os.path.basename(f).split('_')[1] for f in hipp_files]

    print(f"Found {len(case_ids)} cases to process")

    for case_id in tqdm(case_ids, desc="Processing cases"):
        # Initialize combined label as all zeros
        combined = None

        # Process hippocampus labels (1-2)
        hipp_path = os.path.join(hipp_dir, f'LISA_{case_id}_HF_hipp.nii.gz')
        if os.path.exists(hipp_path):
            hipp_img = nib.load(hipp_path)
            hipp_data = hipp_img.get_fdata()
            if combined is None:
                combined = np.zeros_like(hipp_data, dtype=np.uint8)
                affine = hipp_img.affine
            # Map original 1-2 to 1-2 in combined
            combined[(hipp_data == 1)] = 1
            combined[(hipp_data == 2)] = 2

        # Process basal ganglia labels (original 1-4 -> 3-6)
        baga_path = os.path.join(baga_dir, f'LISA_{case_id}_HF_baga.nii.gz')
        if os.path.exists(baga_path):
            baga_img = nib.load(baga_path)
            baga_data = baga_img.get_fdata()
            if combined is None:
                combined = np.zeros_like(baga_data, dtype=np.uint8)
                affine = baga_img.affine
            # Map original 1-4 to 3-6 in combined
            combined[(baga_data == 5)] = 5
            combined[(baga_data == 6)] = 6
            combined[(baga_data == 7)] = 7
            combined[(baga_data == 8)] = 8

        # Process ventricle labels (original 1-2 -> 7-8)
        vent_path = os.path.join(vent_dir, f'LISA_{case_id}_vent.nii.gz')
        if os.path.exists(vent_path):
            vent_img = nib.load(vent_path)
            vent_data = vent_img.get_fdata()
            if combined is None:
                combined = np.zeros_like(vent_data, dtype=np.uint8)
                affine = vent_img.affine
            # Map original 1-2 to 7-8 in combined
            combined[(vent_data == 3)] = 3
            combined[(vent_data == 4)] = 4

        # Save combined label
        if combined is not None:
            output_path = os.path.join(output_dir, f'LISA_{case_id}_combined.nii.gz')
            new_img = nib.Nifti1Image(combined, affine)
            nib.save(new_img, output_path)

def main():
    parser = argparse.ArgumentParser(description='Combine multiple label sets into unified labeling scheme')
    parser.add_argument('--hipp_dir', required=True, help='Directory with hippocampus labels (LISA_*_HF_hipp.nii.gz)')
    parser.add_argument('--baga_dir', required=True, help='Directory with basal ganglia labels (LISA_*_HF_baga.nii.gz)')
    parser.add_argument('--vent_dir', required=True, help='Directory with ventricle labels (LISA_*_vent.nii.gz)')
    parser.add_argument('--output_dir', required=True, help='Output directory for combined labels')

    args = parser.parse_args()

    print(f"Combining labels from:")
    print(f"- Hippocampus: {args.hipp_dir}")
    print(f"- Basal ganglia: {args.baga_dir}")
    print(f"- Ventricles: {args.vent_dir}")
    print(f"Output will be saved to: {args.output_dir}")

    combine_labels(args.hipp_dir, args.baga_dir, args.vent_dir, args.output_dir)
    print("Label combining complete!")

if __name__ == '__main__':
    main()

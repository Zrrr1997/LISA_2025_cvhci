import argparse
import os
import re
import nibabel as nib
import numpy as np

def load_nifti(path):
    return nib.load(path)

def save_nifti(data, affine, output_path):
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), output_path)

def extract_id(filename):
    match = re.search(r'\d{4}', filename)
    return match.group(0) if match else None

def main():
    parser = argparse.ArgumentParser(description="Generate labeled error maps from matched prediction and label NIfTI files.")
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory with prediction NIfTI files")
    parser.add_argument('--label_dir', type=str, required=True, help="Directory with label NIfTI files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save error NIfTI files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pred_files = {extract_id(f): f for f in os.listdir(args.pred_dir) if extract_id(f)}
    label_files = {extract_id(f): f for f in os.listdir(args.label_dir) if extract_id(f)}

    common_ids = set(pred_files.keys()) & set(label_files.keys())

    if not common_ids:
        print("No matching IDs found between prediction and label files.")
        return

    for id_ in sorted(common_ids):
        pred_path = os.path.join(args.pred_dir, pred_files[id_])
        label_path = os.path.join(args.label_dir, label_files[id_])

        pred_img = load_nifti(pred_path)
        label_img = load_nifti(label_path)

        pred_data = pred_img.get_fdata()
        label_data = label_img.get_fdata()

        # Convert to boolean masks
        pred_fg = pred_data > 0
        label_fg = label_data > 0

        error_map = np.zeros_like(pred_fg, dtype=np.uint8)

        # Oversegmentation: predicted foreground, label background
        overseg_mask = (pred_fg == True) & (label_fg == False)
        error_map[overseg_mask] = 1

        # Undersegmentation: predicted background, label foreground
        underseg_mask = (pred_fg == False) & (label_fg == True)
        error_map[underseg_mask] = 2

        output_filename = f"error_{id_}.nii.gz"
        output_path = os.path.join(args.output_dir, output_filename)
        save_nifti(error_map, pred_img.affine, output_path)

        # Compute volumes
        overseg_volume = np.sum(overseg_mask)
        underseg_volume = np.sum(underseg_mask)

        print(f"ID {id_}: Oversegmentation voxels = {overseg_volume}, Undersegmentation voxels = {underseg_volume}")
        print(f"Saved labeled error map: {output_filename}")

if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import nibabel as nib
from skimage import measure


def elongate_tip(mask, label, extension_length=3):
    out_mask = mask.copy()
    labeled = measure.label(mask == label)

    for region in measure.regionprops(labeled):
        coords = region.coords
        if coords.shape[0] < 10:
            continue

        centered = coords - np.mean(coords, axis=0)
        _, _, Vt = np.linalg.svd(centered)
        main_direction = Vt[0]

        # Project coords onto main direction
        projections = centered @ main_direction
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)

        # Choose tip with less local neighborhood (tip) by comparing neighbors
        tip_candidates = [coords[min_idx], coords[max_idx]]
        tip = None
        for candidate in tip_candidates:
            neighbor_box = mask[
                max(0, candidate[0]-1):candidate[0]+2,
                max(0, candidate[1]-1):candidate[1]+2,
                max(0, candidate[2]-1):candidate[2]+2
            ]
            if np.sum(neighbor_box == label) < 5:  # Heuristic: few neighbors = tip
                tip = candidate
                break
        if tip is None:
            tip = coords[min_idx]  # Fallback

        # Extend along main direction away from center
        tip_direction = (tip - np.mean(coords, axis=0))
        tip_direction = tip_direction / (np.linalg.norm(tip_direction) + 1e-8)

        for i in range(1, extension_length + 1):
            new_point = np.round(tip + tip_direction * i).astype(int)
            if np.all((0 <= new_point) & (new_point < mask.shape)):
                out_mask[tuple(new_point)] = label

    return out_mask


def process_file(input_path, output_path, labels, extension_length):
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.uint8)

    out_data = data.copy()
    for label in labels:
        out_data = elongate_tip(out_data, label, extension_length)

    nib.save(nib.Nifti1Image(out_data, img.affine, img.header), output_path)


def batch_process(input_dir, output_dir, labels, extension_length):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            process_file(in_path, out_path, labels, extension_length)


def main():
    parser = argparse.ArgumentParser(description='Elongate thin parts of segmented objects in NIfTI files.')
    parser.add_argument('input_dir', type=str, help='Input directory of NIfTI files')
    parser.add_argument('output_dir', type=str, help='Output directory to save processed files')
    parser.add_argument('--labels', nargs='+', type=int, default=[5, 6], help='Labels to elongate (default: 5 6)')
    parser.add_argument('--extension_length', type=int, default=4, help='Number of voxels to elongate (default: 4)')
    args = parser.parse_args()

    batch_process(args.input_dir, args.output_dir, args.labels, args.extension_length)


if __name__ == '__main__':
    main()

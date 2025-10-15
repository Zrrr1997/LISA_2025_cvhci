import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, label, distance_transform_edt


def compute_volumes(data, label1=1, label2=2):
    vol1 = np.sum(data == label1)
    vol2 = np.sum(data == label2)
    return vol1, vol2

def volume_diff_pct(vol1, vol2):
    return (abs(vol1 - vol2) / max(vol1, vol2)) * 100 if max(vol1, vol2) > 0 else 0

def keep_largest_cc(data, label_val):
    mask = (data == label_val)
    labeled, num_features = label(mask)
    if num_features == 0:
        return data
    max_label = 0
    max_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > max_size:
            max_size = size
            max_label = i
    largest_cc_mask = (labeled == max_label)
    data[data == label_val] = 0
    data[largest_cc_mask] = label_val
    return data

def harmonize_with_distance_transform(data, label1=1, label2=2, axis=0, max_iters=5):
    for _ in range(max_iters):
        mask1 = data == label1
        mask2 = data == label2

        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            break

        dt1 = distance_transform_edt(mask1)
        dt2 = distance_transform_edt(mask2)

        dt2_mirrored = np.flip(dt2, axis=axis)
        target_thickness = np.maximum(dt1, dt2_mirrored)

        growth_zone = (dt1 < target_thickness) & ~mask1
        if not np.any(growth_zone):
            break

        # Grow by one voxel into the growth zone
        dilated = binary_dilation(mask1)
        new_growth = dilated & growth_zone
        data[new_growth] = label1
    return data

def harmonize_labels(data, label1=1, label2=2, max_iters=5, th=15):
    for _ in range(max_iters):
        vol1, vol2 = compute_volumes(data, label1, label2)
        diff = volume_diff_pct(vol1, vol2)
        if diff <= th:
            break
        if vol1 < vol2:
            data = harmonize_with_distance_transform(data, label1, label2, axis=0, max_iters=1)
        else:
            data = harmonize_with_distance_transform(data, label2, label1, axis=0, max_iters=1)
    return data

def main():
    parser = argparse.ArgumentParser(description="Harmonize labels in NIfTI volumes using distance transform method.")
    parser.add_argument('--input_dir', required=True, help='Directory with .nii.gz files')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed files')
    parser.add_argument('--no_harmonize', action='store_true', help='Disable harmonization step')
    parser.add_argument('--label1', default=1, type=int, help='Label 1')
    parser.add_argument('--label2', default=2, type=int, help='Label 2')
    parser.add_argument('--th', default=15, type=int, help='Volume difference threshold')


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    label1, label2 = args.label1, args.label2
    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.nii.gz'):
            continue

        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)

        img = nib.load(input_path)
        data = img.get_fdata().astype(np.int32)

        data = keep_largest_cc(data, label_val=1)
        data = keep_largest_cc(data, label_val=2)

        vol1, vol2 = compute_volumes(data, label1=label1, label2=label2)
        diff = volume_diff_pct(vol1, vol2)
        print(f"{fname} - initial volume diff: {diff:.2f}%")

        for i in range(1, 9):
            data = keep_largest_cc(data, label_val=i)

        if diff > args.th and not args.no_harmonize:
            data = harmonize_labels(data, label1=label1, label2=label2, th=args.th)
            for i in range(1, 9):
                data = keep_largest_cc(data, label_val=i)
            vol1_final, vol2_final = compute_volumes(data, label1=label1, label2=label2)
            final_diff = volume_diff_pct(vol1_final, vol2_final)
            print(f"{fname} - final volume diff: {final_diff:.2f}%")
        else:
            print(f"{fname} - no harmonization needed.")

        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, output_path)

if __name__ == "__main__":
    main()

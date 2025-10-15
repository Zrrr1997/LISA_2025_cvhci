import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, label
import shutil

def compute_volumes(data, label1=1, label2=2):
    vol1 = np.sum(data == label1)
    vol2 = np.sum(data == label2)
    return vol1, vol2

def volume_diff_pct(vol1, vol2):
    return (abs(vol1 - vol2) / max(vol1, vol2)) * 100 if max(vol1, vol2) > 0 else 0

def label_extent_z(data, label_val):
    z_slices = np.any(np.any(data == label_val, axis=0), axis=0)
    return np.where(z_slices)[0]

def mirror_missing_slices(data, label1=1, label2=2):
    for z in range(data.shape[2]):
        slice_data = data[:, :, z]

        mask1 = slice_data == label1
        mask2 = slice_data == label2

        count1 = np.sum(mask1)
        count2 = np.sum(mask2)

        if count1 == 0 and count2 == 0:
            continue  # Nothing to mirror

        if count1 > 0 and count2 == 0:
            # Only label1 exists — mirror it into label2
            flipped = np.flip(mask1, axis=0)  # flip along left-right axis
            data[:, :, z][flipped] = label2

        elif count2 > 0 and count1 == 0:
            # Only label2 exists — mirror it into label1
            flipped = np.flip(mask2, axis=0)
            data[:, :, z][flipped] = label1


    return data




def dilate_slices(data, smaller_label, larger_label):
    for z in range(data.shape[2]):
        small_mask = data[:, :, z] == smaller_label
        large_mask = data[:, :, z] == larger_label

        small_count = np.sum(small_mask)
        large_count = np.sum(large_mask)

        if large_count == 0 or small_count / large_count >= 0.9:
            continue  # No need to dilate

        dilated = binary_dilation(small_mask)
        data[:, :, z][dilated] = smaller_label
    return data

def dilate_slices_high_diff(data, label1=1, label2=2, threshold=60):
    for z in range(data.shape[2]):
        while True:
            mask1 = data[:, :, z] == label1
            mask2 = data[:, :, z] == label2

            vol1 = np.sum(mask1)
            vol2 = np.sum(mask2)

            if vol1 == 0 or vol2 == 0:
                break  # skip if one is missing

            if vol1 < vol2:
                smaller_mask = mask1
                smaller_label = label1
                larger_vol = vol2
            else:
                smaller_mask = mask2
                smaller_label = label2
                larger_vol = vol1

            smaller_vol = np.sum(smaller_mask)
            diff_pct = abs(smaller_vol - larger_vol) / larger_vol * 100

            if diff_pct <= threshold:
                break

            dilated = binary_dilation(smaller_mask)
            new_mask = dilated & (~smaller_mask)
            if not np.any(new_mask):
                break  # can't grow further

            data[:, :, z][new_mask] = smaller_label

    return data


def harmonize_labels(data, label1=1, label2=2, max_iters=5):
    for _ in range(max_iters):
        vol1, vol2 = compute_volumes(data, label1, label2)
        diff = volume_diff_pct(vol1, vol2)

        if diff <= 15:
            break

        if vol1 < vol2:
            smaller, larger = label1, label2
        else:
            smaller, larger = label2, label1


        vol1, vol2 = compute_volumes(data, label1, label2)
        diff = volume_diff_pct(vol1, vol2)
        if diff <= 15:
            break



        #data = dilate_slices(data, smaller, larger)
        data = dilate_slices_high_diff(data, smaller, larger, threshold=60)
        data = mirror_missing_slices(data, label1=label1, label2=label2)


    return data

def keep_largest_cc(data, label_val):
    mask = (data == label_val)
    labeled, num_features = label(mask)
    if num_features == 0:
        return data  # no components to keep

    # Find largest connected component
    max_label = 0
    max_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > max_size:
            max_size = size
            max_label = i

    # Create new mask keeping only the largest CC for label_val
    largest_cc_mask = (labeled == max_label)

    # Remove all other voxels of this label from data
    data[data == label_val] = 0
    data[largest_cc_mask] = label_val

    return data

def main():
    parser = argparse.ArgumentParser(description="Harmonize label 1 and 2 in NIfTI volumes.")
    parser.add_argument('--input_dir', required=True, help='Directory with .nii.gz files')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed files')
    parser.add_argument('--no_harmonize', action='store_true', help='Disable harmonization step (default is enabled)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.nii.gz'):
            continue

        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)

        img = nib.load(input_path)
        data = img.get_fdata().astype(np.int32)

        # Keep only largest connected component for labels 1 and 2
        data = keep_largest_cc(data, label_val=1)
        data = keep_largest_cc(data, label_val=2)

        vol1, vol2 = compute_volumes(data)
        diff = volume_diff_pct(vol1, vol2)
        print(f"{fname} - initial volume diff: {diff:.2f}%")
        for i in np.arange(1, 9):
            data = keep_largest_cc(data, label_val=i)
        if diff > 15 and not args.no_harmonize:
            data = harmonize_labels(data)
            for i in np.arange(1, 9):
                data = keep_largest_cc(data, label_val=i)
            vol1_final, vol2_final = compute_volumes(data)
            final_diff = volume_diff_pct(vol1_final, vol2_final)
            print(f"{fname} - final volume diff: {final_diff:.2f}%")

            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, output_path)
        else:
            for i in np.arange(1, 9):
                data = keep_largest_cc(data, label_val=i)
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, output_path)
            print(f"{fname} - copied original but kept only largest component (no harmonization needed).")

if __name__ == "__main__":
    main()

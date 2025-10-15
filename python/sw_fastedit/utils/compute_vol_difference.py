import os
import argparse
import nibabel as nib
import numpy as np

def compute_volume_and_zdiff(nifti_file):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    label_1 = 5
    label_2 = 6
    # Volume (voxel count)
    vol1 = np.sum(data == label_1)
    vol2 = np.sum(data == label_2)

    # Volume % difference
    if max(vol1, vol2) > 0:
        vol_diff_pct = (abs(vol1 - vol2) / max(vol1, vol2)) * 100
    else:
        vol_diff_pct = None

    # Z-slice extent function
    def label_extent_z(label_val):
        z_slices = np.any(np.any(data == label_val, axis=0), axis=0)
        indices = np.where(z_slices)[0]
        return (indices[0], indices[-1]) if indices.size > 0 else (None, None)

    start1, end1 = label_extent_z(label_1)
    start2, end2 = label_extent_z(label_2)

    z_diff_start = start2 - start1 if None not in (start1, start2) else None
    z_diff_end = end2 - end1 if None not in (end1, end2) else None

    # Slice-wise percentage differences (only where both labels are present)
    slice_pct_diffs = []
    for z in range(data.shape[2]):
        slice_label1 = data[:, :, z] == label_1
        slice_label2 = data[:, :, z] == label_2

        if np.any(slice_label1) and np.any(slice_label2):
            vol1_slice = np.sum(slice_label1)
            vol2_slice = np.sum(slice_label2)
            if max(vol1_slice, vol2_slice) > 0:
                pct_diff = abs(vol1_slice - vol2_slice) / max(vol1_slice, vol2_slice) * 100
                slice_pct_diffs.append(pct_diff)

    max_slice_diff_pct = max(slice_pct_diffs) if slice_pct_diffs else 0
    mean_slice_diff_pct = np.mean(slice_pct_diffs) if slice_pct_diffs else 0

    return {
        'file': os.path.basename(nifti_file),
        'volume_label1': vol1,
        'volume_label2': vol2,
        'volume_diff_pct': vol_diff_pct,
        'z_diff_start': z_diff_start,
        'z_diff_end': z_diff_end,
        'max_slice_diff_pct': max_slice_diff_pct,
        'mean_slice_diff_pct': mean_slice_diff_pct
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze label volumes in NIfTI files")
    parser.add_argument('--input_dir', required=True, help="Directory containing .nii.gz files")
    args = parser.parse_args()

    results = []

    for fname in os.listdir(args.input_dir):
        if fname.endswith('.nii.gz'):
            path = os.path.join(args.input_dir, fname)
            results.append(compute_volume_and_zdiff(path))

    # Print results
    print("\nPer-File Results:")
    all_vol_diffs = []
    all_z_start_diffs = []
    all_z_end_diffs = []
    all_max_slice_diffs = []
    all_mean_slice_diffs = []
    vols_1 = []

    for res in results:
        print(f"\nFile: {res['file']}")
        print(f"  Volume label 1: {res['volume_label1']}")
        print(f"  Volume label 2: {res['volume_label2']}")
        print(f"  Volume difference (%): {res['volume_diff_pct']:.2f}%" if res['volume_diff_pct'] is not None else "  Volume difference (%): N/A")
        print(f"  Z-slice start difference (slices): {res['z_diff_start']}")
        print(f"  Z-slice end difference (slices): {res['z_diff_end']}")
        print(f"  Max slice-wise volume difference (%): {res['max_slice_diff_pct']:.2f}%")
        print(f"  Mean slice-wise volume difference (%): {res['mean_slice_diff_pct']:.2f}%")

        if res['volume_diff_pct'] is not None:
            all_vol_diffs.append(res['volume_diff_pct'])
        if res['z_diff_start'] is not None:
            all_z_start_diffs.append(res['z_diff_start'])
        if res['z_diff_end'] is not None:
            all_z_end_diffs.append(res['z_diff_end'])
        all_max_slice_diffs.append(res['max_slice_diff_pct'])
        all_mean_slice_diffs.append(res['mean_slice_diff_pct'])

    # Print mean differences
    print("\n=== Mean Differences Across Files ===")
    if all_vol_diffs:
        print(f"Mean volume difference (%): {np.mean(all_vol_diffs):.2f}%")
    if all_z_start_diffs:
        print(f"Mean Z-slice start difference (slices): {np.mean(all_z_start_diffs):.2f}")
    if all_z_end_diffs:
        print(f"Mean Z-slice end difference (slices): {np.mean(all_z_end_diffs):.2f}")
    if all_max_slice_diffs:
        print(f"Mean max slice-wise volume difference (%): {np.mean(all_max_slice_diffs):.2f}%")
    if all_mean_slice_diffs:
        print(f"Mean mean slice-wise volume difference (%): {np.mean(all_mean_slice_diffs):.2f}%")

if __name__ == "__main__":
    main()

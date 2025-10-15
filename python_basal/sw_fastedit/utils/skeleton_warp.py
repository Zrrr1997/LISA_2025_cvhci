import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_nifti(filepath, dtype=np.float32):
    nii = nib.load(filepath)
    data = nii.get_fdata(dtype=dtype)
    return data, nii.affine, nii.header, nii

def save_nifti(data, affine, header, filepath):
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, filepath)

def dice_coefficient(vol1, vol2, class_id):
    mask1 = (vol1 == class_id)
    mask2 = (vol2 == class_id)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    total = np.count_nonzero(mask1) + np.count_nonzero(mask2)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total

def overall_dice(vol1, vol2, classes=[1,2]):
    return np.mean([dice_coefficient(vol1, vol2, c) for c in classes])

def interpolate_classes_separately(pred_vol, label_vol, classes=[1,2], alpha=0.5):
    warped = np.zeros_like(pred_vol, dtype=np.uint8)
    for c in classes:
        pred_mask = (pred_vol == c).astype(np.float32)
        label_mask = (label_vol == c).astype(np.float32)
        interp = alpha * pred_mask + (1 - alpha) * label_mask
        warped[interp >= 0.5] = c
    return warped

def find_most_similar_label(pred_vol, pred_nii, label_files):
    best_sim, best_name, best_vol = -1, None, None
    for label_name, label_path in label_files.items():
        _, _, _, label_nii = load_nifti(label_path)
        resampled_nii = resample_from_to(label_nii, pred_nii, order=0)
        resampled_data = resampled_nii.get_fdata(dtype=np.float32)
        resampled_data = np.round(resampled_data).astype(np.uint8)  # ← Fix here
        sim = overall_dice(pred_vol, resampled_data)
        if sim > best_sim:
            best_sim = sim
            best_name = label_name
            best_vol = resampled_data
    return best_name, best_vol, best_sim

def process_prediction(args):
    pred_fname, pred_path, output_dir, threshold, label_files = args
    pred_vol, affine, header, pred_nii = load_nifti(pred_path, dtype=np.float32)
    pred_vol = np.round(pred_vol).astype(np.uint8)  # Ensure integer label image

    nonzero_coords = np.argwhere(pred_vol > 0)

    if nonzero_coords.size > 0:
        min_x, min_y, min_z = nonzero_coords.min(axis=0)
        max_x, max_y, max_z = nonzero_coords.max(axis=0)
    else:
        print(f"{pred_fname} has no foreground voxels.")

    best_label_name, best_label_vol, sim = find_most_similar_label(pred_vol, pred_nii, label_files)

    #if sim >= threshold:
    #    warped = interpolate_classes_separately(pred_vol, best_label_vol, alpha=0.3)
    if sim >= threshold:
        warped = pred_vol.copy()

        # Define Z split point
        z_half = min_z + (max_z - min_z) // 2

        # Interpolate only the second half of Z
        interpolated_half = interpolate_classes_separately(
            pred_vol[:, :, z_half:max_z + 1],
            best_label_vol[:, :, z_half:max_z + 1],
            alpha=0.3
        )

        warped[:, :, z_half:max_z + 1] = interpolated_half
    else:
        warped = interpolate_classes_separately(pred_vol, best_label_vol, alpha=1.0)

    output_path = os.path.join(output_dir, pred_fname)
    save_nifti(warped, affine, header, output_path)
    return f"{pred_fname}: Best match = {best_label_name} (Dice = {sim:.3f})"

def main(labels_dir, predictions_dir, output_dir, threshold=0.8, n_workers=4):
    label_files = {
        fname: os.path.join(labels_dir, fname)
        for fname in os.listdir(labels_dir)
        if fname.endswith(('.nii', '.nii.gz'))
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    for fname in os.listdir(predictions_dir):
        if not fname.endswith(('.nii', '.nii.gz')):
            continue
        pred_path = os.path.join(predictions_dir, fname)
        tasks.append((fname, pred_path, output_dir, threshold, label_files))

    with Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_prediction, tasks), total=len(tasks)):
            print(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Efficient label warping with class-wise interpolation.")
    parser.add_argument("--label_dir", required=True, help="Directory of label NIfTI files")
    parser.add_argument("--pred_dir", required=True, help="Directory of prediction NIfTI files")
    parser.add_argument("--out_dir", required=True, help="Directory for output NIfTI files")
    parser.add_argument("--threshold", type=float, default=0.8, help="Dice threshold for interpolation")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel processes")
    args = parser.parse_args()

    main(args.label_dir, args.pred_dir, args.out_dir, args.threshold, args.workers)

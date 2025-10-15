import argparse
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def intensity_guided_morphology_minmax(image, prediction, margin_dilate=10, margin_erode=30, iterations=1):
    """
    Use min/max intensity inside prediction with margins for erosion and dilation.
    Dilate outside voxels with intensity inside [min-margin_dilate, max+margin_dilate].
    Erode inside boundary voxels with intensity outside [min-margin_erode, max+margin_erode].
    """
    pred_bin = (prediction > 0).astype(np.uint8)
    if np.sum(pred_bin) == 0:
        return pred_bin
    struct = generate_binary_structure(3, 1)

    # Identify boundary voxels of the predicted mask
    eroded_once = binary_erosion(pred_bin, structure=struct, iterations=1)
    boundary = pred_bin.astype(bool) & (~eroded_once)

    # Get intensities at boundary voxels only
    boundary_intensities = image[boundary]
    min_boundary = boundary_intensities.min()
    max_boundary = boundary_intensities.max()

    # Define erosion thresholds (outside this range => erode)
    erode_min = min_boundary - margin_erode
    erode_max = max_boundary + margin_erode

    # Mark boundary pixels for erosion if intensity is outside erosion thresholds
    erosion_mask = np.zeros_like(pred_bin, dtype=bool)
    erosion_mask[boundary] = (image[boundary] < erode_min) | (image[boundary] > erode_max)

    # Apply erosion on marked pixels
    pred_eroded = pred_bin.copy()
    pred_eroded[erosion_mask] = 0

    # Define dilation thresholds (inside this range => allow dilation)
    dilate_min = min_boundary - margin_dilate
    dilate_max = max_boundary + margin_dilate

    # Find candidate pixels for dilation (neighbors of eroded mask outside current mask)
    outside_mask = (pred_eroded == 0)
    dilate_candidates = binary_dilation(pred_eroded, structure=struct) & outside_mask

    # Select candidates whose intensities fall within dilation thresholds
    dilation_mask = np.zeros_like(pred_bin, dtype=bool)
    dilation_mask[dilate_candidates] = (image[dilate_candidates] >= dilate_min) & (image[dilate_candidates] <= dilate_max)

    # Final dilated mask
    pred_dilated = pred_eroded.copy()
    pred_dilated[dilation_mask] = 1


    return pred_dilated.astype(np.uint8)

def process_case(img_path, pred_path, label_path, output_path, margin_dilate, margin_erode, iterations):
    img_nii = nib.load(img_path)
    pred_nii = nib.load(pred_path)
    label_nii = nib.load(label_path)

    img_data = img_nii.get_fdata()
    pred_data = pred_nii.get_fdata()
    label_data = label_nii.get_fdata()

    # Process label 1
    pred_label1 = (pred_data == 1).astype(np.uint8)
    label_label1 = (label_data == 1).astype(np.uint8)
    dice_before_l1 = dice_score(label_label1, pred_label1)
    refined_label1 = pred_label1
    for i in range(iterations):
        refined_label1 = intensity_guided_morphology_minmax(
            img_data, refined_label1, margin_dilate=margin_dilate, margin_erode=margin_erode, iterations=1)
    dice_after_l1 = dice_score(label_label1, refined_label1)

    # Process label 2
    pred_label2 = (pred_data == 2).astype(np.uint8)
    label_label2 = (label_data == 2).astype(np.uint8)
    dice_before_l2 = dice_score(label_label2, pred_label2)
    refined_label2 = pred_label2
    for i in range(iterations):
        refined_label2 = intensity_guided_morphology_minmax(
            img_data, refined_label2, margin_dilate=margin_dilate, margin_erode=margin_erode, iterations=1)
    dice_after_l2 = dice_score(label_label2, refined_label2)

    # Combine refined masks (priority: label 2 overwrites label 1 if overlap)
    refined_pred = np.zeros_like(pred_data, dtype=np.uint8)
    refined_pred[refined_label1 == 1] = 1
    refined_pred[refined_label2 == 1] = 2

    refined_nii = nib.Nifti1Image(refined_pred, affine=pred_nii.affine, header=pred_nii.header)
    nib.save(refined_nii, output_path)

    return (dice_before_l1, dice_after_l1), (dice_before_l2, dice_after_l2)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith('.nii.gz')])
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(args.labels_dir) if f.endswith('.nii.gz')])

    def base_id(filename):
        return filename.split('_')[0] + '_' + filename.split('_')[1]

    pred_map = {base_id(f): f for f in pred_files}
    label_map = {base_id(f): f for f in label_files}
    image_map = {base_id(f): f for f in image_files}

    common_ids = sorted(set(pred_map.keys()) & set(label_map.keys()) & set(image_map.keys()))

    dice_before_all_l1 = []
    dice_after_all_l1 = []
    dice_before_all_l2 = []
    dice_after_all_l2 = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for id_ in common_ids:
            img_path = os.path.join(args.image_dir, image_map[id_])
            pred_path = os.path.join(args.pred_dir, pred_map[id_])
            label_path = os.path.join(args.labels_dir, label_map[id_])
            output_path = os.path.join(args.output_dir, pred_map[id_])

            futures.append(
                executor.submit(process_case, img_path, pred_path, label_path, output_path,
                                args.margin_dilate, args.margin_erode, args.iterations)
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
            (dice_b_l1, dice_a_l1), (dice_b_l2, dice_a_l2) = future.result()
            dice_before_all_l1.append(dice_b_l1)
            dice_after_all_l1.append(dice_a_l1)
            dice_before_all_l2.append(dice_b_l2)
            dice_after_all_l2.append(dice_a_l2)

    mean_before_l1 = np.mean(dice_before_all_l1)
    mean_after_l1 = np.mean(dice_after_all_l1)
    mean_before_l2 = np.mean(dice_before_all_l2)
    mean_after_l2 = np.mean(dice_after_all_l2)

    print(f"Label 1 Mean Dice before refinement: {mean_before_l1:.4f}")
    print(f"Label 1 Mean Dice after refinement:  {mean_after_l1:.4f}")
    print(f"Label 2 Mean Dice before refinement: {mean_before_l2:.4f}")
    print(f"Label 2 Mean Dice after refinement:  {mean_after_l2:.4f}")

    print(f"Overall Mean Dice before refinement: {(mean_before_l1 + mean_before_l2)/2:.4f}")
    print(f"Overall Mean Dice after refinement:  {(mean_after_l1 + mean_after_l2)/2:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Min/Max Intensity Guided Morphology on Multiple Labels with Dice Evaluation")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with MRI images (NIfTI)')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted masks (NIfTI)')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory with ground truth masks (NIfTI)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save refined masks')
    parser.add_argument('--margin_dilate', type=float, default=20,
                        help='Margin around min/max for dilation (default: 10)')
    parser.add_argument('--margin_erode', type=float, default=20,
                        help='Margin around min/max for erosion (default: 30)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of erosion/dilation iterations (default: 1)')
    args = parser.parse_args()

    main(args)

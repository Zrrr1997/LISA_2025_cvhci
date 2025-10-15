import os
import re
import argparse
import SimpleITK as sitk
from glob import glob
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def extract_id(fname):
    match = re.search(r'\d{4}', fname)
    return match.group(0) if match else None


def find_matching_pairs(pred_dir, label_dir):
    pred_files = sorted(glob(os.path.join(pred_dir, "*.nii*")))
    label_files = sorted(glob(os.path.join(label_dir, "*.nii*")))
    pred_map = {extract_id(f): f for f in pred_files if extract_id(f)}
    label_map = {extract_id(f): f for f in label_files if extract_id(f)}
    common_ids = sorted(set(pred_map.keys()) & set(label_map.keys()))
    return [(i, pred_map[i], label_map[i]) for i in common_ids]


def downsample_mask(mask, factor=2):
    shrink = [factor] * mask.GetDimension()
    return sitk.Shrink(mask, shrink)


def fit_bspline_transform(args):
    image_id, pred_path, gt_path, label_value = args
    try:
        # Load images
        fixed_img = sitk.ReadImage(gt_path)
        moving_img = sitk.ReadImage(pred_path)

        # Create binary masks, cast to float
        fixed_mask = sitk.Cast(sitk.Equal(fixed_img, label_value), sitk.sitkFloat32)
        moving_mask = sitk.Cast(sitk.Equal(moving_img, label_value), sitk.sitkFloat32)

        # Downsample masks to speed up registration
        fixed_mask_ds = downsample_mask(fixed_mask, factor=2)
        moving_mask_ds = downsample_mask(moving_mask, factor=2)

        # Setup B-spline grid (coarser = faster)
        grid_spacing = 20  # bigger spacing for speed, tune if needed
        mesh_size = [max(1, int(sz / grid_spacing)) for sz in fixed_img.GetSize()]
        initial_transform = sitk.BSplineTransformInitializer(fixed_img, mesh_size)

        registration_method = sitk.ImageRegistrationMethod()

        # Use MeanSquares metric for speed on binary masks
        registration_method.SetMetricAsMeanSquares()

        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.3)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Faster optimizer parameters
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=30,
            maximumNumberOfCorrections=3,
            maximumNumberOfFunctionEvaluations=500,
            costFunctionConvergenceFactor=1e+7)

        registration_method.SetShrinkFactorsPerLevel([2, 1])
        registration_method.SetSmoothingSigmasPerLevel([1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Run registration on downsampled masks
        final_transform = registration_method.Execute(fixed_mask_ds, moving_mask_ds)

        return image_id, final_transform

    except Exception as e:
        print(f"[ERROR] Failed to register {image_id} (label {label_value}): {e}")
        return image_id, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory of predicted NIfTIs")
    parser.add_argument("--gt_dir", required=True, help="Directory of ground-truth labels")
    parser.add_argument("--out_path", required=True, help="Path to save the learned transforms (pickle)")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of processes to use")
    args = parser.parse_args()

    pairs = find_matching_pairs(args.pred_dir, args.gt_dir)
    if not pairs:
        print("No matching pairs found.")
        return

    transforms = {}

    for label_id in [1, 2]:
        print(f"Fitting BSpline transforms for label {label_id} on {len(pairs)} pairs...")
        job_args = [(image_id, pred_path, gt_path, label_id) for image_id, pred_path, gt_path in pairs]
        with Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap(fit_bspline_transform, job_args), total=len(job_args)))

        transforms[label_id] = {image_id: tx for image_id, tx in results if tx is not None}

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pickle.dump(transforms, f)

    print(f"Saved transform dictionary to: {args.out_path}")


if __name__ == "__main__":
    main()

import os
import re
import argparse
import SimpleITK as sitk
from glob import glob
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def extract_id(fname):
    match = re.search(r'\d{4}', fname)
    return match.group(0) if match else None


def load_transforms(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_transform_to_file(args):
    pred_path, out_path, transforms = args
    try:
        image_id = extract_id(pred_path)
        if image_id is None:
            raise ValueError(f"Could not extract ID from filename: {pred_path}")

        pred_img = sitk.ReadImage(pred_path)
        new_pred = sitk.Image(pred_img.GetSize(), sitk.sitkUInt8)
        new_pred.CopyInformation(pred_img)

        # For each label, check if we have a transform for this image and apply it
        for lbl, tx_dict in transforms.items():
            if image_id not in tx_dict or tx_dict[image_id] is None:
                continue

            tx = tx_dict[image_id]

            # Create binary mask of label in predicted image
            mask = sitk.Cast(sitk.Equal(pred_img, lbl), sitk.sitkFloat32)

            # Apply transform to mask
            resampled = sitk.Resample(mask, pred_img, tx, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)

            # Threshold and assign label value
            resampled_lbl = sitk.Cast(sitk.Greater(resampled, 0.5), sitk.sitkUInt8) * lbl

            # Combine with output image
            new_pred = sitk.Maximum(new_pred, resampled_lbl)

        sitk.WriteImage(new_pred, out_path)
        return True

    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Input predicted NIfTI directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for transformed predictions")
    parser.add_argument("--transform_path", required=True, help="Pickle file from fitting script")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of parallel processes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    transforms = load_transforms(args.transform_path)

    pred_files = sorted(glob(os.path.join(args.pred_dir, "*.nii*")))
    jobs = [(pred_path, os.path.join(args.out_dir, os.path.basename(pred_path)), transforms) for pred_path in pred_files]

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(apply_transform_to_file, jobs), total=len(jobs)))

    print(f"Finished {sum(results)} / {len(results)} files successfully.")


if __name__ == "__main__":
    main()

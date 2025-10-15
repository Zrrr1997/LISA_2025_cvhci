import os
import re
import argparse
import nibabel as nib
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def extract_id(filename):
    match = re.search(r'\d{4}', filename)
    return match.group(0) if match else None


def find_matching_pairs(img_dir, label_dir):
    img_files = glob(os.path.join(img_dir, "*.nii*"))
    label_files = glob(os.path.join(label_dir, "*.nii*"))

    img_map = {extract_id(f): f for f in img_files if extract_id(f)}
    label_map = {extract_id(f): f for f in label_files if extract_id(f)}

    common_ids = sorted(set(img_map.keys()) & set(label_map.keys()))
    return [(img_map[i], label_map[i]) for i in common_ids]


def compute_single_pair(args):
    img_path, label_path = args
    try:
        img = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata()

        global_min = float(np.min(img))
        global_max = float(np.max(img))
        global_range = global_max - global_min if global_max > global_min else 1e-6

        result = {}
        for lbl in (1, 2):
            mask = label == lbl
            if np.any(mask):
                values = img[mask]
                p_low = np.percentile(values, 0.05)
                p_high = np.percentile(values, 99.95)


                result[lbl] = {
                    "0.5%": float(p_low),
                    "99.5%": float(p_high),
                    "rel_0.5%": float((p_low - global_min) / global_range),
                    "rel_99.5%": float((p_high - global_min) / global_range),
                }
            else:
                result[lbl] = None

        return result
    except Exception as e:
        print(f"Error processing {img_path} and {label_path}: {e}")
        return {1: None, 2: None}


def aggregate_results(results):
    stats = {1: [], 2: []}
    for res in results:
        for lbl in (1, 2):
            if res[lbl] is not None:
                stats[lbl].append(res[lbl])

    summary = {}
    for lbl in (1, 2):
        if stats[lbl]:
            p005s = [r["0.5%"] for r in stats[lbl]]
            p995s = [r["99.5%"] for r in stats[lbl]]
            rel_p005s = [r["rel_0.5%"] for r in stats[lbl]]
            rel_p995s = [r["rel_99.5%"] for r in stats[lbl]]

            summary[lbl] = {
                "0.5%": float(np.mean(p005s)),
                "99.5%": float(np.mean(p995s)),
                "rel_0.5%": float(np.min(rel_p005s)),
                "rel_99.5%": float(np.max(rel_p995s)),
            }
        else:
            summary[lbl] = None
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Directory with input NIfTI MRI images")
    parser.add_argument("--label_dir", required=True, help="Directory with label NIfTI files")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of parallel processes to use")
    args = parser.parse_args()

    pairs = find_matching_pairs(args.img_dir, args.label_dir)
    if not pairs:
        print("No matching image-label pairs found.")
        exit(1)

    print(f"Found {len(pairs)} image-label pairs. Processing...")

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(compute_single_pair, pairs), total=len(pairs)))

    summary = aggregate_results(results)
    for lbl in (1, 2):
        print(f"\nLabel {lbl}:")
        if summary[lbl] is not None:
            print(f"  0.5%        = {summary[lbl]['0.5%']:.4f}")
            print(f"  99.5%       = {summary[lbl]['99.5%']:.4f}")
            print(f"  rel_0.5%    = {summary[lbl]['rel_0.5%']:.4f}  (0 = global min, 1 = global max)")
            print(f"  rel_99.5%   = {summary[lbl]['rel_99.5%']:.4f}")
        else:
            print("  No data for this label.")

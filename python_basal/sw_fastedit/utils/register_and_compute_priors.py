import os
from glob import glob
import numpy as np
import nibabel as nib
import ants
from concurrent.futures import ProcessPoolExecutor

def find_best_label_match(mri_file, label_files):
    base_mri = os.path.basename(mri_file)
    # Assuming prefix is first two underscore parts: e.g. LISA_1015
    parts = base_mri.split('_')
    if len(parts) < 2:
        prefix = base_mri.split('.')[0]  # fallback to filename without extension
    else:
        prefix = parts[0] + '_' + parts[1]
    candidates = [lf for lf in label_files if os.path.basename(lf).startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Choose label filename closest in length to MRI filename
        candidates.sort(key=lambda x: abs(len(os.path.basename(x)) - len(base_mri)))
        return candidates[0]
    else:
        return None

def register_and_transform(pair, template_ants, registered_label_dir):
    mri_path, label_path = pair
    mri_basename = os.path.basename(mri_path)
    try:
        print(f"Processing: {mri_basename}")
        # Load MRI and label as ANTs images
        moving_img = ants.image_read(mri_path)
        fixed_img = template_ants

        # Register moving (current MRI) to fixed (template MRI)
        reg = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN')

        # Load label image as ANTs image
        label_img = ants.image_read(label_path)

        # Apply transform to label image (nearest neighbor interpolation)
        warped_label = ants.apply_transforms(fixed=fixed_img, moving=label_img,
                                             transformlist=reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')

        # Save registered label
        out_path = os.path.join(registered_label_dir, mri_basename.replace('.nii.gz', '_registered_label.nii.gz'))
        ants.image_write(warped_label, out_path)
        return out_path
    except Exception as e:
        print(f"Error processing {mri_basename}: {e}")
        return None

def compute_priors(registered_label_dir, output_intersection_path, output_union_path):
    registered_labels = glob(os.path.join(registered_label_dir, '*_registered_label.nii.gz'))
    if not registered_labels:
        print("No registered labels found for prior computation.")
        return

    sum_data = None
    count = 0

    for label_file in registered_labels:
        img = nib.load(label_file)
        data = img.get_fdata().astype(np.uint8)
        if sum_data is None:
            sum_data = np.zeros_like(data, dtype=np.float32)
        sum_data += data
        count += 1

    # Intersection = voxels present in all registered labels
    intersection = (sum_data == count).astype(np.uint8)

    # Union = voxels present in at least one label
    union = (sum_data >= 1).astype(np.uint8)

    # Save priors
    template_img = nib.load(registered_labels[0])
    nib.save(nib.Nifti1Image(intersection, template_img.affine, template_img.header), output_intersection_path)
    nib.save(nib.Nifti1Image(union, template_img.affine, template_img.header), output_union_path)

    print(f"Priors saved:\n- Intersection: {output_intersection_path}\n- Union: {output_union_path}")

def register_pair(args):
    pair, template_ants, registered_label_dir = args
    return register_and_transform(pair, template_ants, registered_label_dir)

def main(mri_dir, label_dir, template_mri_path, registered_label_dir, output_intersection_path, output_union_path, num_workers=8):
    os.makedirs(registered_label_dir, exist_ok=True)

    mri_files = sorted(glob(os.path.join(mri_dir, '*.nii.gz')))
    label_files = sorted(glob(os.path.join(label_dir, '*.nii.gz')))
    template_ants = ants.image_read(template_mri_path)

    paired = []
    for mri_file in mri_files:
        label_file = find_best_label_match(mri_file, label_files)
        if label_file and os.path.exists(label_file):
            paired.append((mri_file, label_file))
        else:
            print(f"Warning: No label found for {os.path.basename(mri_file)}, skipping.")

    print(f"Found {len(paired)} pairs for registration.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(register_pair, [(pair, template_ants, registered_label_dir) for pair in paired]))

    # Filter out failed registrations
    registered_label_paths = [res for res in results if res is not None]

    if len(registered_label_paths) == 0:
        print("No labels registered successfully. Exiting.")
        return

    compute_priors(registered_label_dir, output_intersection_path, output_union_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register labels to a template MRI and compute prior maps (intersection and union).")
    parser.add_argument('--mri_dir', required=True, help='Directory containing MRI .nii.gz files')
    parser.add_argument('--label_dir', required=True, help='Directory containing label .nii.gz files')
    parser.add_argument('--template_mri', required=True, help='Template MRI filepath (.nii.gz)')
    parser.add_argument('--registered_label_dir', required=True, help='Output directory to save registered label images')
    parser.add_argument('--output_intersection', required=True, help='Output file path for intersection prior (.nii.gz)')
    parser.add_argument('--output_union', required=True, help='Output file path for union prior (.nii.gz)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for registration')

    args = parser.parse_args()

    main(args.mri_dir, args.label_dir, args.template_mri, args.registered_label_dir,
         args.output_intersection, args.output_union, args.num_workers)

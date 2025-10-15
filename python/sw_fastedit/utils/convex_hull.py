import os
import argparse
import numpy as np
import nibabel as nib
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import label

def compute_component_convex_hull(component_mask):
    points = np.argwhere(component_mask > 0)
    if points.shape[0] < 4:  # ConvexHull needs at least 4 non-coplanar points
        print("Component too small for convex hull — skipping.")
        return np.zeros_like(component_mask, dtype=np.uint8)

    hull = ConvexHull(points)
    delaunay = Delaunay(points[hull.vertices])

    hull_mask = np.zeros_like(component_mask, dtype=np.uint8)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    for x in range(min_coords[0], max_coords[0] + 1):
        for y in range(min_coords[1], max_coords[1] + 1):
            for z in range(min_coords[2], max_coords[2] + 1):
                if delaunay.find_simplex([x, y, z]) >= 0:
                    hull_mask[x, y, z] = 1
    return hull_mask

def compute_largest_components_convex_hull(input_nifti, output_nifti):
    img = nib.load(input_nifti)
    data = img.get_fdata()
    binary_mask = (data > 0).astype(np.uint8)

    labeled_array, num_features = label(binary_mask)
    if num_features == 0:
        raise ValueError("No connected components found in the mask.")

    print(f"Found {num_features} connected components.")

    component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    sorted_components = sorted(component_sizes, key=lambda x: x[1], reverse=True)
    largest_components = [comp_id for comp_id, size in sorted_components[:2]]

    final_hull = np.zeros_like(data, dtype=np.uint8)

    for comp_id in largest_components:
        comp_mask = (labeled_array == comp_id).astype(np.uint8)
        print(f"Processing component {comp_id} with size {np.sum(comp_mask)}")
        hull_mask = compute_component_convex_hull(comp_mask)
        final_hull = np.maximum(final_hull, hull_mask)

    nib.save(nib.Nifti1Image(final_hull, affine=img.affine, header=img.header), output_nifti)
    print(f"Saved convex hull of largest components to {output_nifti}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute convex hull of the two largest connected components in a binary NIfTI mask.")
    parser.add_argument("--input_nifti", required=True, help="Path to the input binary NIfTI file")
    parser.add_argument("--output_nifti", required=True, help="Path to save the convex hull NIfTI file")
    args = parser.parse_args()

    compute_largest_components_convex_hull(args.input_nifti, args.output_nifti)

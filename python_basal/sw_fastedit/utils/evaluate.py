import argparse
import numpy as np
import nibabel as nib
import os
import glob
from collections import defaultdict
import csv
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import label

def keep_largest_connected_component(mask: np.ndarray):
    if mask.ndim != 3:
        raise ValueError("Input mask must be a 3D array")

    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return np.zeros_like(mask, dtype=np.float32)

    component_sizes = np.bincount(labeled_mask.ravel().astype(np.int32))
    largest_component = component_sizes[1:].argmax() + 1

    return (labeled_mask == largest_component).astype(np.float32)

def calculate_dice_coefficient(y_true, y_pred, num_classes, start_index=1):
    """Calculate Dice scores for each class and mean Dice."""
    dice_scores = {}
    eps = 1e-7  # To avoid division by zero
    eps = 0


    # Erode y_pred once with 3D structuring element
    structure = np.ones((2,2,2), dtype=bool)


    for class_idx in np.arange(start_index, start_index+num_classes):
        true_mask = (y_true == class_idx).astype(np.float32)
        pred_mask = (y_pred == class_idx).astype(np.float32)
        pred_mask = keep_largest_connected_component(pred_mask)

        #pred_mask = binary_erosion(pred_mask, structure=structure, iterations=1)

        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)

        dice = (2. * intersection) / (union)
        dice_scores[f'class_{class_idx}'] = dice


    dice_scores['mean_dice'] = np.mean(list(dice_scores.values()))
    return dice_scores


def load_nifti_file(file_path):
    """Load NIfTI file and return numpy array."""
    img = nib.load(file_path)
    return img.get_fdata()


def find_matching_pairs(labels_dir, preds_dir):
    """Find all matching label/prediction pairs across two directories."""
    label_files = sorted(glob.glob(os.path.join(labels_dir, 'LISA_*_HF_hipp.nii.gz')))
    if len(label_files) == 0:
        label_files = sorted(glob.glob(os.path.join(labels_dir, 'LISA_*_HF_baga.nii.gz')))

    pred_files = sorted(glob.glob(os.path.join(preds_dir, '*LISA_*_ciso.nii.gz')))

    if not label_files:
        raise ValueError(f"No label files found in {labels_dir} with pattern LISA_*_HF_hipp.nii.gz")
    if not pred_files:
        pred_files = sorted(glob.glob(os.path.join(preds_dir, '*LISAHF*segprediction.nii.gz')))
        if not pred_files:
            raise ValueError(f"No prediction files found in {preds_dir} with pattern LISA_*_ciso.nii.gz")

    # Create mapping of case IDs to file paths
    label_map = {}
    for lf in label_files:
        case_id = os.path.basename(lf).split('_')[1]
        label_map[case_id] = lf

    pred_map = {}
    for pf in pred_files:
        if 'segprediction' in os.path.basename(pf):
            case_id = os.path.basename(pf).split('segprediction')[0][-4:]
        else:
            case_id = os.path.basename(pf).split('_')[1]
        pred_map[case_id] = pf

    # Find intersection of cases
    common_cases = set(label_map.keys()) & set(pred_map.keys())
    if not common_cases:
        raise ValueError("No matching cases found between label and prediction directories")

    # Return matched pairs
    matched_pairs = []
    for case_id in sorted(common_cases):
        matched_pairs.append((case_id, label_map[case_id], pred_map[case_id]))

    return matched_pairs


def process_cases(matched_pairs, num_classes, start_index=1):
    """Process all matched label/prediction pairs."""
    all_results = []
    class_results = defaultdict(list)

    print(f"\nProcessing {len(matched_pairs)} matched cases...")

    for case_id, label_path, pred_path in matched_pairs:
        try:
            # Load files
            label = load_nifti_file(label_path)
            pred = load_nifti_file(pred_path)

            # Verify shapes match
            if label.shape != pred.shape:
                print(f"Shape mismatch for case {case_id}: label {label.shape} vs prediction {pred.shape}")
                continue

            # Calculate Dice scores
            dice_scores = calculate_dice_coefficient(label, pred, num_classes, start_index=start_index)

            # Store results
            result = {'case_id': case_id, **dice_scores}
            all_results.append(result)

            # Aggregate class scores
            for class_idx in np.arange(start_index, start_index+num_classes):
                class_results[class_idx].append(dice_scores[f'class_{class_idx}'])

        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            continue

    # Calculate summary statistics
    summary = {
        'cases_processed': len(all_results),
        'per_class_mean': {},
        'overall_mean': 0.0
    }

    class_means = []
    class_stds = []
    for class_idx in np.arange(start_index, start_index+num_classes):
        mean_score = np.mean(class_results[class_idx])
        class_stds.append(np.std(class_results[class_idx]))
        summary['per_class_mean'][f'class_{class_idx}'] = mean_score
        class_means.append(mean_score)


    summary['overall_mean'] = np.mean(class_means)
    summary['overall_std'] = np.mean(class_stds)


    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='Calculate multi-class Dice scores between matched label and prediction files in separate directories.')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory containing ground truth labels (LISA_*_HF_hipp.nii.gz)')
    parser.add_argument('--preds_dir', type=str, required=True, help='Directory containing predictions (LISA_*_ciso.nii.gz)')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes (including background)')
    parser.add_argument('--start_index', type=int, default=1, help='Start class index.')

    parser.add_argument('--output_csv', type=str, help='Optional CSV file to save results')

    args = parser.parse_args()

    # Find all matching pairs
    matched_pairs = find_matching_pairs(args.labels_dir, args.preds_dir)

    # Process all cases
    case_results, summary = process_cases(matched_pairs, args.num_classes, args.start_index)
    case_results = sorted(case_results, key=lambda x: x['mean_dice'], reverse=True)


    '''
    # Print individual case results
    print("\nIndividual Case Results:")
    for result in case_results:
        print(f"\nCase {result['case_id']}:")
        for class_idx in np.arange(args.start_index, args.start_index+args.num_classes):
            print(f"  Class {class_idx}: {result[f'class_{class_idx}']:.4f}")
        print(f"  Mean Dice: {result['mean_dice']:.4f}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Cases processed: {summary['cases_processed']}")
    '''
    for class_idx in np.arange(args.start_index, args.start_index+args.num_classes):
        print(f"Mean Class {class_idx}: {summary['per_class_mean'][f'class_{class_idx}']:.4f}")

    print(f"Overall Mean Dice: {summary['overall_mean']:.4f}")
    print(f"Overall Std Dice: {summary['overall_std']:.4f}")

    # Save to CSV if requested
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as csvfile:
            fieldnames = ['case_id'] + [f'class_{i}' for i in np.arange(args.start_index, args.start_index+args.num_classes)] + ['mean_dice']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in case_results:
                writer.writerow(result)

        print(f"\nResults saved to {args.output_csv}")


if __name__ == '__main__':
    main()

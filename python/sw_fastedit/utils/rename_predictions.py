import os
import re
import argparse

def rename_files(directory):
    for fname in os.listdir(directory):
        if fname.endswith('.nii.gz'):
            match = re.match(r"LISA_VALIDATION_(\d+)_ciso\.nii\.gz", fname)
            if match:
                case_id = match.group(1)
                new_name = f"LISAHF{case_id}segprediction.nii.gz"
                src = os.path.join(directory, fname)
                dst = os.path.join(directory, new_name)
                os.rename(src, dst)
                print(f"Renamed: {fname} -> {new_name}")
            else:
                print(f"Skipped (no match): {fname}")

def main():
    parser = argparse.ArgumentParser(description="Rename LISA validation files to LISAHF format.")
    parser.add_argument('--dir', required=True, help='Directory containing .nii.gz files to rename')
    args = parser.parse_args()

    rename_files(args.dir)

if __name__ == "__main__":
    main()

import os
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import typer
from typing_extensions import Annotated

def ensure_same_shape(reference_file: Path, predicted_file: Path):
    ref_img = nib.load(str(reference_file))
    pred_img = nib.load(str(predicted_file))

    if ref_img.shape != pred_img.shape:
        print(f"Resizing {predicted_file.name} from {pred_img.shape} to {ref_img.shape}")
        data = pred_img.get_fdata()
        resized_data = np.zeros(ref_img.shape)
        min_shape = tuple(min(r, p) for r, p in zip(ref_img.shape, pred_img.shape))
        resized_data[:min_shape[0], :min_shape[1], :min_shape[2]] = data[:min_shape[0], :min_shape[1], :min_shape[2]]
        new_img = nib.Nifti1Image(resized_data, affine=ref_img.affine)
        nib.save(new_img, str(predicted_file))
    else:
        print(f"Shape match confirmed for {predicted_file.name}")

def run_prediction(fold: int, model_type: str, temp_input: Path, temp_output: Path):
    model_path = f"models/best_model_{model_type}_{fold}.pt"
    cmd = (
        f'python train.py -i {temp_input} -o {temp_output} -x -1.0 -ta --dataset LISA '
        f'--non_interactive --dont_check_output_dir --target validation -a --network smalldynunet '
        f'--eval_only --save_pred --resume_from {model_path} --no_log --no_data'
    )
    os.system(cmd)

    pred_dir = temp_output / 'predictions'
    renamed_pred = temp_output / f'predictions_{model_type}_{fold}'
    if pred_dir.exists():
        if renamed_pred.exists():
            shutil.rmtree(renamed_pred)
        shutil.move(pred_dir, renamed_pred)

def main(
    input_dir: Annotated[str, typer.Option()] = "/input",
    output_dir: Annotated[str, typer.Option()] = "/output",
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_files = input_path.glob("*.nii.gz")

    for file in input_files:
        base_name = file.name
        print(f"\n🔁 Processing: {base_name}")

        # Create temp dirs
        temp_input = Path("./temp_input")
        temp_output = Path("./temp_output")
        shutil.rmtree(temp_input, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)
        temp_input.mkdir(parents=True, exist_ok=True)
        temp_output.mkdir(parents=True, exist_ok=True)

        # Copy input file
        temp_input_file = temp_input / base_name
        shutil.copy(file, temp_input_file)

        # Run predictions for ALL folds sequentially
        for fold in range(5):
            run_prediction(fold, 'all', temp_input, temp_output / f'predictions_all_{fold}')

        # Run predictions for HIPP folds sequentially
        for fold in range(5):
            run_prediction(fold, 'hipp', temp_input, temp_output / f'predictions_hipp_{fold}')

        # Ensemble step
        cmd = 'python sw_fastedit/utils/ensemble_folds.py --input_dirs'
        for fold in range(5):
            cmd += f' {temp_output}/predictions_all_{fold}'
            cmd += f' {temp_output}/predictions_hipp_{fold}'
        cmd += f' --output_dir {temp_output}/predictions_ensemble --mode majority --num_workers 1'
        os.system(cmd)

        # Harmonize step
        os.system(
            f'python sw_fastedit/utils/harmonize_labels_centerline.py '
            f'--input_dir {temp_output}/predictions_ensemble '
            f'--output_dir {temp_output}/predictions_final '
            f'--label1 1 --label2 2 --th 10'
        )

        # Move final predictions to output folder
        predictions_final_dir = temp_output / 'predictions_final'
        if predictions_final_dir.exists():
            for pred_file in predictions_final_dir.glob("*.nii.gz"):
                new_name = base_name.replace("CISO", "hipp").replace("ciso", "hipp").replace("TESTING", "TESTING_SEG")
                final_output = output_path / new_name
                #ensure_same_shape(file, pred_file)
                shutil.move(pred_file, final_output)
                print(f"✅ Saved: {final_output.name}")
        else:
            print(f"⚠️ No predictions found for {base_name}")

        # Clean up temporary folders
        shutil.rmtree(temp_input, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)

if __name__ == "__main__":
    typer.run(main)

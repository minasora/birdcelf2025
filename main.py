import pandas as pd
import os
import mlflow

from config import cfg # Use the global cfg instance
from utils.general_utils import set_seed
from training.trainer import run_training

def main():
    print(f"Using device: {cfg.device}")
    set_seed(cfg.seed)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\nLoading training data...")
    try:
        train_df_full = pd.read_csv(cfg.train_csv)
        # Ensure 'filename' and 'primary_label' columns exist, crucial for dataset and stratified split
        if 'filename' not in train_df_full.columns or 'primary_label' not in train_df_full.columns:
            raise ValueError("train.csv must contain 'filename' and 'primary_label' columns.")

        # Pre-generate samplename and filepath if not done by dataset (for consistency if needed outside dataset)
        if 'filepath' not in train_df_full.columns:
            train_df_full['filepath'] = cfg.train_datadir + '/' + train_df_full.filename
        if 'samplename' not in train_df_full.columns:
            train_df_full['samplename'] = train_df_full.filename.map(
                lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0] if isinstance(x, str) else 'unknown_sample'
            )

    except FileNotFoundError:
        print(f"Error: Training CSV '{cfg.train_csv}' not found. Please check the path in config.py.")
        return
    except ValueError as ve:
        print(f"Error with training data: {ve}")
        return

    try:
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df) # Set num_classes globally from taxonomy
        print(f"Number of classes set to: {cfg.num_classes} from {cfg.taxonomy_csv}")
    except FileNotFoundError:
        print(f"Error: Taxonomy CSV '{cfg.taxonomy_csv}' not found. num_classes might not be set correctly.")
        # The model or dataset might try to load it again, or training might fail if num_classes remains 0.
        # For robustness, it's good to ensure it's loaded here or fail early.
        if cfg.num_classes == 0: # If it's still 0, this is a critical issue.
            print("Critical Error: cfg.num_classes is 0 and taxonomy.csv is missing. Cannot proceed.")
            return


    print("\nStarting training...")
    print(f"LOAD_DATA (use pre-computed spectrograms) is set to: {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print(f"Attempting to use pre-computed mel spectrograms from: {cfg.spectrogram_npy}")
        if not os.path.exists(cfg.spectrogram_npy):
            print(f"Warning: Spectrogram NPY file not found at {cfg.spectrogram_npy}. Training might fail or switch to on-the-fly.")
    else:
        print("Will generate spectrograms on-the-fly during training (if not already processed by dataset).")

    # Set MLflow experiment
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    print(f"MLflow experiment set to: '{cfg.mlflow_experiment_name}'")

    # Run training
    mean_auc = run_training(train_df_full, cfg)
    print(f"\nTraining complete! Mean CV AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    main()
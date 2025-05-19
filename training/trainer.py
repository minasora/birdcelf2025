import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time
import gc
import os
import mlflow
import mlflow.pytorch

# from config import CFG # cfg will be passed as argument
from data.dataset import BirdCLEFDatasetFromNPY, collate_fn
from data.preprocessing import generate_spectrograms # If on-the-fly generation is needed for initial run
from models.bird_model import BirdCLEFModel
from utils.training_utils import get_optimizer, get_scheduler, get_criterion, calculate_auc

def train_one_epoch(model, loader, optimizer, criterion, device, cfg, scheduler=None): # Added cfg
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for step, batch in pbar:
        if not batch: # Handle empty batch from collate_fn
            print("Skipping empty batch in training.")
            continue


        inputs = batch['melspec'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()


        outputs_from_model = model(inputs, targets if model.mixup_enabled else None)

        if isinstance(outputs_from_model, tuple):
            logits, loss = outputs_from_model  # Assigns logits
        else:
            logits = outputs_from_model      # Assigns logits
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None and cfg.scheduler == 'OneCycleLR': # Step OneCycleLR per batch
            scheduler.step()

        all_outputs.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        losses.append(loss.item())

        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    return avg_loss, auc

def validate(model, loader, criterion, device, cfg): # Added cfg
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            if not batch: # Handle empty batch
                print("Skipping empty batch in validation.")
                continue

            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs) # No targets needed for eval unless model specifically uses them
            loss = criterion(outputs, targets)

            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            losses.append(loss.item())
            pbar.set_postfix({'val_loss': np.mean(losses[-10:]) if losses else 0})


    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    return avg_loss, auc

def run_training(df_full, cfg):
    if cfg.debug: # Ensure debug settings are applied if not already
        cfg.update_debug_settings()

    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Load taxonomy to set num_classes if not already set
    # This is also done in Dataset, but good to have it early for model init if needed
    if cfg.num_classes == 0:
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)
        print(f"run_training: Updated cfg.num_classes to {cfg.num_classes}")


    spectrograms = None
    if cfg.LOAD_DATA:
        print("Loading pre-computed mel spectrograms from NPY file...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms.")
        except Exception as e:
            print(f"Error loading pre-computed spectrograms: {e}. Will try to generate if needed or fail.")
            print("Consider setting LOAD_DATA=False or providing the correct path to spectrogram_npy.")
            cfg.LOAD_DATA = False # Fallback or raise error



    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    overall_best_auc = 0
    fold_results = []


    # Start MLflow parent run for the entire training process
    with mlflow.start_run(run_name="BirdCLEF_Training_Overall") as parent_run:
        mlflow.log_params(vars(cfg)) # Log all config parameters

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['primary_label'])):
            if fold not in cfg.selected_folds:
                continue

            print(f'\n{"="*30} Fold {fold} {"="*30}')
            mlflow.set_tag("current_fold", fold) # Tag for the parent run indicating current fold processing

            with mlflow.start_run(run_name=f"Fold_{fold}", nested=True) as child_run:
                mlflow.log_param("fold", fold)

                train_df = df_full.iloc[train_idx].reset_index(drop=True)
                val_df = df_full.iloc[val_idx].reset_index(drop=True)

                print(f'Training set: {len(train_df)} samples')
                print(f'Validation set: {len(val_df)} samples')
                mlflow.log_param("train_samples", len(train_df))
                mlflow.log_param("val_samples", len(val_df))

                # Pass the correct subset of spectrograms if they are pre-loaded
                # The dataset class will handle filtering based on the df passed.
                train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
                val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')

                # cfg.num_classes should now be set by the dataset if it wasn't before
                if cfg.num_classes == 0 and hasattr(train_dataset, 'num_classes'):
                    cfg.num_classes = train_dataset.num_classes
                    print(f"Fold {fold}: cfg.num_classes updated by dataset to {cfg.num_classes}")


                train_loader = DataLoader(
                    train_dataset, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn
                )

                model = BirdCLEFModel(cfg).to(cfg.device)
                # Store criterion instance in cfg if model needs it (e.g., for mixup)
                cfg.criterion_instance = get_criterion(cfg).to(cfg.device)


                optimizer = get_optimizer(model, cfg)

                scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader) if cfg.scheduler == 'OneCycleLR' else None)


                best_fold_auc = 0
                best_epoch_for_fold = 0
                model_save_path_fold = ""

                for epoch in range(cfg.epochs):
                    print(f"\nEpoch {epoch+1}/{cfg.epochs}")
                    epoch_start_time = time.time()

                    train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, cfg.criterion_instance, cfg.device, cfg, scheduler)
                    val_loss, val_auc = validate(model, val_loader, cfg.criterion_instance, cfg.device, cfg)

                    epoch_duration = time.time() - epoch_start_time

                    if scheduler is not None and cfg.scheduler != 'OneCycleLR': # OneCycleLR is stepped per batch
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()

                    print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} | Time: {epoch_duration:.2f}s")
                    mlflow.log_metrics({
                        f"fold_{fold}_train_loss": train_loss,
                        f"fold_{fold}_train_auc": train_auc,
                        f"fold_{fold}_val_loss": val_loss,
                        f"fold_{fold}_val_auc": val_auc,
                        f"fold_{fold}_lr": optimizer.param_groups[0]['lr']
                    }, step=epoch)


                    if val_auc > best_fold_auc:
                        best_fold_auc = val_auc
                        best_epoch_for_fold = epoch + 1
                        print(f"Fold {fold} - New best AUC: {best_fold_auc:.4f} at epoch {best_epoch_for_fold}")
                        model_save_path_fold = os.path.join(cfg.OUTPUT_DIR, f"model_fold{fold}_best.pth")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'epoch': epoch,
                            'val_auc': val_auc,
                            'cfg_dict': vars(cfg) # Save config for reference
                        }, model_save_path_fold)
                        mlflow.log_metric(f"fold_{fold}_best_val_auc", best_fold_auc, step=epoch) # Log best AUC for this fold when it updates


                print(f"\nBest AUC for fold {fold}: {best_fold_auc:.4f} at epoch {best_epoch_for_fold}")
                mlflow.log_metric(f"fold_final_best_auc", best_fold_auc) # Log final best AUC for this fold run
                if model_save_path_fold and os.path.exists(model_save_path_fold):
                     # Log the best model for this fold
                    mlflow.pytorch.log_model(pytorch_model=model, # you might need to load the state_dict into a fresh model instance
                                             artifact_path=f"model_fold_{fold}_best",
                                             conda_env = mlflow.pytorch.get_default_conda_env()) # or specify your own
                    # mlflow.log_artifact(model_save_path_fold, artifact_path=f"checkpoints_fold_{fold}")


                fold_results.append(best_fold_auc)
                if best_fold_auc > overall_best_auc:
                    overall_best_auc = best_fold_auc
                    # Optionally save the overall best model across folds (might be different from single fold best)
                    # torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_overall_best.pth"))
                    # mlflow.log_artifact(os.path.join(cfg.OUTPUT_DIR, "model_overall_best.pth"), "overall_best_model")

                del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                gc.collect()
                # End of child (fold) run

        # Back in parent run
        mean_cv_auc = np.mean(fold_results) if fold_results else 0
        print("\n" + "="*60)
        print("Cross-Validation Results:")
        for i, score in enumerate(fold_results):
            fold_num = cfg.selected_folds[i]
            print(f"Fold {fold_num}: {score:.4f}")
            mlflow.log_metric(f"cv_fold_{fold_num}_best_auc", score)
        print(f"Mean CV AUC: {mean_cv_auc:.4f}")
        mlflow.log_metric("mean_cv_auc", mean_cv_auc)
        print("="*60)

    return mean_cv_auc
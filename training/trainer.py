import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time
import gc
import os
from torch.utils.tensorboard import SummaryWriter

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


    # Set up TensorBoard writer for the entire training process
    os.makedirs(cfg.tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.tensorboard_log_dir)
    
    # Log all config parameters
    for key, value in vars(cfg).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            writer.add_text(f"config/{key}", str(value))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['primary_label'])):
        if fold not in cfg.selected_folds:
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')
        # Add current fold tag to tensorboard
        writer.add_text("current_fold", str(fold))
        
        # Log fold parameter
        writer.add_text(f"fold_{fold}/info", f"Fold {fold} training started")

        train_df = df_full.iloc[train_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)

        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        writer.add_text(f"fold_{fold}/train_samples", str(len(train_df)))
        writer.add_text(f"fold_{fold}/val_samples", str(len(val_df)))

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
            # Log metrics to TensorBoard
            writer.add_scalar(f"fold_{fold}/train_loss", train_loss, epoch)
            writer.add_scalar(f"fold_{fold}/train_auc", train_auc, epoch)
            writer.add_scalar(f"fold_{fold}/val_loss", val_loss, epoch)
            writer.add_scalar(f"fold_{fold}/val_auc", val_auc, epoch)
            writer.add_scalar(f"fold_{fold}/lr", optimizer.param_groups[0]['lr'], epoch)

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_epoch_for_fold = epoch + 1
                print(f"Fold {fold} - New best AUC: {best_fold_auc:.4f} at epoch {best_epoch_for_fold}")
                model_save_path_fold = os.path.join(cfg.OUTPUT_DIR, f"model_fold{fold}_best.pth")
                torch.save(model.state_dict(), model_save_path_fold)
                writer.add_scalar(f"fold_{fold}/best_val_auc", best_fold_auc, epoch) # Log best AUC for this fold when it updates

        print(f"\nBest AUC for fold {fold}: {best_fold_auc:.4f} at epoch {best_epoch_for_fold}")
        writer.add_scalar(f"fold_{fold}/final_best_auc", best_fold_auc, 0) # Log final best AUC for this fold run
        if model_save_path_fold and os.path.exists(model_save_path_fold):
            # Add model graph to TensorBoard
            dummy_input = torch.randn(1, cfg.in_channels, cfg.TARGET_SHAPE[0], cfg.TARGET_SHAPE[1], device=cfg.device)
            writer.add_graph(model, dummy_input)

        fold_results.append(best_fold_auc)
        if best_fold_auc > overall_best_auc:
            overall_best_auc = best_fold_auc

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
        writer.add_scalar(f"cv/fold_{fold_num}_best_auc", score, 0)
    print(f"Mean CV AUC: {mean_cv_auc:.4f}")
    writer.add_scalar("cv/mean_cv_auc", mean_cv_auc, 0)
    
    # Close the TensorBoard writer
    writer.close()
    print("="*60)

    return mean_cv_auc
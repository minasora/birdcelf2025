import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
    return optimizer

def get_scheduler(optimizer, cfg, steps_per_epoch=None): # Added steps_per_epoch for OneCycleLR
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for OneCycleLR scheduler")
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            pct_start=0.1
        )
    else:
        scheduler = None
    return scheduler

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
    return criterion

def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []
    probs = 1 / (1 + np.exp(-outputs)) # Sigmoid activation for outputs

    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0 and len(np.unique(targets[:, i])) > 1 : # Check for at least one positive instance and more than one class
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
            except ValueError as e:
                # print(f"Could not calculate AUC for class {i}: {e}") # Optional: for debugging
                pass # Skip if AUC cannot be computed (e.g. only one class present in targets)
    return np.mean(aucs) if aucs else 0.0
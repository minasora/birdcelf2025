import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import torchvision


import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        prob = torch.sigmoid(logits)
        ce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t  = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal = self.alpha * (1 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 0.6,
            focal_weight: float = 1.4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss







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
    """根据 cfg.criterion 来返回相应的损失函数实例。"""
    if cfg.criterion == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif cfg.criterion == 'FocalLoss':
        # 从 cfg 中读取超参（可在 CFG 里定义 focal_alpha, focal_gamma, reduction）
        alpha     = getattr(cfg, 'focal_alpha', 0.25)
        gamma     = getattr(cfg, 'focal_gamma', 2.0)
        reduction = getattr(cfg, 'focal_reduction', 'mean')
        return FocalLossBCE(alpha=alpha, gamma=gamma, reduction=reduction)
    elif cfg.criterion == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")


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
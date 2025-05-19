import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd # For reading taxonomy if num_classes not in cfg initially
import numpy as np # For np.random.beta
# from config import CFG # cfg will be passed as argument

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Ensure num_classes is set. It should be set by BirdCLEFDatasetFromNPY or main script.
        if self.cfg.num_classes == 0:
             # Fallback, though it should be set before model init
            try:
                taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
                self.cfg.num_classes = len(taxonomy_df)
                print(f"Model: num_classes dynamically set to {self.cfg.num_classes} from taxonomy.")
            except FileNotFoundError:
                 raise ValueError("taxonomy.csv not found and cfg.num_classes is not set.")


        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,      # Example, you might want to make these configurable in CFG
            drop_path_rate=0.2  # Example
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name or 'resnext' in cfg.model_name: # Added resnext
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        # Add other model families if needed, e.g., convnext
        elif 'convnext' in cfg.model_name:
            backbone_out = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        else: # Default for timm models that have a get_classifier method
            try:
                backbone_out = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, '') # Resets to identity
            except AttributeError:
                # Fallback for models where get_classifier().in_features might not directly apply
                # This might need specific handling based on the model architecture
                # For many models, num_features attribute exists
                if hasattr(self.backbone, 'num_features'):
                    backbone_out = self.backbone.num_features
                else: # Last resort, try to infer, may fail for some architectures
                    dummy_input = torch.randn(1, cfg.in_channels, *cfg.TARGET_SHAPE)
                    dummy_output = self.backbone(dummy_input)
                    if isinstance(dummy_output, tuple): # some models return multiple outputs
                        dummy_output = dummy_output[0]
                    if dummy_output.ndim == 4: # If it's [B, C, H, W]
                        backbone_out = dummy_output.shape[1] # Assuming global pooling will happen later
                    else: # If it's [B, Features]
                         backbone_out = dummy_output.shape[-1]
                # For these cases, ensure the classifier part of the backbone is removed or made an identity op
                # This part is tricky without knowing all possible timm model structures.
                # A common pattern is to replace the final layer with nn.Identity()
                # self.backbone.classifier = nn.Identity() # or self.backbone.fc, self.backbone.head etc.
                print(f"Warning: Automatic feature dim deduction for {cfg.model_name}. Final layer might need manual handling.")


        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        # Handle models that return features in a dict (e.g. some transformers)
        if isinstance(features, dict):
            # Try to get the 'features' key, or the last hidden state for transformer-like models
            features = features.get('features', features.get('last_hidden_state', features))


        if len(features.shape) == 4: # If output is [B, C, H, W]
            features = self.pooling(features)
            features = features.view(features.size(0), -1) # Flatten

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            # Ensure criterion is suitable for mixup, BCEWithLogitsLoss works well
            loss_func = self.cfg.criterion_instance if hasattr(self.cfg, 'criterion_instance') else F.binary_cross_entropy_with_logits
            loss = self.mixup_criterion(loss_func, logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
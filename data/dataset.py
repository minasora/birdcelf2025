import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

from config import CFG
# from config import CFG # cfg will be passed as argument
from .preprocessing import process_audio_file # Relative import

from .augment import SpectrogramAugment 


class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.cfg.num_classes = self.num_classes # Update CFG
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}
        if mode == "train":
            self.transforms = SpectrogramAugment(
                image_size=cfg.TARGET_SHAPE[1],
                randaugment_n=2, randaugment_m=9,
                time_mask_param=40, freq_mask_param=12,
                erase_prob=0.7
            )
        elif mode == "val":
            # 仅缩放+归一化
            self.transforms = SpectrogramAugment(
                image_size=cfg.TARGET_SHAPE[1],
                randaugment_n=0, randaugment_m=0,   # 关闭 RandAug
                erase_prob=0.0,                     # 关闭擦除
                time_mask_param=0, freq_mask_param=0
            )
        else:
            self.transforms = None

        if 'filepath' not in self.df.columns: # Ensure filepath exists
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        if 'samplename' not in self.df.columns: # Ensure samplename exists
             self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0] if isinstance(x, str) else 'unknown_sample')


        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples (total unique in df: {len(sample_names)})")
        elif not self.cfg.LOAD_DATA:
             print(f"No pre-loaded spectrograms for {mode}. Will generate on-the-fly if needed.")


        if cfg.debug and mode == "train": # In debug mode, sample only from training set
            sample_size = min(1000, len(self.df))
            print(f"Debug mode: Sampling {sample_size} from {mode} dataset.")
            self.df = self.df.sample(n=sample_size, random_state=cfg.seed).reset_index(drop=True)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA: # Generate on-the-fly if not loading from NPY
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None: # Fallback if still None
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            # if self.mode == "train": # Only print warning during training
            # print(f"Warning: Spectrogram for {samplename} (file: {row.get('filepath', 'N/A')}) not found/generated. Using zeros.")


        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

        if self.transforms is not None:
            spec = self.transforms(spec)

        target = self.encode_label(row['primary_label'])

        if 'secondary_labels' in row and pd.notna(row['secondary_labels']) and row['secondary_labels'] not in [[''], None]:
            secondary_labels_str = row['secondary_labels']
            try:
                # Handles cases like "['label1', 'label2']" or "label1 label2" (less common but good to be robust)
                if isinstance(secondary_labels_str, str) and secondary_labels_str.startswith('[') and secondary_labels_str.endswith(']'):
                    secondary_labels = eval(secondary_labels_str)
                elif isinstance(secondary_labels_str, str): # Simple space separated string, split it
                    secondary_labels = secondary_labels_str.split(' ')
                elif isinstance(secondary_labels_str, list):
                     secondary_labels = secondary_labels_str
                else:
                    secondary_labels = []

                for label in secondary_labels:
                    if label in self.label_to_idx:
                        target[self.label_to_idx[label]] = 1.0
            except Exception as e:
                # print(f"Warning: Could not parse secondary_labels '{secondary_labels_str}' for {samplename}: {e}")
                pass


        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        # Time masking
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                if spec.shape[2] > width : # Ensure width is not larger than spectrogram time dimension
                    start = random.randint(0, spec.shape[2] - width)
                    spec[0, :, start:start+width] = 0
        # Frequency masking
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                if spec.shape[1] > height: # Ensure height is not larger than spectrogram freq dimension
                    start = random.randint(0, spec.shape[1] - height)
                    spec[0, start:start+height, :] = 0
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)
        return spec

    def encode_label(self, label):
        target = np.zeros(self.num_classes, dtype=np.float32)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    batch = [item for item in batch if item is not None and item['melspec'] is not None]
    if len(batch) == 0:
        return {} # Return empty dict if batch is empty after filtering

    result = {key: [] for key in batch[0].keys()}

    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            # Ensure all spectrograms in the batch are stacked correctly.
            # This was handled by resizing in process_audio_file. If shapes mismatch, stacking will fail.
            try:
                result[key] = torch.stack(result[key])
            except RuntimeError as e:
                print(f"Error stacking melspecs in collate_fn: {e}")
                # Option: pad to max shape in batch or return as list (would require model to handle list)
                # For now, we rely on consistent shapes from preprocessing.
                # If this occurs, it means some spectrograms were not processed to TARGET_SHAPE.
                # As a fallback, skip this batch item or the whole batch.
                # For simplicity here, we'll assume this won't happen due to preprocessing.
                # If it does, the training loop will likely fail.
                pass # Or raise error
    return result
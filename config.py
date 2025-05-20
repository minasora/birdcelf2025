import torch

class CFG:
    # General
    seed = 42
    debug = False  # Set to False for full training
    apex = False
    print_freq = 100
    num_workers = 2 # os.cpu_count() can be used

    # Paths - Adjust these as per your environment if not running on Kaggle
    OUTPUT_DIR = './output/' # Changed from /kaggle/working/
    train_datadir = 'birdclef-2025/train_audio'
    train_csv = 'birdclef-2025/train.csv'       
    test_soundscapes = 'birdclef-2025/test_soundscapes'
    submission_csv = 'birdclef-2025/sample_submission.csv'
    taxonomy_csv = 'birdclef-2025/taxonomy.csv' 

    spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'

    # Model
    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1
    num_classes = 0 # Will be updated based on taxonomy.csv

    # Audio Processing
    LOAD_DATA = False  # Set to True to load pre-computed spectrograms
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 256
    FMIN = 20
    FMAX = 16000

    # Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    batch_size = 64
    criterion         = 'FocalLoss'    # 或 'BCEWithLogitsLoss'
    focal_alpha       = 0.25
    focal_gamma       = 2.0
    focal_reduction   = 'mean'         # 可选 'mean' 或 'sum'

    # Cross-validation
    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4] # Or [0] for debug

    # Optimizer
    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5

    # Scheduler
    scheduler = 'CosineAnnealingLR' # Options: 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'OneCycleLR'
    min_lr = 1e-6
    T_max = epochs # For CosineAnnealingLR

    # Augmentations
    aug_prob = 0.5
    mixup_alpha = 0# Set to 0 to disable mixup

    # TensorBoard
    tensorboard_log_dir = "./runs/BirdCLEF2025_Training"

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]
            self.LOAD_DATA = False # Ensure we can test on-the-fly generation in debug
            print("DEBUG mode enabled: epochs set to 2, selected_folds to [0].")

cfg = CFG()
if cfg.debug:
    cfg.update_debug_settings()
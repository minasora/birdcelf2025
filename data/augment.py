# aug.py
import torch
import torchvision.transforms.v2 as T   # TorchVision ≥0.18 推荐用 transforms.v2
import torchaudio.transforms as AT

class SpectrogramAugment:
    """
    把 mel 视为单通道图像做 RandAugment + RandomErasing，
    然后再做 Time/Freq Masking（SpecAugment）。
    """
    def __init__(
        self,
        image_size: int,
        randaugment_n: int = 2,
        randaugment_m: int = 9,
        time_mask_param: int = 40,
        freq_mask_param: int = 12,
        erase_prob: float = 0.5,
    ):
        self.transforms_img = T.Compose([
            T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m),
            # RandAugment 要求输入已是 tensor 且值域 0-1
            T.RandomErasing(p=erase_prob, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])

        self.transforms_spec = torch.nn.Sequential(
            AT.TimeMasking(time_mask_param=time_mask_param),
            AT.FrequencyMasking(freq_mask_param=freq_mask_param),
        )

        self.resize = T.Resize((image_size, image_size), antialias=True)
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])   # 0-1 → -1-1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor[C,H,W]，值域 0-1，dtype float32
        """
        # 图像级增强 (RandAug / Erasing)
        x = self.resize(x)           # TorchVision 的 RandAug 期望较大分辨率
        x = self.transforms_img(x)

        # 转成 [B=1, F, T] 形状给 torchaudio
        x_bt = x.squeeze(0).unsqueeze(0)    # [1, H, W]

        x_bt = self.transforms_spec(x_bt)   # Time & Freq Masking

        # 还原回 [C,H,W]
        x = x_bt.squeeze(0).unsqueeze(0)

        # 归一化到 -1~1
        x = self.normalize(x)
        return x.clamp(-1, 1)

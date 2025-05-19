import os
import argparse
import numpy as np
import pandas as pd
import librosa
import cv2
import math
import time
from tqdm.auto import tqdm

# 假设你的 CFG 定义在 config.py 中
from config import CFG

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        # 短音频循环拼接
        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            audio_data = np.tile(audio_data, n_copy)

        # 中心截取
        start_idx = max(0, len(audio_data)//2 - target_samples//2)
        end_idx = start_idx + target_samples
        center_audio = audio_data[start_idx:end_idx]

        # 不足长度时补零
        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio,
                                  (0, target_samples - len(center_audio)),
                                  mode='constant')

        mel_spec = audio2melspec(center_audio, cfg)

        # 调整到目标形状
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg, save_npy: bool=False, save_path: str=None):
    """
    遍历 DataFrame，生成 mel spec dict。
    如果 save_npy=True 且给定了 save_path，则把整个 dict 存为单个 .npy 文件。
    """
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug and i >= 1000:
            break

        samplename = row["samplename"]
        filepath   = row["filepath"]
        mel_spec = process_audio_file(filepath, cfg)
        if mel_spec is not None:
            all_bird_data[samplename] = mel_spec
        else:
            errors.append(filepath)

    # —— 保存整个 dict 到一个 .npy
    if save_npy and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, all_bird_data, allow_pickle=True)
        print(f"Saved all spectrograms to {save_path}")

    end_time = time.time()
    print(f"Done in {end_time - start_time:.2f}s — success {len(all_bird_data)}, fail {len(errors)}")
    return all_bird_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate mel spectrograms and save to a single .npy file"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="输入的 CSV 文件路径，需包含 samplename 和 filepath 列"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="输出 .npy 文件的完整路径"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式（仅处理前 1000 条）"
    )
    args = parser.parse_args()

    # 设置调试模式
    CFG.debug = args.debug

    # 读取 DataFrame
    df = pd.read_csv(args.csv)

    # 生成并保存
    generate_spectrograms(
        df,
        CFG,
        save_npy=True,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()

"""WAV 文件读取与基础元数据提取。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class WavData:
    path: Path
    fs: int              # 采样率 Hz
    n_samples: int        # 总采样点数
    n_channels: int
    duration_s: float
    subtype: str          # 原始位深/编码，如 PCM_16
    signal: np.ndarray     # 单通道信号，float64，已按 config.channel 处理


def load_wav(path: str | Path, channel: str = "mean") -> WavData:
    """读取 wav 文件，返回单通道信号 + 元数据。

    channel: 'mean' 取所有声道均值；'0'/'1'/... 指定声道号。
    """
    path = Path(path)
    data, fs = sf.read(str(path), always_2d=True)  # shape: (n_samples, n_channels)
    n_samples, n_channels = data.shape

    if channel == "mean":
        sig = data.mean(axis=1)
    else:
        idx = int(channel)
        if idx >= n_channels:
            raise ValueError(
                f"{path.name}: 请求声道 {idx}，但文件只有 {n_channels} 个声道"
            )
        sig = data[:, idx]

    info = sf.info(str(path))
    return WavData(
        path=path,
        fs=fs,
        n_samples=n_samples,
        n_channels=n_channels,
        duration_s=n_samples / fs,
        subtype=info.subtype,
        signal=sig.astype(np.float64),
    )

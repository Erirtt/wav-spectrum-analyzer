"""STFT 时频分析：信号 -> (频率轴, 时间轴, dB幅值矩阵)。"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

from .config import AnalysisConfig


@dataclass
class Spectrogram:
    freqs: np.ndarray     # (F,) Hz，已按 fmin/fmax 裁剪
    times: np.ndarray     # (T,) s
    db: np.ndarray        # (F, T) dB 幅值，freqs 为行、times 为列
    fs: int
    n_fft: int


def compute_stft(sig: np.ndarray, fs: int, cfg: AnalysisConfig, fmax_override: float | None = None) -> Spectrogram:
    """对信号做预处理 + STFT，返回 dB 表示的频谱矩阵。

    fmax_override: 覆盖 cfg.fmax 的裁剪上限。检测流程用它取一个比显示范围略宽
    的频段（如 +5%），避免恰好落在 fmax 边界上的峰因右侧坡被裁掉而漏检。
    """
    x = sig.copy()
    if cfg.remove_dc:
        x = x - np.mean(x)  # 去直流，否则 0Hz 附近出现巨大假峰

    n_fft = cfg.n_fft
    if n_fft > len(x):
        raise ValueError(
            f"n_fft={n_fft} 大于信号长度={len(x)}，请调小 n_fft 或检查文件时长"
        )
    noverlap = int(n_fft * cfg.overlap)

    freqs, times, Zxx = sp_signal.stft(
        x,
        fs=fs,
        window=cfg.window,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=True,  # 末尾补零到整帧：文件最后 <1 hop 的样本不会被静默丢弃
    )

    mag = np.abs(Zxx)
    # 单边谱补偿：rfft 只保留正频率一半，实信号的能量本应在正负频率对称分布，
    # 这里丢弃了负频率那一半，因此非 DC/Nyquist 的 bin 需要 ×2（+6dB）才能让
    # 满量程正弦波读数对齐到 0dBFS。不补偿的话，所有电平会系统性偏低 6dB，
    # 频谱看起来比实际更“暗”（本底/动态范围显得被夸大）。
    mag[1:-1, :] *= 2.0

    # 幅值 -> dB，加 floor 避免 log(0)；同时把过小的值夹到 floor，保证色带/检测不受极小值干扰
    floor_lin = 10 ** (cfg.floor_db / 20.0)
    mag = np.maximum(mag, floor_lin)
    db = 20.0 * np.log10(mag)

    # 按 fmin/fmax 裁剪频率轴
    fmin = cfg.fmin if cfg.fmin is not None else 0.0
    cfg_fmax = fmax_override if fmax_override is not None else cfg.fmax
    fmax = cfg_fmax if cfg_fmax is not None else fs / 2.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    db = db[mask, :]

    return Spectrogram(freqs=freqs, times=times, db=db, fs=fs, n_fft=n_fft)


def mean_spectrum(spec: Spectrogram, method: str = "median") -> np.ndarray:
    """沿时间轴压缩成一条“平均谱”，用于峰值检测。

    method='median' 对瞬态更鲁棒（不会被短暂冲击拉高整体基线）。
    """
    if method == "median":
        return np.median(spec.db, axis=1)
    return np.mean(spec.db, axis=1)

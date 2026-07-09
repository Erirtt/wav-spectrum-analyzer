#!/usr/bin/env python3
"""生成合成测试 wav（用于验证分析链路，非真实传感器数据）。"""
import numpy as np
import soundfile as sf
from pathlib import Path

fs = 48000
duration = 6.0
t = np.arange(int(fs * duration)) / fs
rng = np.random.default_rng(42)

out_dir = Path("samples")
out_dir.mkdir(exist_ok=True)

# 阶段1: 正常件 —— 只有底噪，无明显波峰
sig1 = 0.02 * rng.standard_normal(len(t))
sf.write(out_dir / "stage1_ok.wav", sig1.astype(np.float32), fs)

# 阶段2: 疑似/异常波峰 —— 底噪 + 一个持续的 3200Hz 尖峰（贯穿全程，类似轴承故障特征频率）
sig2 = 0.02 * rng.standard_normal(len(t))
sig2 += 0.35 * np.sin(2 * np.pi * 3200 * t)
sf.write(out_dir / "stage2_peak.wav", sig2.astype(np.float32), fs)

# 阶段3: 某频段能量偏高 —— 底噪 + 800~1500Hz 宽带噪声能量抬升 + 一个瞬态冲击（1.5s处短促脉冲）
sig3 = 0.02 * rng.standard_normal(len(t))
from scipy.signal import butter, filtfilt
b, a = butter(4, [800, 1500], btype="bandpass", fs=fs)
band_noise = filtfilt(b, a, rng.standard_normal(len(t)))
sig3 += 0.25 * band_noise / np.max(np.abs(band_noise))
impulse_idx = int(1.5 * fs)
sig3[impulse_idx:impulse_idx + 30] += 0.6 * np.hanning(30)
sf.write(out_dir / "stage3_band.wav", sig3.astype(np.float32), fs)

print("已生成测试文件:")
for f in sorted(out_dir.glob("*.wav")):
    print(" -", f)

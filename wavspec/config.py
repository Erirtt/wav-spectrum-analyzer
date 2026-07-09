"""分析参数配置。全部有默认值，可通过命令行覆盖。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisConfig:
    # ---- STFT 参数 ----
    n_fft: int = 4096              # FFT 长度：越大频率分辨率越高、时间分辨率越低
    overlap: float = 0.75          # 帧重叠比例 (0~1)
    window: str = "hann"           # 窗函数，抑制频谱泄漏

    # ---- 频率范围（关心的频段）----
    fmin: Optional[float] = None   # 下限 Hz，None=0
    fmax: Optional[float] = None   # 上限 Hz，None=Nyquist(fs/2)

    # ---- 预处理 ----
    remove_dc: bool = True         # 去直流，避免 0Hz 假峰
    channel: str = "mean"          # 多通道处理：'mean' 或 '0'/'1'... 指定通道号

    # ---- 检测阈值（绝对突出度启发式；有 OK 基准后可换成基线对比）----
    peak_prominence_db: float = 8.0    # 峰突出度阈值：超过才算“峰”
    strong_prominence_db: float = 15.0 # 强峰阈值：超过判为“异常”而非“疑似”
    peak_min_sep_hz: float = 50.0      # 相邻峰最小间隔，避免一个峰拆成多个
    band_count: int = 8                # 频段能量分析的频带数
    band_hot_db: float = 10.0          # 某频带能量高出中位频带多少 dB 算“偏高”
    band_strong_db: float = 15.0       # 高出多少判为“异常”
    top_peaks: int = 8                 # 结论里最多列几个峰

    # ---- 渲染 ----
    colorscale: str = "Plasma"     # 3D 曲面色带
    floor_db: float = -120.0       # dB 下限，避免 log(0) 和拉低色带
    max_grid_freq: int = 400       # 3D 渲染时频率轴最多点数（抽稀，检测仍用全分辨率）
    max_grid_time: int = 400       # 3D 渲染时时间轴最多点数
    export_png: bool = False       # 是否额外导出 PNG 截图（需要 kaleido，较慢）；默认只生成交互式 HTML
    annotate: bool = False         # 是否在图上叠加异常判定标注（峰值线/图例/结论文字）；默认干净视图
    freq_scale: str = "linear"     # 频率轴刻度：'linear' 或 'log'（log 更利于观察低频故障特征）
    relative_db: bool = False      # True: 显示相对该文件自身峰值归一化的dB(峰值=0dB)；False: 绝对dBFS

    # ---- 3D 显示平滑（仅影响 3D 曲面外观，检测和 2D 复核图仍用全分辨率数据）----
    smooth_display: bool = True    # True: 按频带/时间块做能量域平均聚合（平滑）；False: 原始抽稀（毛躁但原汁原味）
    display_freq_bins: int = 160   # 聚合后频率方向的显示带数（参考: mel 谱常用 128）
    display_time_bins: int = 120   # 聚合后时间方向的显示块数
    smooth_sigma: float = 1.0      # 聚合后对显示网格再做轻度高斯平滑的σ，0=关闭

    # ---- 3D 场景轴比例（“平铺地毯”式布局：频率轴拉长、幅值轴压扁，脊线更易读）----
    aspect_freq: float = 2.0       # 频率轴相对长度
    aspect_time: float = 1.1       # 时间轴相对长度
    aspect_db: float = 0.45        # 幅值轴相对高度（越小越平铺）

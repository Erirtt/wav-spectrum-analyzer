"""3D 瀑布图 + 2D 辅助图渲染。

3D 曲面: x=频率(Hz) y=时间(s) z=幅值(dB)。
提供 俯视/正面/侧面/斜上 四个相机视角预设，交互式 HTML 中仍可自由旋转缩放。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

from .config import AnalysisConfig
from .dsp import Spectrogram, mean_spectrum
from .detect import DetectionResult
from .io_utils import WavData

# 相机视角预设（Plotly scene.camera.eye，单位为场景坐标的相对方向）。
# 注意：正交投影(orthographic)在浏览器 WebGL 中通过 relayout 动态切换时，
# Plotly.js 存在渲染 bug（曲面消失、刻度错乱），因此统一使用透视投影 + 较远
# 的观察距离来 近似 得到干净的俯视/正面/侧面效果。3D 图仅用于直觉展示，
# 精确读数以 2D 热力图/平均谱为准（见 build_2d_figure）。
CAMERA_PRESETS = {
    "斜上(默认)": dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
    "俯视": dict(eye=dict(x=0.0001, y=0.0, z=3.6), up=dict(x=0, y=1, z=0)),
    "正面(沿时间轴看频率-幅值)": dict(eye=dict(x=0.0, y=-3.6, z=0.0001)),
    "侧面(沿频率轴看时间-幅值)": dict(eye=dict(x=3.6, y=0.0, z=0.0001)),
}

# 暗色主题配色
BG_COLOR = "#0d0e14"
PANEL_COLOR = "#12131c"
GRID_COLOR = "#333747"
TEXT_COLOR = "#e6e6ef"
MUTED_TEXT = "#9a9db0"

DARK_SCENE_AXIS = dict(
    backgroundcolor=PANEL_COLOR,
    gridcolor=GRID_COLOR,
    zerolinecolor=GRID_COLOR,
    showbackground=True,
    color=TEXT_COLOR,
)


def _hz_to_mel(f):
    """Hz -> Mel（O'Shaughnessy 公式，librosa 同款 htk 风格）。"""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def _blockmax_downsample(
    freqs: np.ndarray, times: np.ndarray, db: np.ndarray, max_f: int, max_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保峰降采样：每个显示块取块内最大 dB（峰值保持，类似频谱仪的 peak-hold）。

    单纯抽稀(每隔N取1)可能恰好跳过峰顶 bin，让峰在图上损失几 dB；
    块内取最大保证任何峰的顶点都会被显示出来，代价是本底噪声略被抬高
    (块内最大≥块内均值)——用于"未平滑"检视模式，这个取舍是合适的。
    """
    nf = min(max_f, len(freqs))
    nt = min(max_t, len(times))
    f_edges = np.linspace(0, len(freqs), nf + 1).astype(int)
    t_edges = np.linspace(0, len(times), nt + 1).astype(int)
    # reduceat 要求严格递增的起点；min() 保证了每块至少 1 个元素
    db_f = np.maximum.reduceat(db, f_edges[:-1], axis=0)
    db_ft = np.maximum.reduceat(db_f, t_edges[:-1], axis=1)
    freqs_d = np.array([freqs[f_edges[i]:f_edges[i + 1]].mean() for i in range(nf)])
    times_d = np.array([times[t_edges[i]:t_edges[i + 1]].mean() for i in range(nt)])
    return freqs_d, times_d, db_ft


def _aggregate_for_display(
    freqs: np.ndarray, times: np.ndarray, db: np.ndarray, cfg: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把全分辨率 STFT 网格聚合成粗显示网格（线性功率域块平均，能量守恒）。

    抽稀(每隔N取1)会随机采到噪声尖刺，曲面显得毛躁；块平均把每个显示单元内
    所有 bin 的能量平均，既平滑又保持局部能量真实——这正是参考图(粗网格~128带)
    平滑外观的来源。仅用于 3D 显示，检测和 2D 复核图仍使用全分辨率数据。
    """
    power = 10.0 ** (db / 10.0)  # db=20log10(mag) -> 功率域 mag²=10^(db/10)

    # 时间方向：均匀分块平均
    nt = min(cfg.display_time_bins, len(times))
    t_edges = np.linspace(0, len(times), nt + 1).astype(int)
    p_t = np.empty((power.shape[0], nt))
    times_d = np.empty(nt)
    for j in range(nt):
        lo, hi = t_edges[j], max(t_edges[j + 1], t_edges[j] + 1)
        p_t[:, j] = power[:, lo:hi].mean(axis=1)
        times_d[j] = times[lo:hi].mean()

    # 频率方向分带（与显示轴刻度一致）：
    #   linear 均匀分带 / log 等比分带 / mel 梅尔等间隔分带（即梅尔频谱的三角滤波器组中心思路）
    nf = min(cfg.display_freq_bins, len(freqs))
    if cfg.freq_scale == "log":
        f_edges = np.geomspace(freqs[0], freqs[-1], nf + 1)
        centers = np.sqrt(f_edges[:-1] * f_edges[1:])  # 几何中心
    elif cfg.freq_scale == "mel":
        mel_edges = np.linspace(_hz_to_mel(freqs[0]), _hz_to_mel(freqs[-1]), nf + 1)
        f_edges = _mel_to_hz(mel_edges)
        centers = _mel_to_hz((mel_edges[:-1] + mel_edges[1:]) / 2.0)  # 梅尔域中心
    else:
        f_edges = np.linspace(freqs[0], freqs[-1], nf + 1)
        centers = (f_edges[:-1] + f_edges[1:]) / 2.0

    band_idx = np.clip(np.searchsorted(f_edges, freqs, side="right") - 1, 0, nf - 1)
    p_ft = np.zeros((nf, nt))
    counts = np.bincount(band_idx, minlength=nf).astype(float)
    np.add.at(p_ft, band_idx, p_t)
    empty = counts == 0
    p_ft[~empty] /= counts[~empty, None]
    if empty.any():
        # log 分带在低频端可能窄于一个 bin 而落空：用最近的原始 bin 填充
        nearest = np.searchsorted(freqs, centers[empty])
        nearest = np.clip(nearest, 0, len(freqs) - 1)
        p_ft[empty] = p_t[nearest]

    # 高斯平滑必须在【功率域】做（转 dB 之前）：
    # dB 是对数域，在 dB 上做加权平均等价于线性域的几何平均——窄带峰会被周围
    # 接近 floor(-120dB) 的值拉低几十 dB（实测满幅正弦 0dBFS 被压到 -49dBFS），
    # 这是数学伪影不是物理平滑。功率域卷积则是能量守恒的扩散：峰适度展宽、
    # 峰顶只降几 dB（能量转移到相邻显示单元），物理意义等同于加宽分析带宽。
    if cfg.smooth_sigma > 0:
        p_ft = gaussian_filter(p_ft, sigma=cfg.smooth_sigma)

    floor_p = 10.0 ** (cfg.floor_db / 10.0)
    db_d = 10.0 * np.log10(np.maximum(p_ft, floor_p))
    return centers, times_d, db_d


def _robust_color_range(db: np.ndarray) -> tuple[float, float]:
    """按分位数而非绝对 min/max 设定色带范围。

    极少数 bin 会被数值下限(floor_db)夹住，若用真实 min/max 设色带，
    这些离群点会把色带拉得很宽，压缩真正有信息量的区间的对比度，
    视觉上显得动态范围被“夸大”。用 1%~100% 分位数更贴近实际数据分布。
    """
    lo = float(np.percentile(db, 1))
    hi = float(np.max(db))
    return lo, hi


def _compute_shift(spec: Spectrogram, cfg: AnalysisConfig) -> float:
    """按 cfg.db_mode 决定图表 dB 的显示基准（平移量）。

    检测逻辑(detect.py)永远在原始绝对 dBFS 上进行，不受这里影响；
    这里的平移只改变图表怎么显示同一份数据，不改变数据本身。

    - 'noise_floor': 减去本底噪声(默认第10分位数)，本底≈0dB，异常峰显示为正数，
      类似"信号比本底噪声高多少"(SNR)，最直观，默认模式。
    - 'peak': 减去该文件自身最大值，峰值=0dB，其余为负。
    - 'dbfs': 不平移，绝对满量程基准，行业标准但大多数读数是负数。

    注意：平移必须在功率域聚合/平滑(_aggregate_for_display)【之后】应用在最终
    显示值上，而不是提前平移原始 dB 再聚合——提前平移会让聚合函数内部的绝对
    floor_db 数值安全下限失去意义（比较的量纲不对，虽不会报错但逻辑不干净）。
    """
    if cfg.db_mode == "peak":
        return float(spec.db.max())
    elif cfg.db_mode == "dbfs":
        return 0.0
    else:  # noise_floor (默认)
        return float(np.percentile(spec.db, cfg.noise_floor_percentile))


def _freq_mask(freqs: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """log 轴不能显示 0Hz 必须过滤；mel(0)=0 虽合法，但 DC bin 本身也没什么
    诊断意义（前面已去直流），过滤掉能让梅尔轴从第一个真实频率 bin 开始，
    和参考图表现一致。"""
    if cfg.freq_scale in ("log", "mel"):
        return freqs > 0
    return np.ones_like(freqs, dtype=bool)


def _x_coords(freqs_hz: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """频率 -> 图表 x 坐标。mel 模式画在梅尔域（等间隔），其余直接用 Hz。"""
    if cfg.freq_scale == "mel":
        return _hz_to_mel(freqs_hz)
    return freqs_hz


def _format_hz_label(hz: float) -> str:
    """1400 -> '1.4k'，10000 -> '10k'，525 -> '525'。"""
    if hz >= 1000:
        s = f"{hz / 1000:.1f}"
        if s.endswith(".0"):
            s = s[:-2]
        return f"{s}k"
    return f"{round(hz)}"


def _freq_axis_opts(cfg: AnalysisConfig, f_lo: float, f_hi: float, n_ticks: int = 6) -> dict:
    """频率轴配置。Plotly 无原生 mel 轴：mel 模式下 x 坐标是梅尔值。

    刻度在梅尔域【等间隔】布置（而不是选好看的整数Hz再换算位置），换回 Hz 做
    标签——这样刻度线间距在视觉上是真正均匀的，代价是标签变成非整数(如1.4k、
    2.9k)。这是梅尔轴的标准呈现方式：每一格代表人耳感知上相等的"一步"。
    """
    if cfg.freq_scale == "mel":
        mel_ticks = np.linspace(_hz_to_mel(f_lo), _hz_to_mel(f_hi), n_ticks)
        hz_ticks = _mel_to_hz(mel_ticks)
        labels = [_format_hz_label(hz) for hz in hz_ticks]
        return dict(
            title="频率 Frequency (Hz, Mel刻度)",
            type="linear",
            tickvals=mel_ticks.tolist(),
            ticktext=labels,
        )
    return dict(title="频率 Frequency (Hz)", type=cfg.freq_scale)


_DB_MODE_LABEL = {
    "noise_floor": "dB, 相对本底噪声",
    "peak": "dB, 相对峰值",
    "dbfs": "dB",
}


def _colorbar_title(cfg: AnalysisConfig) -> str:
    return _DB_MODE_LABEL[cfg.db_mode]


def _db_axis_title(cfg: AnalysisConfig) -> str:
    return f"幅值 Amplitude ({_DB_MODE_LABEL[cfg.db_mode]})"


def build_3d_figure(wav: WavData, spec: Spectrogram, det: DetectionResult, cfg: AnalysisConfig) -> go.Figure:
    shift = _compute_shift(spec, cfg)
    fmask = _freq_mask(spec.freqs, cfg)
    freqs_masked = spec.freqs[fmask]
    db_masked = spec.db[fmask, :]  # 绝对dB，聚合/平滑安全；平移放最后应用

    if cfg.smooth_display:
        # 能量域块平均聚合：平滑、能量守恒（参考图的平滑来源）
        freqs_ds, times_ds, db_ds = _aggregate_for_display(freqs_masked, spec.times, db_masked, cfg)
    else:
        # 保峰降采样(块内取最大)：任何峰顶都不会因降采样而丢失
        freqs_ds, times_ds, db_ds = _blockmax_downsample(
            freqs_masked, spec.times, db_masked, cfg.max_grid_freq, cfg.max_grid_time
        )

    db_ds = db_ds - shift  # 聚合完成后再平移到目标显示基准

    verdict_color = {"OK": "#2ecc71", "SUSPECT": "#f39c12", "ABNORMAL": "#e74c3c"}[det.verdict]
    verdict_label = {"OK": "OK - 未见明显异常", "SUSPECT": "疑似异常", "ABNORMAL": "异常"}[det.verdict]

    cmin, cmax = _robust_color_range(db_ds)
    db_title = _colorbar_title(cfg)
    # 悬停始终显示真实 Hz（mel 模式下 x 坐标是内部梅尔值，不加 customdata 悬停会读成梅尔数）
    hz_grid = np.tile(freqs_ds, (len(times_ds), 1))  # shape 同 z: (T_ds, F_ds)
    surface = go.Surface(
        x=_x_coords(freqs_ds, cfg),
        y=times_ds,
        z=db_ds.T,  # z 需要 shape (len(y), len(x)) = (T_ds, F_ds)
        customdata=hz_grid,
        hovertemplate="频率: %{customdata:.0f} Hz<br>时间: %{y:.3f} s<br>幅值: %{z:.1f} dB<extra></extra>",
        colorscale=cfg.colorscale,
        cmin=cmin,
        cmax=cmax,
        colorbar=dict(title=db_title, x=1.02, tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR)),
        contours=dict(z=dict(show=False)),
        # 绸缎质感光照：环境光防止暗部死黑，适度高光让脊线有丝绸反光
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.35, roughness=0.6, fresnel=0.15),
        lightposition=dict(x=-2000, y=1500, z=3000),
    )

    fig = go.Figure(data=[surface])

    # 标注（可选）：在最高的若干个峰位置画竖线标记；默认关闭保持干净视图
    if cfg.annotate:
        for p in det.peaks[:5]:
            color = "#ff2d2d" if p.severity == "abnormal" else "#ffb020"
            px = float(_x_coords(np.array([p.freq_hz]), cfg)[0])
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px],
                    y=[times_ds[0], times_ds[-1]],
                    z=[p.level_db - shift, p.level_db - shift],
                    mode="lines",
                    line=dict(color=color, width=4, dash="dash"),
                    name=f"{p.freq_hz:.0f}Hz ({'异常' if p.severity=='abnormal' else '疑似'})",
                    showlegend=True,
                )
            )

    n_samples_str = f"{wav.n_samples:,} 点 @ {wav.fs:,} Hz, {wav.duration_s:.2f}s, {wav.n_channels}声道"
    # 固定拆成多行，避免长结论文字自动换行时与视角按钮重叠；
    # 文件名过长(真实传感器导出常见)时在标题里截断，完整文件名见报告表格/HTML<title>
    display_name = wav.path.name if len(wav.path.name) <= 48 else wav.path.name[:30] + "…" + wav.path.name[-12:]
    if cfg.annotate:
        title_text = (
            f"<b style='color:{TEXT_COLOR}'>{display_name}</b>  "
            f"<span style='color:{verdict_color}'>[{verdict_label}]</span><br>"
            f"<sup style='color:{MUTED_TEXT}'>{n_samples_str} | FFT={cfg.n_fft} 重叠={cfg.overlap:.0%}</sup><br>"
            f"<sup style='color:{MUTED_TEXT}'>{det.summary}</sup>"
        )
    else:
        # 干净视图：只保留文件名和采样信息，不显示判定结论
        title_text = (
            f"<b style='color:{TEXT_COLOR}'>{display_name}</b><br>"
            f"<sup style='color:{MUTED_TEXT}'>{n_samples_str} | FFT={cfg.n_fft} 重叠={cfg.overlap:.0%}</sup>"
        )

    # 干净视图标题只有两行，场景可以占据更大面积
    scene_top = 0.82 if cfg.annotate else 0.90
    top_margin = 170 if cfg.annotate else 120
    fig.update_layout(
        height=850,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        title=dict(text=title_text, x=0.02, xanchor="left", y=0.90 if cfg.annotate else 0.94, yanchor="top"),
        scene=dict(
            xaxis=dict(**_freq_axis_opts(cfg, float(freqs_masked[0]), float(freqs_masked[-1])), **DARK_SCENE_AXIS),
            yaxis=dict(title="时间 Time (s)", **DARK_SCENE_AXIS),
            zaxis=dict(title=_db_axis_title(cfg), **DARK_SCENE_AXIS),
            camera=CAMERA_PRESETS["斜上(默认)"],
            # “平铺地毯”比例：频率轴拉长、幅值轴压扁（参考图风格），
            # 立方体比例会把幅值抖动在视觉上放大，显得拥挤
            aspectmode="manual",
            aspectratio=dict(x=cfg.aspect_freq, y=cfg.aspect_time, z=cfg.aspect_db),
            domain=dict(x=[0, 1], y=[0, scene_top]),
        ),
        margin=dict(l=0, r=0, t=top_margin, b=0),
        legend=dict(x=0.75, y=0.80, bgcolor="rgba(20,21,32,0.75)", font=dict(color=TEXT_COLOR)),
    )

    # 视角切换按钮：固定在最顶部一行，标题另起在按钮下方，避免长结论文字重叠
    buttons = [
        dict(
            label=name,
            method="relayout",
            args=[{"scene.camera": preset}],
        )
        for name, preset in CAMERA_PRESETS.items()
    ]
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                x=0.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                showactive=True,
                bgcolor=PANEL_COLOR,
                bordercolor=GRID_COLOR,
                font=dict(color=TEXT_COLOR),
                active=0,
            )
        ]
    )

    return fig


def build_2d_figure(wav: WavData, spec: Spectrogram, det: DetectionResult, cfg: AnalysisConfig) -> go.Figure:
    """俯视热力图 + 平均谱曲线，用于量化复核（3D 图仅做直觉展示）。"""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("时频热力图 (俯视)", "平均谱 (中位数压缩)"),
    )

    shift = _compute_shift(spec, cfg)
    fmask = _freq_mask(spec.freqs, cfg)
    freqs_masked = spec.freqs[fmask]
    db_masked = spec.db[fmask, :] - shift

    cmin, cmax = _robust_color_range(db_masked)
    db_title = _colorbar_title(cfg)
    x_masked = _x_coords(freqs_masked, cfg)
    ax_opts = _freq_axis_opts(cfg, float(freqs_masked[0]), float(freqs_masked[-1]))
    ax_opts["title_text"] = ax_opts.pop("title").replace("Frequency ", "")  # 2D 图轴标题紧凑些
    fig.add_trace(
        go.Heatmap(
            x=x_masked,
            y=spec.times,
            z=db_masked.T,
            customdata=np.tile(freqs_masked, (len(spec.times), 1)),
            hovertemplate="频率: %{customdata:.0f} Hz<br>时间: %{y:.3f} s<br>幅值: %{z:.1f} dB<extra></extra>",
            colorscale=cfg.colorscale,
            zmin=cmin,
            zmax=cmax,
            colorbar=dict(title=db_title, x=0.58, tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR)),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(**ax_opts, row=1, col=1)
    fig.update_yaxes(title_text="时间 (s)", row=1, col=1)

    spectrum_db = mean_spectrum(spec, method="median")[fmask] - shift
    fig.add_trace(
        go.Scatter(
            x=x_masked,
            y=spectrum_db,
            mode="lines",
            name="平均谱",
            line=dict(color="#7dd3fc"),
            customdata=freqs_masked,
            hovertemplate="频率: %{customdata:.0f} Hz<br>幅值: %{y:.1f} dB<extra></extra>",
        ),
        row=1,
        col=2,
    )
    if cfg.annotate:
        for p in det.peaks:
            color = "#ff5c5c" if p.severity == "abnormal" else "#ffb020"
            fig.add_trace(
                go.Scatter(
                    x=[float(_x_coords(np.array([p.freq_hz]), cfg)[0])],
                    y=[p.level_db - shift],
                    mode="markers+text",
                    marker=dict(color=color, size=9, symbol="triangle-down"),
                    text=[f"{p.freq_hz:.0f}Hz"],
                    textposition="top center",
                    textfont=dict(color=TEXT_COLOR),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
    fig.update_xaxes(**ax_opts, row=1, col=2)
    fig.update_yaxes(title_text=_db_axis_title(cfg).replace("Amplitude ", ""), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"{wav.path.name} - 2D 复核视图", font=dict(color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PANEL_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_COLOR)
    fig.update_annotations(font=dict(color=TEXT_COLOR))  # subplot_titles
    return fig


def save_figures(fig3d: go.Figure, fig2d: go.Figure | None, out_dir: Path, stem: str, export_png: bool) -> dict:
    """保存交互式 HTML(3D)，可选导出 PNG(3D/2D)，返回相对文件名字典供报告引用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    html_name = f"{stem}_3d.html"
    # 完整内嵌 plotly.js（而非 CDN），保证在完全断网的电脑上也能双击打开
    fig3d.write_html(str(out_dir / html_name), include_plotlyjs=True)
    paths["html_3d"] = html_name

    if export_png:
        try:
            png3d = f"{stem}_3d.png"
            fig3d.write_image(str(out_dir / png3d), width=1200, height=800, scale=2)
            paths["png_3d"] = png3d

            if fig2d is not None:
                png2d = f"{stem}_2d.png"
                fig2d.write_image(str(out_dir / png2d), width=1400, height=600, scale=2)
                paths["png_2d"] = png2d
        except Exception as e:
            # kaleido 缺失或渲染失败时不阻断整体流程，仅跳过 PNG 导出
            paths["png_error"] = str(e)

    return paths

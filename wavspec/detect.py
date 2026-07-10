"""异常检测：在“平均谱”和频带能量上找突出峰 / 偏高频段，给出结论。

注意：判据作用在底层数据矩阵上，而不是 3D 图像上——3D 曲面会因为视角和
山脊遮挡而产生视觉误导，图仅用于人工复核，不作为判定依据。
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .config import AnalysisConfig
from .dsp import Spectrogram, mean_spectrum


@dataclass
class PeakFinding:
    freq_hz: float
    level_db: float
    prominence_db: float
    severity: str  # 'suspect' | 'abnormal'


@dataclass
class BandFinding:
    f_lo: float
    f_hi: float
    level_db: float
    delta_db: float  # 相对全局中位电平的偏高量
    severity: str  # 'suspect' | 'abnormal'


@dataclass
class DetectionResult:
    verdict: str  # 'OK' | 'SUSPECT' | 'ABNORMAL'
    summary: str  # 一句话结论，中文
    peaks: list[PeakFinding] = field(default_factory=list)
    bands: list[BandFinding] = field(default_factory=list)
    baseline_db: float = 0.0


def _find_spectral_peaks(freqs: np.ndarray, spectrum_db: np.ndarray, cfg: AnalysisConfig) -> list[PeakFinding]:
    if len(freqs) < 3:
        return []
    df = freqs[1] - freqs[0]
    min_dist = max(1, int(round(cfg.peak_min_sep_hz / df))) if df > 0 else 1

    # 单边谱换算里 DC(0Hz) 和 Nyquist bin 没有 ×2 补偿，天然比相邻 bin 低约 6dB。
    # scipy 的突出度(prominence)在找不到更高的峰时会一路搜索到数组边界当作基准，
    # 这两个边缘凹陷会被当成“参照低点”，人为放大中间某个峰的突出度。
    # 检测时用相邻值顶替边缘 bin，避免这个物理边界伪影影响峰值判定（仅用于检测，不影响显示）。
    spectrum_for_peaks = spectrum_db.copy()
    if len(spectrum_for_peaks) >= 2:
        spectrum_for_peaks[0] = spectrum_for_peaks[1]
        spectrum_for_peaks[-1] = spectrum_for_peaks[-2]

    idx, props = find_peaks(
        spectrum_for_peaks,
        prominence=cfg.peak_prominence_db,
        distance=min_dist,
    )
    findings = []
    for i, prom in zip(idx, props["prominences"]):
        severity = "abnormal" if prom >= cfg.strong_prominence_db else "suspect"
        findings.append(
            PeakFinding(
                freq_hz=float(freqs[i]),
                level_db=float(spectrum_db[i]),
                prominence_db=float(prom),
                severity=severity,
            )
        )
    # 按突出度从大到小排序，最多保留 top_peaks 个
    findings.sort(key=lambda p: p.prominence_db, reverse=True)
    return findings[: cfg.top_peaks]


def _find_band_energy(freqs: np.ndarray, spectrum_db: np.ndarray, cfg: AnalysisConfig) -> tuple[list[BandFinding], float]:
    if len(freqs) == 0:
        return [], 0.0
    fmin, fmax = freqs[0], freqs[-1]
    edges = np.linspace(fmin, fmax, cfg.band_count + 1)

    band_levels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi) if hi < fmax else (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            band_levels.append(-np.inf)
            continue
        # 用能量（线性域求和）而非 dB 均值，更符合物理意义上的“能量偏高”
        lin = 10 ** (spectrum_db[mask] / 20.0)
        energy_db = 20.0 * np.log10(np.sqrt(np.mean(lin ** 2)))
        band_levels.append(energy_db)

    band_levels = np.array(band_levels)
    finite = band_levels[np.isfinite(band_levels)]
    baseline = float(np.median(finite)) if len(finite) else 0.0

    findings = []
    for (lo, hi), lvl in zip(zip(edges[:-1], edges[1:]), band_levels):
        if not np.isfinite(lvl):
            continue
        delta = lvl - baseline
        if delta >= cfg.band_strong_db:
            severity = "abnormal"
        elif delta >= cfg.band_hot_db:
            severity = "suspect"
        else:
            continue
        findings.append(BandFinding(f_lo=float(lo), f_hi=float(hi), level_db=float(lvl), delta_db=float(delta), severity=severity))

    findings.sort(key=lambda b: b.delta_db, reverse=True)
    return findings, baseline


def detect(spec: Spectrogram, cfg: AnalysisConfig, report_fmax: float | None = None) -> DetectionResult:
    """在压缩后的平均谱上做峰值检测 + 频带能量检测，汇总结论。

    report_fmax: 只报告该频率以下的发现。配合"检测频段比显示频段宽一个余量"
    使用——恰好落在显示上限附近的峰，其突出度需要右侧数据才能正确计算，
    检测在宽频段上做、报告时再裁回显示范围，避免边界漏检。
    """
    spectrum_db = mean_spectrum(spec, method="median")
    baseline = float(np.median(spectrum_db))

    peaks = _find_spectral_peaks(spec.freqs, spectrum_db, cfg)
    bands, band_baseline = _find_band_energy(spec.freqs, spectrum_db, cfg)

    if report_fmax is not None:
        peaks = [p for p in peaks if p.freq_hz <= report_fmax]
        # 完全在报告范围之外的频带丢弃；跨界的频带把上界裁到 report_fmax
        bands = [
            BandFinding(f_lo=b.f_lo, f_hi=min(b.f_hi, report_fmax), level_db=b.level_db,
                        delta_db=b.delta_db, severity=b.severity)
            for b in bands if b.f_lo < report_fmax
        ]

    has_abnormal_peak = any(p.severity == "abnormal" for p in peaks)
    has_suspect_peak = any(p.severity == "suspect" for p in peaks)
    has_abnormal_band = any(b.severity == "abnormal" for b in bands)
    has_suspect_band = any(b.severity == "suspect" for b in bands)

    if has_abnormal_peak or has_abnormal_band:
        verdict = "ABNORMAL"
    elif has_suspect_peak or has_suspect_band:
        verdict = "SUSPECT"
    else:
        verdict = "OK"

    summary = _build_summary(verdict, peaks, bands)

    return DetectionResult(
        verdict=verdict,
        summary=summary,
        peaks=peaks,
        bands=bands,
        baseline_db=baseline,
    )


def _build_summary(verdict: str, peaks: list[PeakFinding], bands: list[BandFinding]) -> str:
    if verdict == "OK":
        return "OK：全频段未见明显异常波峰，能量分布平稳。"

    parts = []
    top_abn_peaks = [p for p in peaks if p.severity == "abnormal"]
    top_susp_peaks = [p for p in peaks if p.severity == "suspect"]
    top_abn_bands = [b for b in bands if b.severity == "abnormal"]
    top_susp_bands = [b for b in bands if b.severity == "suspect"]

    if top_abn_peaks:
        freqs_str = "、".join(f"{p.freq_hz:.0f}Hz(+{p.prominence_db:.1f}dB)" for p in top_abn_peaks[:3])
        parts.append(f"检测到异常波峰: {freqs_str}")
    elif top_susp_peaks:
        freqs_str = "、".join(f"{p.freq_hz:.0f}Hz(+{p.prominence_db:.1f}dB)" for p in top_susp_peaks[:3])
        parts.append(f"疑似异常波峰: {freqs_str}")

    if top_abn_bands:
        b = top_abn_bands[0]
        parts.append(f"{b.f_lo:.0f}-{b.f_hi:.0f}Hz 频段能量明显偏高(+{b.delta_db:.1f}dB)")
    elif top_susp_bands:
        b = top_susp_bands[0]
        parts.append(f"{b.f_lo:.0f}-{b.f_hi:.0f}Hz 频段能量偏高(+{b.delta_db:.1f}dB)")

    prefix = "异常：" if verdict == "ABNORMAL" else "疑似异常："
    return prefix + "；".join(parts) if parts else prefix + "存在超阈值特征，请人工复核。"

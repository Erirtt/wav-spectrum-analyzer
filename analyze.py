#!/usr/bin/env python3
"""WAV 传感器信号频谱分析工具 —— 命令行入口。

用法示例:
  # 单文件 / 批量分析一个文件夹（自动查找所有 .wav）
  # 不指定 -o 时自动输出到 out/<时间戳>/，每次运行互不覆盖
  python3 analyze.py samples/stage1.wav
  python3 analyze.py samples/

  # 指定固定输出目录（会覆盖同名目录里的旧结果）
  python3 analyze.py samples/ -o out_fixed/

  # 自定义 FFT 参数 / 关心频段
  python3 analyze.py samples/ --n-fft 8192 --overlap 0.5 --fmax 5000

  # 频率轴对数刻度(便于观察低频故障特征)；幅值默认已是"相对本底噪声"(本底≈0dB，异常峰为正数)
  python3 analyze.py samples/ --freq-scale log

  # 切换幅值显示基准: --dbfs(绝对满量程) 或 --relative(相对该文件峰值)
  python3 analyze.py samples/ --dbfs
  python3 analyze.py samples/ --relative

  # 额外导出 PNG 截图 / 在图上叠加异常判定标注（默认只生成干净的交互式 HTML）
  python3 analyze.py samples/ --png --annotate
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

from wavspec.config import AnalysisConfig
from wavspec.io_utils import load_wav
from wavspec.dsp import compute_stft
from wavspec.detect import detect
from wavspec.viz import build_3d_figure, build_2d_figure, save_figures
from wavspec.report import FileReportEntry, render_html_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WAV 传感器信号频谱分析（3D 瀑布图 + 异常检测）")
    p.add_argument("input", help="wav 文件路径，或包含多个 wav 的文件夹")
    p.add_argument(
        "-o", "--out", default=None,
        help="输出目录（默认不指定：自动生成 out/<时间戳>/，同一次分析结果不会互相覆盖）",
    )

    p.add_argument("--n-fft", type=int, default=4096, help="FFT 长度（默认 4096）")
    p.add_argument("--overlap", type=float, default=0.75, help="帧重叠比例 0~1（默认 0.75）")
    p.add_argument("--window", default="hann", help="窗函数（默认 hann）")
    p.add_argument("--fmin", type=float, default=None, help="关心频段下限 Hz（默认 0）")
    p.add_argument("--fmax", type=float, default=10000.0, help="关心频段上限 Hz（默认 10000；想看全频段可设为采样率一半，如 22050）")
    p.add_argument("--channel", default="mean", help="多声道处理: mean 或声道号，如 0（默认 mean）")

    p.add_argument("--peak-prominence-db", type=float, default=8.0, help="疑似峰突出度阈值 dB（默认 8）")
    p.add_argument("--strong-prominence-db", type=float, default=15.0, help="异常峰突出度阈值 dB（默认 15）")
    p.add_argument("--band-hot-db", type=float, default=10.0, help="频段偏高阈值 dB（默认 10）")
    p.add_argument("--band-strong-db", type=float, default=15.0, help="频段异常阈值 dB（默认 15）")
    p.add_argument("--band-count", type=int, default=8, help="频段划分数量（默认 8）")

    p.add_argument("--png", action="store_true", help="额外导出 PNG 截图和 2D 复核图（需要 kaleido，较慢；默认只生成交互式 HTML）")
    p.add_argument("--annotate", action="store_true", help="在图上叠加异常判定标注（峰值线/图例/结论文字；默认干净视图，判定结论只打印到终端）")

    p.add_argument(
        "--freq-scale", choices=["linear", "log", "mel"], default="mel",
        help="频率轴刻度（默认 mel，人耳感知刻度）。log: 对数轴；linear: 线性轴。"
             "均只影响显示，检测始终用全分辨率线性频率",
    )
    db_group = p.add_mutually_exclusive_group()
    db_group.add_argument(
        "--dbfs", action="store_const", dest="db_mode", const="dbfs",
        help="幅值显示为绝对满量程基准(dBFS)，行业标准做法，但大多数读数是负数",
    )
    db_group.add_argument(
        "--relative", action="store_const", dest="db_mode", const="peak",
        help="幅值显示相对该文件自身峰值归一化(峰值=0dB)，其余为负",
    )
    p.set_defaults(db_mode="noise_floor")  # 默认: 相对本底噪声，本底≈0dB，异常峰为正数
    p.add_argument(
        "--no-smooth", action="store_true",
        help="关闭 3D 曲面的平滑聚合，显示原始抽稀数据（毛躁但未经加工）；检测始终用全分辨率数据，不受此项影响",
    )

    return p.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        n_fft=args.n_fft,
        overlap=args.overlap,
        window=args.window,
        fmin=args.fmin,
        fmax=args.fmax,
        channel=args.channel,
        peak_prominence_db=args.peak_prominence_db,
        strong_prominence_db=args.strong_prominence_db,
        band_hot_db=args.band_hot_db,
        band_strong_db=args.band_strong_db,
        band_count=args.band_count,
        export_png=args.png,
        freq_scale=args.freq_scale,
        db_mode=args.db_mode,
        smooth_display=not args.no_smooth,
        annotate=args.annotate,
    )


def find_wav_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.wav")) + sorted(input_path.glob("*.WAV"))
        return files
    raise FileNotFoundError(f"找不到输入路径: {input_path}")


def process_one(path: Path, cfg: AnalysisConfig, out_dir: Path) -> FileReportEntry | None:
    print(f"\n[处理] {path.name}")
    try:
        wav = load_wav(path, channel=cfg.channel)
    except Exception as e:
        print(f"  ✗ 读取失败: {e}", file=sys.stderr)
        return None

    print(f"  采样率={wav.fs}Hz  样本数={wav.n_samples:,}  时长={wav.duration_s:.2f}s  声道数={wav.n_channels}")

    try:
        spec = compute_stft(wav.signal, wav.fs, cfg)  # 显示用，按 cfg.fmax 裁剪
    except Exception as e:
        print(f"  ✗ STFT 计算失败: {e}", file=sys.stderr)
        return None

    # 检测用比显示宽 5% 的频段，避免恰好在 fmax 边界上的峰漏检；报告仍裁回 fmax
    if cfg.fmax is not None:
        det_fmax = min(cfg.fmax * 1.05, wav.fs / 2.0)
        spec_det = compute_stft(wav.signal, wav.fs, cfg, fmax_override=det_fmax)
        det = detect(spec_det, cfg, report_fmax=cfg.fmax)
    else:
        det = detect(spec, cfg)
    print(f"  结论: [{det.verdict}] {det.summary}")

    fig3d = build_3d_figure(wav, spec, det, cfg)
    fig2d = build_2d_figure(wav, spec, det, cfg)
    fig_paths = save_figures(fig3d, fig2d, out_dir, stem=path.stem, export_png=cfg.export_png)

    if fig_paths.get("png_error"):
        print(f"  ⚠ PNG 导出失败（可能缺少 kaleido 或渲染超时），已跳过: {fig_paths['png_error']}", file=sys.stderr)

    return FileReportEntry(wav=wav, det=det, fig_paths=fig_paths)


def main():
    args = parse_args()
    cfg = build_config(args)
    input_path = Path(args.input)
    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("out") / timestamp
    else:
        out_dir = Path(args.out)

    files = find_wav_files(input_path)
    if not files:
        print(f"未在 {input_path} 找到 .wav 文件", file=sys.stderr)
        sys.exit(1)

    print(f"共找到 {len(files)} 个 wav 文件，输出目录: {out_dir}")

    entries = []
    for f in files:
        entry = process_one(f, cfg, out_dir)
        if entry:
            entries.append(entry)

    if not entries:
        print("没有成功分析的文件，未生成报告", file=sys.stderr)
        sys.exit(1)

    report_path = render_html_report(entries, cfg, out_dir)
    print(f"\n✓ 分析完成，共 {len(entries)} 个文件成功处理")
    print(f"✓ 汇总报告: {report_path.resolve()}")


if __name__ == "__main__":
    main()

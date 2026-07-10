"""批量分析汇总 + HTML 报告生成。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from .config import AnalysisConfig
from .detect import DetectionResult
from .io_utils import WavData


@dataclass
class FileReportEntry:
    wav: WavData
    det: DetectionResult
    fig_paths: dict


_VERDICT_BADGE = {
    "OK": ("🟢", "#2ecc71", "OK"),
    "SUSPECT": ("🟡", "#f39c12", "疑似异常"),
    "ABNORMAL": ("🔴", "#e74c3c", "异常"),
}


def render_html_report(entries: list[FileReportEntry], cfg: AnalysisConfig, out_dir: Path, title: str = "WAV 频谱批量分析报告") -> Path:
    rows = []
    for e in entries:
        thumb = e.fig_paths.get("png_3d")
        thumb_html = (
            f'<a href="{e.fig_paths["html_3d"]}"><img src="{thumb}" style="max-width:260px;border:1px solid #333;border-radius:6px" /></a>'
            if thumb
            else f'<a href="{e.fig_paths["html_3d"]}">打开交互式 3D 图</a>'
        )
        links = [thumb_html]
        if e.fig_paths.get("html_2d"):
            links.append(f'<a href="{e.fig_paths["html_2d"]}">2D 复核图</a>')
        file_cell = f"""
          <td>{"<br>".join(links)}</td>
          <td><b>{e.wav.path.name}</b><br>
              <span class="meta">{e.wav.fs:,} Hz · {e.wav.n_samples:,} 点 · {e.wav.duration_s:.2f}s · {e.wav.n_channels}声道</span>
          </td>"""

        if not cfg.annotate:
            # 干净模式：只列文件信息和图链接，不显示判定结论
            rows.append(f"<tr>{file_cell}</tr>")
            continue

        emoji, color, label = _VERDICT_BADGE[e.det.verdict]
        peak_str = "、".join(
            f"{p.freq_hz:.0f}Hz(+{p.prominence_db:.1f}dB{'⚠️' if p.severity=='abnormal' else ''})"
            for p in e.det.peaks[:5]
        ) or "—"
        band_str = "、".join(
            f"{b.f_lo:.0f}-{b.f_hi:.0f}Hz(+{b.delta_db:.1f}dB)" for b in e.det.bands[:3]
        ) or "—"

        rows.append(f"""
        <tr>{file_cell}
          <td><span class="badge" style="background:{color}">{emoji} {label}</span><br>
              <span class="summary">{e.det.summary}</span>
          </td>
          <td class="small">峰值: {peak_str}</td>
          <td class="small">频段: {band_str}</td>
        </tr>""")

    n_ok = sum(1 for e in entries if e.det.verdict == "OK")
    n_susp = sum(1 for e in entries if e.det.verdict == "SUSPECT")
    n_abn = sum(1 for e in entries if e.det.verdict == "ABNORMAL")

    # 干净模式下不显示判定统计/阈值参数/判定列
    if cfg.annotate:
        verdict_stats = f" — 🟢 OK: <b>{n_ok}</b> 🟡 疑似异常: <b>{n_susp}</b> 🔴 异常: <b>{n_abn}</b>"
        threshold_cfg = (
            f", 峰值突出度阈值={cfg.peak_prominence_db}dB(异常{cfg.strong_prominence_db}dB)"
            f", 频段阈值={cfg.band_hot_db}dB(异常{cfg.band_strong_db}dB)"
        )
        table_head = "<th>3D 瀑布图</th><th>文件信息</th><th>结论</th><th>波峰明细</th><th>频段明细</th>"
    else:
        verdict_stats = ""
        threshold_cfg = ""
        table_head = "<th>3D 瀑布图</th><th>文件信息</th>"

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; color: #e6e6ef; background: #0d0e14; }}
  h1 {{ font-size: 20px; color: #f2f2f8; }}
  .stats {{ margin: 12px 0 20px; font-size: 14px; color: #9a9db0; }}
  .stats b {{ font-size: 16px; color: #f2f2f8; }}
  table {{ border-collapse: collapse; width: 100%; background: #12131c; box-shadow: 0 1px 3px rgba(0,0,0,0.4); }}
  th, td {{ border-bottom: 1px solid #262838; padding: 10px 12px; vertical-align: top; text-align: left; font-size: 13px; }}
  th {{ background: #1a1c28; color: #c9cbdb; }}
  a {{ color: #7dd3fc; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; color: white; font-weight: 600; font-size: 12px; }}
  .meta {{ color: #9a9db0; font-size: 12px; }}
  .summary {{ font-size: 12px; color: #c9cbdb; }}
  .small {{ font-size: 12px; color: #c9cbdb; max-width: 260px; }}
  .cfg {{ font-size: 12px; color: #9a9db0; margin-top: 6px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="stats">
    共分析 <b>{len(entries)}</b> 个文件{verdict_stats}
    <div class="cfg">参数: FFT={cfg.n_fft}, 重叠={cfg.overlap:.0%}, 窗={cfg.window}{threshold_cfg}</div>
  </div>
  <table>
    <thead><tr>{table_head}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path

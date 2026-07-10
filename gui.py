#!/usr/bin/env python3
"""WAV 频谱分析工具 —— 桌面图形界面（纯净版：只生成干净的交互式 3D HTML）。

选文件/选文件夹 -> 点生成 -> 自动在默认浏览器打开结果，全程离线可用。
"""
from __future__ import annotations

import queue
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, scrolledtext
import tkinter as tk

from wavspec.config import AnalysisConfig
from wavspec.io_utils import load_wav
from wavspec.dsp import compute_stft
from wavspec.detect import detect
from wavspec.viz import build_3d_figure, save_figures
from wavspec.report import FileReportEntry, render_html_report

BG = "#0d0e14"
PANEL = "#12131c"
BORDER = "#333747"
TEXT = "#e6e6ef"
MUTED = "#9a9db0"
ACCENT = "#7dd3fc"
ACCENT_HOVER = "#a5e2fd"


def app_dir() -> Path:
    """打包成 exe 后，返回 exe 所在目录；开发模式下返回本脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def find_wav_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV")))
        elif p.is_file():
            files.append(p)
    # 去重保序
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WAV 频谱分析工具")
        self.geometry("720x520")
        self.configure(bg=BG)
        self.minsize(600, 420)

        self.selected_paths: list[Path] = []
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_out_dir: Path | None = None

        self._build_ui()
        self.after(100, self._drain_log_queue)

    # ---------- UI ----------
    def _build_ui(self):
        pad = dict(padx=16, pady=8)

        header = tk.Label(
            self, text="WAV 频谱分析工具", bg=BG, fg=TEXT,
            font=("Microsoft YaHei UI", 16, "bold"), anchor="w",
        )
        header.pack(fill="x", **pad)

        sub = tk.Label(
            self, text="选择 WAV 文件或文件夹，生成可离线打开的交互式 3D 频谱图",
            bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 10), anchor="w",
        )
        sub.pack(fill="x", padx=16)

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", **pad)

        self.btn_files = self._make_button(btn_row, "选择 WAV 文件（可多选）", self._pick_files)
        self.btn_files.pack(side="left", padx=(0, 8))

        self.btn_folder = self._make_button(btn_row, "选择文件夹（批量）", self._pick_folder)
        self.btn_folder.pack(side="left")

        self.selected_label = tk.Label(
            self, text="尚未选择文件", bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 10),
            anchor="w", justify="left", wraplength=680,
        )
        self.selected_label.pack(fill="x", padx=16, pady=(0, 8))

        self.btn_run = self._make_button(self, "开始生成", self._start, primary=True)
        self.btn_run.pack(fill="x", padx=16, pady=(0, 8))
        self.btn_run.config(state="disabled")

        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self.log_box = scrolledtext.ScrolledText(
            log_frame, bg=PANEL, fg=TEXT, insertbackground=TEXT,
            font=("Menlo", 10), relief="flat", borderwidth=0,
        )
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        bottom_row = tk.Frame(self, bg=BG)
        bottom_row.pack(fill="x", padx=16, pady=(0, 16))

        self.btn_open = self._make_button(bottom_row, "打开输出文件夹", self._open_output)
        self.btn_open.pack(side="left")
        self.btn_open.config(state="disabled")

        self.status_label = tk.Label(bottom_row, text="", bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 10))
        self.status_label.pack(side="left", padx=12)

    def _make_button(self, parent, text, command, primary=False):
        bg = ACCENT if primary else PANEL
        fg = "#04141c" if primary else TEXT
        b = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=ACCENT_HOVER if primary else BORDER,
            activeforeground=fg, relief="flat", font=("Microsoft YaHei UI", 10, "bold" if primary else "normal"),
            padx=14, pady=8, cursor="hand2", borderwidth=0, highlightthickness=0,
        )
        return b

    # ---------- 交互 ----------
    def _pick_files(self):
        files = filedialog.askopenfilenames(title="选择 WAV 文件", filetypes=[("WAV 音频", "*.wav *.WAV")])
        if files:
            self.selected_paths = [Path(f) for f in files]
            self._refresh_selected_label()

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="选择包含 WAV 文件的文件夹")
        if folder:
            self.selected_paths = [Path(folder)]
            self._refresh_selected_label()

    def _refresh_selected_label(self):
        wavs = find_wav_files(self.selected_paths)
        if not wavs:
            self.selected_label.config(text="所选路径中没有找到 .wav 文件", fg="#ff8080")
            self.btn_run.config(state="disabled")
            return
        names = "、".join(f.name for f in wavs[:5])
        more = f" 等共 {len(wavs)} 个文件" if len(wavs) > 5 else ""
        self.selected_label.config(text=f"已选择: {names}{more}", fg=MUTED)
        self.btn_run.config(state="normal")

    def _log(self, msg: str):
        self.log_queue.put(msg)

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        wavs = find_wav_files(self.selected_paths)
        if not wavs:
            return
        self.btn_run.config(state="disabled", text="正在生成…")
        self.btn_files.config(state="disabled")
        self.btn_folder.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.status_label.config(text="")

        self.worker = threading.Thread(target=self._run_analysis, args=(wavs,), daemon=True)
        self.worker.start()
        self._poll_worker()

    def _poll_worker(self):
        if self.worker and self.worker.is_alive():
            self.after(200, self._poll_worker)
        else:
            self.btn_run.config(state="normal", text="开始生成")
            self.btn_files.config(state="normal")
            self.btn_folder.config(state="normal")

    def _run_analysis(self, wavs: list[Path]):
        try:
            cfg = AnalysisConfig()  # 纯净默认：不标注、不导出PNG、全离线嵌入plotly.js
            out_dir = app_dir() / "频谱分析结果" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log(f"共找到 {len(wavs)} 个 WAV 文件")
            self._log(f"输出目录: {out_dir}\n")

            entries = []
            for wav_path in wavs:
                self._log(f"[处理] {wav_path.name}")
                try:
                    wav = load_wav(wav_path, channel=cfg.channel)
                    spec = compute_stft(wav.signal, wav.fs, cfg)  # 显示用
                    # 检测用比显示宽 5% 的频段，避免 fmax 边界上的峰漏检
                    if cfg.fmax is not None:
                        det_fmax = min(cfg.fmax * 1.05, wav.fs / 2.0)
                        spec_det = compute_stft(wav.signal, wav.fs, cfg, fmax_override=det_fmax)
                        det = detect(spec_det, cfg, report_fmax=cfg.fmax)
                    else:
                        det = detect(spec, cfg)
                    fig3d = build_3d_figure(wav, spec, det, cfg)
                    fig_paths = save_figures(fig3d, None, out_dir, stem=wav_path.stem, export_png=False)
                    entries.append(FileReportEntry(wav=wav, det=det, fig_paths=fig_paths))
                    self._log(f"  ✓ 完成 -> {fig_paths['html_3d']}")
                except Exception as e:
                    self._log(f"  ✗ 失败: {e}")

            if not entries:
                self._log("\n没有成功生成的文件。")
                self.status_label.config(text="生成失败", fg="#ff8080")
                return

            report_path = render_html_report(entries, cfg, out_dir)
            self.last_out_dir = out_dir
            self._log(f"\n✓ 全部完成，共 {len(entries)} 个文件")
            self.status_label.config(text=f"完成，共 {len(entries)} 个文件", fg="#7ee787")
            self.btn_open.config(state="normal")

            # 单文件直接打开该文件的 3D 图，多文件打开汇总报告
            target = entries[0].fig_paths["html_3d"] if len(entries) == 1 else "report.html"
            webbrowser.open((out_dir / target).resolve().as_uri())
        except Exception:
            self._log("发生错误:\n" + traceback.format_exc())
            self.status_label.config(text="发生错误", fg="#ff8080")

    def _open_output(self):
        if self.last_out_dir and self.last_out_dir.exists():
            webbrowser.open(self.last_out_dir.resolve().as_uri())


if __name__ == "__main__":
    App().mainloop()

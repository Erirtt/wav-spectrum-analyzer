#!/usr/bin/env python3
"""WAV 频谱分析工具 —— 桌面图形界面（纯净版：只生成干净的交互式 3D+2D HTML）。

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
from tkinter import filedialog, scrolledtext, ttk
import tkinter as tk

BG = "#0d0e14"
PANEL = "#12131c"
PANEL_LIGHT = "#1a1c28"
BORDER = "#333747"
TEXT = "#e6e6ef"
MUTED = "#9a9db0"
ACCENT = "#7dd3fc"
ACCENT_HOVER = "#a5e2fd"
GOOD = "#7ee787"
BAD = "#ff8080"
FONT = "Microsoft YaHei UI"

# 重量级依赖(numpy/scipy/plotly/soundfile)故意不在模块顶层导入——那会让窗口
# 出现前多等好几秒。改为窗口一显示就在后台线程"预热"导入，用户挑文件的这几秒
# 时间刚好把导入成本吃掉；真正处理时如果还没导完，等一下也不影响正确性。
_wavspec = {}
_import_done = threading.Event()


def _prefetch_imports():
    from wavspec.config import AnalysisConfig
    from wavspec.io_utils import load_wav
    from wavspec.dsp import compute_stft
    from wavspec.detect import detect
    from wavspec.viz import build_3d_figure, build_2d_figure, save_figures
    from wavspec.report import FileReportEntry, render_html_report

    _wavspec.update(
        AnalysisConfig=AnalysisConfig, load_wav=load_wav, compute_stft=compute_stft,
        detect=detect, build_3d_figure=build_3d_figure, build_2d_figure=build_2d_figure,
        save_figures=save_figures, FileReportEntry=FileReportEntry,
        render_html_report=render_html_report,
    )
    _import_done.set()


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
        self.geometry("760x580")
        self.configure(bg=BG)
        self.minsize(640, 460)

        self.selected_paths: list[Path] = []
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_out_dir: Path | None = None

        self._build_ui()
        self.after(80, self._drain_log_queue)
        threading.Thread(target=_prefetch_imports, daemon=True).start()

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=PANEL, background=ACCENT, bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT,
        )

        header = tk.Label(
            self, text="WAV 频谱分析工具", bg=BG, fg=TEXT,
            font=(FONT, 17, "bold"), anchor="w",
        )
        header.pack(fill="x", padx=18, pady=(16, 2))

        sub = tk.Label(
            self, text="选择 WAV 文件或文件夹，生成可离线打开的交互式 3D + 2D 频谱图",
            bg=BG, fg=MUTED, font=(FONT, 10), anchor="w",
        )
        sub.pack(fill="x", padx=18, pady=(0, 12))

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=18)
        self.btn_files = self._make_button(btn_row, "📄 选择 WAV 文件", self._pick_files)
        self.btn_files.pack(side="left", padx=(0, 8))
        self.btn_folder = self._make_button(btn_row, "📁 选择文件夹（批量）", self._pick_folder)
        self.btn_folder.pack(side="left", padx=(0, 8))
        self.btn_clear = self._make_button(btn_row, "清空", self._clear_selection)
        self.btn_clear.pack(side="left")

        # 已选文件列表（替代原来单行截断的label，批量时更好用）
        list_frame = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        list_frame.pack(fill="both", expand=False, padx=18, pady=(10, 10))
        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side="right", fill="y")
        self.file_listbox = tk.Listbox(
            list_frame, bg=PANEL, fg=TEXT, selectbackground=BORDER, relief="flat",
            font=(FONT, 10), height=6, borderwidth=0, highlightthickness=0,
            yscrollcommand=list_scroll.set,
        )
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=8, pady=6)
        list_scroll.config(command=self.file_listbox.yview)
        self._set_placeholder()

        self.btn_run = self._make_button(self, "开始生成", self._start, primary=True)
        self.btn_run.pack(fill="x", padx=18, pady=(0, 10))
        self.btn_run.config(state="disabled")

        # 进度条 + 当前处理状态（默认隐藏，处理时才显示）
        self.progress_frame = tk.Frame(self, bg=BG)
        self.progress = ttk.Progressbar(
            self.progress_frame, style="Dark.Horizontal.TProgressbar",
            mode="determinate", maximum=100,
        )
        self.progress.pack(fill="x", pady=(0, 4))
        self.progress_label = tk.Label(
            self.progress_frame, text="", bg=BG, fg=MUTED, font=(FONT, 9), anchor="w",
        )
        self.progress_label.pack(fill="x")

        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        tk.Label(log_frame, text="运行日志", bg=BG, fg=MUTED, font=(FONT, 9), anchor="w").pack(fill="x")
        self.log_box = scrolledtext.ScrolledText(
            log_frame, bg=PANEL, fg=TEXT, insertbackground=TEXT,
            font=("Menlo", 10), relief="flat", borderwidth=0,
        )
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))
        self.log_box.configure(state="disabled")

        bottom_row = tk.Frame(self, bg=BG)
        bottom_row.pack(fill="x", padx=18, pady=(0, 16))
        self.btn_open = self._make_button(bottom_row, "📂 打开输出文件夹", self._open_output)
        self.btn_open.pack(side="left")
        self.btn_open.config(state="disabled")
        self.status_label = tk.Label(bottom_row, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.status_label.pack(side="left", padx=12)

    def _make_button(self, parent, text, command, primary=False):
        bg = ACCENT if primary else PANEL_LIGHT
        fg = "#04141c" if primary else TEXT
        hover_bg = ACCENT_HOVER if primary else BORDER
        b = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=hover_bg, activeforeground=fg, relief="flat",
            font=(FONT, 10, "bold" if primary else "normal"),
            padx=14, pady=9, cursor="hand2", borderwidth=0, highlightthickness=0,
        )
        b.bind("<Enter>", lambda e: b.config(bg=hover_bg))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _set_placeholder(self):
        self.file_listbox.delete(0, "end")
        self.file_listbox.insert("end", "  尚未选择文件…")
        self.file_listbox.itemconfig(0, fg=MUTED)

    # ---------- 交互 ----------
    def _pick_files(self):
        files = filedialog.askopenfilenames(title="选择 WAV 文件", filetypes=[("WAV 音频", "*.wav *.WAV")])
        if files:
            self.selected_paths = [Path(f) for f in files]
            self._refresh_selected_list()

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="选择包含 WAV 文件的文件夹")
        if folder:
            self.selected_paths = [Path(folder)]
            self._refresh_selected_list()

    def _clear_selection(self):
        self.selected_paths = []
        self._set_placeholder()
        self.btn_run.config(state="disabled")

    def _refresh_selected_list(self):
        wavs = find_wav_files(self.selected_paths)
        self.file_listbox.delete(0, "end")
        if not wavs:
            self.file_listbox.insert("end", "  所选路径中没有找到 .wav 文件")
            self.file_listbox.itemconfig(0, fg=BAD)
            self.btn_run.config(state="disabled")
            return
        for f in wavs:
            self.file_listbox.insert("end", f"  {f.name}")
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
        self.after(80, self._drain_log_queue)

    def _set_progress(self, done: int, total: int, current_name: str = ""):
        pct = (done / total * 100) if total else 0
        self.progress["value"] = pct
        text = f"处理中 ({done}/{total})" + (f": {current_name}" if current_name else "")
        self.progress_label.config(text=text)

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        wavs = find_wav_files(self.selected_paths)
        if not wavs:
            return
        self.btn_run.config(state="disabled", text="正在生成…")
        self.btn_files.config(state="disabled")
        self.btn_folder.config(state="disabled")
        self.btn_clear.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.status_label.config(text="")
        self.progress_frame.pack(fill="x", padx=18, pady=(0, 10), before=self.log_box.master)
        self._set_progress(0, len(wavs))

        self.worker = threading.Thread(target=self._run_analysis, args=(wavs,), daemon=True)
        self.worker.start()
        self._poll_worker()

    def _poll_worker(self):
        if self.worker and self.worker.is_alive():
            self.after(150, self._poll_worker)
        else:
            self.btn_run.config(state="normal", text="开始生成")
            self.btn_files.config(state="normal")
            self.btn_folder.config(state="normal")
            self.btn_clear.config(state="normal")
            self.progress_frame.pack_forget()

    def _run_analysis(self, wavs: list[Path]):
        try:
            if not _import_done.is_set():
                self._log("首次启动，正在加载分析模块…")
                _import_done.wait()

            cfg = _wavspec["AnalysisConfig"]()  # 纯净默认：不标注、mel轴、noise_floor基准
            out_dir = app_dir() / "频谱分析结果" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log(f"共找到 {len(wavs)} 个 WAV 文件")
            self._log(f"输出目录: {out_dir}\n")

            entries = []
            for i, wav_path in enumerate(wavs):
                self.after(0, self._set_progress, i, len(wavs), wav_path.name)
                self._log(f"[处理] {wav_path.name}")
                try:
                    wav = _wavspec["load_wav"](wav_path, channel=cfg.channel)
                    spec = _wavspec["compute_stft"](wav.signal, wav.fs, cfg)
                    if cfg.fmax is not None:
                        det_fmax = min(cfg.fmax * 1.05, wav.fs / 2.0)
                        spec_det = _wavspec["compute_stft"](wav.signal, wav.fs, cfg, fmax_override=det_fmax)
                        det = _wavspec["detect"](spec_det, cfg, report_fmax=cfg.fmax)
                    else:
                        det = _wavspec["detect"](spec, cfg)
                    fig3d = _wavspec["build_3d_figure"](wav, spec, det, cfg)
                    fig2d = _wavspec["build_2d_figure"](wav, spec, det, cfg)
                    fig_paths = _wavspec["save_figures"](fig3d, fig2d, out_dir, stem=wav_path.stem, export_png=False)
                    entries.append(_wavspec["FileReportEntry"](wav=wav, det=det, fig_paths=fig_paths))
                    self._log(f"  ✓ 完成 -> {fig_paths['html_3d']}")
                except Exception as e:
                    self._log(f"  ✗ 失败: {e}")

            self.after(0, self._set_progress, len(wavs), len(wavs))

            if not entries:
                self._log("\n没有成功生成的文件。")
                self.status_label.config(text="生成失败", fg=BAD)
                return

            report_path = _wavspec["render_html_report"](entries, cfg, out_dir)
            self.last_out_dir = out_dir
            self._log(f"\n✓ 全部完成，共 {len(entries)} 个文件")
            self.status_label.config(text=f"完成，共 {len(entries)} 个文件", fg=GOOD)
            self.btn_open.config(state="normal")

            target = entries[0].fig_paths["html_3d"] if len(entries) == 1 else "report.html"
            webbrowser.open((out_dir / target).resolve().as_uri())
        except Exception:
            self._log("发生错误:\n" + traceback.format_exc())
            self.status_label.config(text="发生错误", fg=BAD)

    def _open_output(self):
        if self.last_out_dir and self.last_out_dir.exists():
            webbrowser.open(self.last_out_dir.resolve().as_uri())


if __name__ == "__main__":
    App().mainloop()

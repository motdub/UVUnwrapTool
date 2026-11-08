"""app.py - minimal UI: pick meshes, pick output folder, generate UV-unwrapped GLBs.
Resumable: finished files are recorded in <output>/.uvunwrap_progress.json and
skipped on the next run (tick "Resume"). Stop at any time and run again later."""
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Portable mode: use dependencies from ./libs next to this file.
_LIBS = Path(__file__).resolve().parent / "libs"
if _LIBS.is_dir():
    sys.path.insert(0, str(_LIBS))

try:
    import uvcore
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    uvcore = None
    IMPORT_ERROR = e

MODEL_TYPES = ("*.obj", "*.ply", "*.stl", "*.glb", "*.gltf", "*.off")
MANIFEST = ".uvunwrap_progress.json"
# Parallel files: xatlas already multithreads chart computation but packing is
# single-threaded; 2-3 workers keep a many-core CPU busy. Each big mesh needs
# roughly 1-2 GB RAM, so cap it.
MAX_PROCS = max(1, min(3, (os.cpu_count() or 2) // 4))


def make_output_path(folder, src):
    stem = Path(src).stem
    while True:
        out = Path(folder) / f"{stem}_{random.randint(0, 999999):06d}.glb"
        if not out.exists() and not Path(str(out) + ".part").exists():
            return out


def file_key(path):
    try:
        st = os.stat(path)
        return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return None


def load_manifest(folder):
    try:
        with open(Path(folder) / MANIFEST, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_manifest(folder, data):
    path = Path(folder) / MANIFEST
    tmp = str(path) + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


def stage_name(frac):
    if frac < 0.15:
        return "Loading"
    if frac < 0.25:
        return "Repairing"
    if frac < 0.80:
        return "Computing charts"
    if frac < 0.90:
        return "Packing atlas"
    return "Exporting GLB"


def new_slot(ctx, out_q):
    in_q = ctx.Queue()
    proc = ctx.Process(target=uvcore.worker_main, args=(in_q, out_q), daemon=True)
    proc.start()
    return {"proc": proc, "q": in_q, "job": None, "t0": 0.0, "stage": ""}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UV Unwrap Tool")
        self.geometry("680x520")
        self.minsize(540, 400)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.files = []
        self.out_dir = None
        self.busy = False
        self.ui_q = queue.Queue()
        self.cancel = threading.Event()

        self.btn_select = ttk.Button(self, text="1. Select Models", command=self.select_models)
        self.btn_select.pack(fill="x", padx=10, pady=(10, 4))

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=4)
        self.table = ttk.Treeview(frame, columns=("file", "status"), show="headings", height=8)
        self.table.heading("file", text="Model")
        self.table.heading("status", text="Status")
        self.table.column("file", width=330)
        self.table.column("status", width=270)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.btn_out = ttk.Button(self, text="2. Select Output Folder", command=self.select_output)
        self.btn_out.pack(fill="x", padx=10, pady=4)
        self.out_label = ttk.Label(self, text="No output folder selected", anchor="w")
        self.out_label.pack(fill="x", padx=12)

        self.resume_var = tk.BooleanVar(value=True)
        self.chk_resume = ttk.Checkbutton(
            self, text="Resume: skip models already finished in this output folder",
            variable=self.resume_var)
        self.chk_resume.pack(anchor="w", padx=12, pady=(4, 0))

        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=(8, 4))
        self.btn_go = ttk.Button(row, text="3. Generate New UV Unwrapped Models", command=self.generate)
        self.btn_go.pack(side="left", fill="x", expand=True)
        self.btn_stop = ttk.Button(row, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(6, 0))

        self.bar = ttk.Progressbar(self, maximum=100)
        self.bar.pack(fill="x", padx=10, pady=(4, 0))
        self.pct_label = ttk.Label(self, text="0%")
        self.pct_label.pack(pady=(2, 10))

        self.after(100, self.poll)

    # ---------- buttons ----------
    def select_models(self):
        if self.busy:
            return
        paths = filedialog.askopenfilenames(
            title="Select 3D models",
            filetypes=[("3D models", MODEL_TYPES), ("All files", "*.*")],
        )
        if not paths:
            return
        self.files = list(paths)
        self.table.delete(*self.table.get_children())
        for i, p in enumerate(self.files):
            self.table.insert("", "end", iid=str(i), values=(Path(p).name, "Queued"))
        self.set_progress(0)

    def select_output(self):
        if self.busy:
            return
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.out_dir = folder
            self.out_label.config(text=f"Output: {folder}")

    def stop(self):
        if self.busy:
            self.cancel.set()
            self.btn_stop.state(["disabled"])

    def on_close(self):
        self.cancel.set()
        self.destroy()

    def generate(self):
        if self.busy:
            return
        if uvcore is None:
            messagebox.showerror("Missing dependencies", f"Could not load libraries:\n{IMPORT_ERROR}")
            return
        if not self.files:
            messagebox.showwarning("Nothing to do", "Select one or more models first.")
            return
        if not self.out_dir:
            messagebox.showwarning("No output folder", "Select an output folder first.")
            return

        self.cancel.clear()
        manifest = load_manifest(self.out_dir) if self.resume_var.get() else {}
        todo, keys, resumed = [], {}, 0
        for i, src in enumerate(self.files):
            key = file_key(src)
            keys[i] = key
            prev = manifest.get(key) if key else None
            if prev and (Path(self.out_dir) / prev).exists():
                resumed += 1
                self.table.set(str(i), "status", f"Done (resumed) -> {prev}")
            else:
                self.table.set(str(i), "status", "Queued")
                todo.append((i, src, str(make_output_path(self.out_dir, src))))
        todo.sort(key=lambda j: os.path.getsize(j[1]) if os.path.exists(j[1]) else 0, reverse=True)

        self.busy = True
        for b in (self.btn_select, self.btn_out, self.btn_go, self.chk_resume):
            b.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.set_progress(resumed / len(self.files) * 100)
        threading.Thread(target=self.worker, args=(todo, keys, manifest, resumed), daemon=True).start()

    # ---------- background work ----------
    def worker(self, todo, keys, manifest, resumed):
        ctx = mp.get_context("spawn")
        n = len(self.files)
        out_q = ctx.Queue()
        jobs = deque(todo)
        slots = [new_slot(ctx, out_q) for _ in range(min(MAX_PROCS, len(jobs)))]
        frac = {}
        ok = failed = 0
        last_tick = 0.0

        def push_progress():
            self.ui_q.put(("progress", (resumed + ok + failed + sum(frac.values())) / n * 100))

        def finish(slot, idx, status):
            frac.pop(idx, None)
            slot["job"] = None
            self.ui_q.put(("status", idx, status))
            push_progress()

        try:
            while (jobs or any(s["job"] for s in slots)) and not self.cancel.is_set():
                for s in slots:
                    if s["job"] is None and jobs and s["proc"].is_alive():
                        job = jobs.popleft()
                        s.update(job=job, t0=time.time(), stage="Starting")
                        frac[job[0]] = 0.0
                        s["q"].put(job)

                try:
                    msg = out_q.get(timeout=0.2)
                except queue.Empty:
                    msg = None

                if msg:
                    kind, idx = msg[0], msg[1]
                    slot = next((s for s in slots if s["job"] and s["job"][0] == idx), None)
                    if slot is not None:
                        if kind == "p":
                            frac[idx] = msg[2]
                            slot["stage"] = stage_name(msg[2])
                            last_tick = 0.0
                            push_progress()
                        elif kind == "ok":
                            ok += 1
                            name = Path(msg[2]).name
                            if keys.get(idx):
                                manifest[keys[idx]] = name
                                save_manifest(self.out_dir, manifest)
                            finish(slot, idx, f"Done in {msg[3]:.1f}s -> {name}")
                        else:
                            failed += 1
                            finish(slot, idx, f"FAILED: {msg[2]}")

                # crash detection: worker died while holding a job
                for k, s in enumerate(slots):
                    if s["job"] and not s["proc"].is_alive():
                        failed += 1
                        finish(s, s["job"][0], "FAILED: worker crashed (mesh too broken or too large)")
                        slots[k] = new_slot(ctx, out_q)

                now = time.time()
                if now - last_tick > 0.5:
                    last_tick = now
                    for s in slots:
                        if s["job"]:
                            self.ui_q.put(("status", s["job"][0],
                                           f"{s['stage']}...  {now - s['t0']:.0f}s"))
        finally:
            stopped = self.cancel.is_set()
            for s in slots:
                try:
                    if stopped:
                        s["proc"].terminate()
                    else:
                        s["q"].put(None)
                        s["proc"].join(timeout=2)
                        if s["proc"].is_alive():
                            s["proc"].terminate()
                except Exception:  # noqa: BLE001
                    pass
            if stopped:
                for s in slots:
                    if s["job"]:
                        self.ui_q.put(("status", s["job"][0], "Stopped (run again to resume)"))
                for j in jobs:
                    self.ui_q.put(("status", j[0], "Queued (run again to resume)"))
            self.ui_q.put(("finished", ok, failed, resumed, stopped))

    # ---------- UI updates ----------
    def set_progress(self, pct):
        pct = max(0.0, min(100.0, pct))
        self.bar["value"] = pct
        self.pct_label.config(text=f"{int(pct)}%")

    def poll(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "status":
                    self.table.set(str(msg[1]), "status", msg[2])
                elif msg[0] == "progress":
                    self.set_progress(msg[1])
                elif msg[0] == "finished":
                    self.busy = False
                    for b in (self.btn_select, self.btn_out, self.btn_go, self.chk_resume):
                        b.state(["!disabled"])
                    self.btn_stop.state(["disabled"])
                    if not msg[4]:
                        self.set_progress(100)
                    text = f"Done: {msg[1]}   Skipped (resumed): {msg[3]}   Failed: {msg[2]}"
                    if msg[4]:
                        text += "\n\nStopped early. Run again with Resume ticked to continue."
                    messagebox.showinfo("Finished", text)
        except queue.Empty:
            pass
        self.after(100, self.poll)


if __name__ == "__main__":
    mp.freeze_support()
    if sys.stdout is None or sys.stderr is None:  # windowed .exe has no console
        sys.stdout = sys.stderr = open(os.devnull, "w")
    if IMPORT_ERROR is not None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing dependencies",
            f"{IMPORT_ERROR}\n\nInstall them with:  pip install numpy trimesh xatlas",
        )
        sys.exit(1)
    App().mainloop()

import os
import sys
# Add project root to path so we can import vulkan_nn_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import threading
import os
import sys
import queue
import json
from datetime import datetime

import zipfile
import shutil
import tempfile
import hashlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

class ProjectManager:
    """Manages the .splatproj container (ZIP) and its temporary working directory."""
    def __init__(self, log_callback):
        self.log = log_callback
        self.project_path = None # Path to .splatproj file
        self.work_dir = None # Path to temp working folder
        self.data = self._get_empty_data()
        self._temp_dir_obj = None

    def _get_empty_data(self):
        return {
            "name": "Untitled Project",
            "videos": [], # list of { "path": ..., "name": ..., "size": ..., "mtime": ... }
            "fps_caches": {}, # fps: { "stage": "done/dirty" }
            "completed_stages": [],
            "iterations": 2000,
            "ai_enabled": False,
            "research_mode": "classic",
            "last_updated": ""
        }

    def new_project(self, save_path):
        """Initializes a new project at the given path."""
        self.project_path = save_path
        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="splat_")
        self.work_dir = self._temp_dir_obj.name
        self.data = self._get_empty_data()
        self.data["name"] = os.path.basename(save_path).replace(".splatproj", "")
        
        # Create folder structure in work_dir
        os.makedirs(os.path.join(self.work_dir, "source"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "cache/frames"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "output"), exist_ok=True)
        
        self.save()
        self.log(f"New project created: {save_path}")

    def load_project(self, splatproj_path):
        """Opens an existing .splatproj container."""
        if not zipfile.is_zipfile(splatproj_path):
            raise Exception("Selected file is not a valid .splatproj (ZIP) container.")
            
        self.project_path = splatproj_path
        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="splat_")
        self.work_dir = self._temp_dir_obj.name
        
        with zipfile.ZipFile(splatproj_path, 'r') as zip_ref:
            zip_ref.extractall(self.work_dir)
            
        # Load metadata
        proj_json = os.path.join(self.work_dir, "project.json")
        if os.path.exists(proj_json):
            with open(proj_json, "r") as f:
                self.data = json.load(f)
        
        self.log(f"Project loaded: {os.path.basename(splatproj_path)}")
        return self.data

    def save(self):
        """Syncs all data from work_dir back into the .splatproj ZIP container."""
        if not self.work_dir or not self.project_path: return
        
        self.data["last_updated"] = datetime.now().isoformat()
        with open(os.path.join(self.work_dir, "project.json"), "w") as f:
            json.dump(self.data, f, indent=4)
            
        # Create ZIP from work_dir
        with zipfile.ZipFile(self.project_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(self.work_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, self.work_dir)
                    zip_ref.write(abs_path, rel_path)
        self.log("Project changes synced to container.")

    def add_video(self, video_path, src_fps=30.0):
        """Copies a video into the project and updates metadata."""
        vid_name = os.path.basename(video_path)
        dest = os.path.join(self.work_dir, "source", vid_name)
        
        # Check if already exists in project
        stat = os.stat(video_path)
        meta = {
            "name": vid_name,
            "path": video_path,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "src_fps": src_fps
        }
        
        # Copy to internal source if not already there or changed
        shutil.copy2(video_path, dest)
        
        if meta not in self.data["videos"]:
            self.data["videos"].append(meta)
            # invalidate frames/colmap if input changed
            self.data["fps_caches"] = {} 
            self.data["completed_stages"] = []
            
        self.save()

    def get_frames_dir(self, fps):
        """Returns the specific directory for the requested FPS cache."""
        path = os.path.join(self.work_dir, "cache/frames", str(fps))
        os.makedirs(path, exist_ok=True)
        return path

    def get_video_fps(self, video_path):
        """Uses ffprobe to get the source FPS of a video."""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "json", video_path
        ]
        try:
            result = subprocess.check_output(cmd).decode()
            data = json.loads(result)
            rate_str = data["streams"][0]["r_frame_rate"]
            num, den = map(int, rate_str.split('/'))
            return num / den
        except:
            return 30.0 # Fallback

class SplatEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Splat Studio Professional")
        self.root.geometry("850x700")
        
        self.queue = queue.Queue()
        self.process = None
        self.is_running = False
        
        self.pm = ProjectManager(self.log)
        
        self.setup_ui()
        self.check_queue()
        
    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="SPLAT STUDIO", font=("Helvetica", 28, "bold"), fg="#2196F3")
        header.pack(pady=10)

        # FILE OPS
        file_frame = tk.Frame(self.root)
        file_frame.pack(fill="x", padx=20)
        tk.Button(file_frame, text="New Project", command=self.new_project).pack(side="left", padx=5)
        tk.Button(file_frame, text="Open Project", command=self.load_project_manual).pack(side="left", padx=5)
        self.proj_label = tk.Label(file_frame, text="No Project Loaded", fg="grey")
        self.proj_label.pack(side="left", padx=20)

        # Input Section
        input_frame = tk.LabelFrame(self.root, text="Project Videos (Stored inside .splatproj)", padx=10, pady=10)
        input_frame.pack(fill="x", padx=20, pady=5)

        self.video_listbox = tk.Listbox(input_frame, height=4)
        self.video_listbox.pack(side="left", fill="both", expand=True)

        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(side="right", fill="y", padx=5)

        tk.Button(btn_frame, text="Add Video", command=self.add_videos).pack(fill="x")
        tk.Button(btn_frame, text="Clear Project", command=self.clear_videos, fg="red").pack(fill="x", pady=2)

        # Config Section
        config_frame = tk.LabelFrame(self.root, text="Pipeline Configuration", padx=10, pady=10)
        config_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(config_frame, text="Target FPS (Multi-Cached):").grid(row=0, column=0, sticky="w")
        self.fps_var = tk.IntVar(value=24)
        self.fps_slider = tk.Scale(config_frame, from_=1, to=60, orient="horizontal", variable=self.fps_var, state="disabled")
        self.fps_slider.grid(row=0, column=1, sticky="ew")

        tk.Label(config_frame, text="Training Iterations:").grid(row=1, column=0, sticky="w")
        self.iter_var = tk.IntVar(value=2000)
        tk.Entry(config_frame, textvariable=self.iter_var).grid(row=1, column=1, sticky="w")

        self.ai_var = tk.BooleanVar(value=False)
        tk.Checkbutton(config_frame, text="Enable AI Depth Auto-Density", variable=self.ai_var).grid(row=2, column=0, columnspan=2, sticky="w")

        # Research Section
        research_frame = tk.LabelFrame(self.root, text="Research Reconstruction (Level 10.1 Crystal Nebula Enabled)", padx=10, pady=10)
        research_frame.pack(fill="x", padx=20, pady=5)
        
        self.reconstruct_mode = tk.StringVar(value="nebula")
        modes = [
            ("Method 0: Classic Gaussian", "classic"),
            ("Method A: Orbital Field", "orbital"), 
            ("Method B: Statistical Graph", "graph"), 
            ("Method C: Impresjonista", "impressionist"), 
            ("Method D: Crystal Growth", "crystal"), 
            ("Method E: Crystal Nebula", "nebula")
        ]
        for i, (name, val) in enumerate(modes):
            row = i // 3
            col = i % 3
            tk.Radiobutton(research_frame, text=name, variable=self.reconstruct_mode, value=val).grid(row=row, column=col, sticky="w")
        
        reconstruct_btn = tk.Button(research_frame, text="RUN RESEARCH ON CURRENT SPLATS", command=self.run_reconstruction_manual, bg="#FF9800", fg="white", font=("Helvetica", 9, "bold"))
        reconstruct_btn.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

        # Controls
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)

        self.start_btn = tk.Button(ctrl_frame, text="START PIPELINE", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=20, command=self.start_pipeline)
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(ctrl_frame, text="STOP", font=("Helvetica", 12, "bold"), bg="#f44336", fg="white", width=10, command=self.stop_pipeline, state="disabled")
        self.stop_btn.pack(side="left", padx=10)

        # Progress Section
        progress_frame = tk.LabelFrame(self.root, text="Terminal Output & Progress", padx=10, pady=10)
        progress_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.status_label = tk.Label(progress_frame, text="No Project Active", font=("Helvetica", 10, "italic"))
        self.status_label.pack(anchor="w")

        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", pady=5)

        log_frame = tk.Frame(progress_frame)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10, state="disabled", bg="#1e1e1e", fg="#d4d4d4", font=("Courier", 9))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def new_project(self):
        path = filedialog.asksaveasfilename(defaultextension=".splatproj", filetypes=[("Splat Project", "*.splatproj")])
        if path:
            self.pm.new_project(path)
            self.proj_label.config(text=os.path.basename(path), fg="green")
            self.refresh_ui()

    def add_videos(self):
        if not self.pm.work_dir:
            messagebox.showwarning("Warning", "Create or open a project first.")
            return
        files = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4 *.mov *.avi")])
        for f in files:
            # We fetch FPS before adding to project
            fps = self.pm.get_video_fps(f)
            self.pm.add_video(f, src_fps=fps)
        self.refresh_ui()

    def clear_videos(self):
        if messagebox.askyesno("Confirm", "Clear all videos and reset project progress?"):
            self.pm.data["videos"] = []
            self.pm.data["completed_stages"] = []
            self.pm.data["fps_caches"] = {}
            self.pm.save()
            self.refresh_ui()

    def load_project_manual(self):
        path = filedialog.askopenfilename(filetypes=[("Splat Project", "*.splatproj")])
        if path:
            data = self.pm.load_project(path)
            self.proj_label.config(text=os.path.basename(path), fg="green")
            self.refresh_ui()

    def refresh_ui(self):
        data = self.pm.data
        self.video_listbox.delete(0, tk.END)
        max_fps = 1
        
        videos = data.get("videos", [])
        if not videos:
            self.fps_slider.config(state="disabled")
        else:
            self.fps_slider.config(state="normal")
            for v in videos:
                self.video_listbox.insert(tk.END, v["name"])
                # Get max FPS from video metadata
                v_fps = v.get("src_fps", 30.0)
                if v_fps > max_fps: max_fps = int(v_fps)
            
            # Update slider limit
            self.fps_slider.config(to=max_fps)
            if self.fps_var.get() > max_fps:
                self.fps_var.set(max_fps)

        self.fps_var.set(data.get("fps", min(24, max_fps)))
        self.iter_var.set(data.get("iterations", 2000))
        self.ai_var.set(data.get("ai_enabled", False))
        self.reconstruct_mode.set(data.get("research_mode", "nebula"))

    def log(self, message):
        self.queue.put(("log", message + "\n"))

    def set_status(self, status, progress=None):
        self.queue.put(("status", (status, progress)))

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, data)
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
                elif msg_type == "status":
                    status, progress = data
                    self.status_label.config(text=status)
                    if progress is not None:
                        self.progress["value"] = progress
                elif msg_type == "done":
                    self.pipeline_finished()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def run_command(self, cmd, status_prefix, progress_start, progress_end, cwd=None):
        self.log(f"Running: {' '.join(cmd)}")
        env = os.environ.copy()
        # Ensure subprocesses can find vulkan_nn_lib even if cwd changes
        env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                        text=True, bufsize=1, cwd=cwd or self.pm.work_dir, env=env)
        
        for line in self.process.stdout:
            self.log(line.strip())
            # Progress heuristics
            if "Iteration" in line:
                try:
                    parts = line.split("/")
                    current = int(parts[0].split()[-1])
                    total = int(parts[1].split()[0])
                    p = progress_start + (current / total) * (progress_end - progress_start)
                    self.set_status(f"{status_prefix} ({current}/{total})", p)
                except: pass
            else:
                self.set_status(status_prefix)

        self.process.wait()
        if self.process.returncode != 0:
            raise Exception(f"Command failed with return code {self.process.returncode}")

    def pipeline_finished(self):
        self.is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.set_status("Pipeline Completed!", 100)
        self.pm.save()
        
        if messagebox.askyesno("Success", "Processing finished! Export results?"):
            save_path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY files", "*.ply")])
            if save_path:
                src = os.path.join(self.pm.work_dir, "output/trained_splats.ply")
                if os.path.exists(src):
                    shutil.copy(src, save_path)
                    self.log(f"Model exported to {save_path}")
                
                subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "train_gs.py"), "--view"], cwd=self.pm.work_dir)
            
    def run_reconstruction_manual(self):
        if not self.pm.work_dir: return
        mode = self.reconstruct_mode.get()
        splat_path = os.path.join(self.pm.work_dir, "output/trained_splats.ply")
        if not os.path.exists(splat_path):
            messagebox.showwarning("Incomplete", "Run Training phase first.")
            return
        
        self.log(f"Starting {mode} research reconstruction...")
        threading.Thread(target=self.run_reconstruction_thread, args=(mode,), daemon=True).start()

    def run_reconstruction_thread(self, mode):
        try:
            self.set_status(f"Researching {mode}...", 0)
            out_file = f"output/research_{mode}.ply"
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "gs_to_mesh.py"), "--input", "output/trained_splats.ply", "--mode", mode, "--output", out_file, "--live"]
            self.run_command(cmd, f"Reconstructing ({mode})", 0, 100)
            self.pm.save()
            self.log(f"Success! Result synced to .splatproj")
            self.set_status("Reconstruction Done!", 100)
            
            if messagebox.askyesno("Complete", f"Generated. Open Viewer?"):
                subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "view_mesh.py"), "--input", out_file], cwd=self.pm.work_dir)
        except Exception as e:
            self.log(f"Failed: {e}")

    def stop_pipeline(self):
        if self.process:
            self.process.terminate()
            self.log("\n!!! Stopped by user !!!")
            self.set_status("Stopped", 0)
        self.is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def start_pipeline(self):
        if not self.pm.work_dir or not self.pm.data["videos"]:
            messagebox.showwarning("Warning", "Project is empty.")
            return
        
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        # Save current config to project meta
        self.pm.data.update({
            "fps": self.fps_var.get(),
            "iterations": self.iter_var.get(),
            "ai_enabled": self.ai_var.get(),
            "research_mode": self.reconstruct_mode.get()
        })
        self.pm.save()
        threading.Thread(target=self.run_pipeline_thread, daemon=True).start()

    def run_pipeline_thread(self):
        try:
            w = self.pm.work_dir
            fps = self.fps_var.get()
            frames_dir = self.pm.get_frames_dir(fps)
            
            # Check for existing work
            has_frames = len(os.listdir(frames_dir)) > 0
            has_colmap = os.path.exists(os.path.join(w, "cache/colmap/sparse/0/points3D.bin"))
            has_training = os.path.exists(os.path.join(w, "output/trained_splats.ply"))

            # 1. Extraction (Multi-FPS aware)
            if has_frames:
                self.log(f"Cache hit: Found existing extraction for {fps} FPS. Skipping Stage 1.")
            else:
                self.set_status("Stage 1/4: Extracting (New FPS Cache)...", 0)
                # Use internal source videos
                src_vids = [os.path.join(w, "source", v["name"]) for v in self.pm.data["videos"]]
                for i, vid in enumerate(src_vids):
                    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "extract_frames.py"), "--video", vid, "--output", frames_dir, "--fps", str(fps), "--prefix", f"v{i}"]
                    self.run_command(cmd, f"Extracting {os.path.basename(vid)}", (i/len(src_vids))*20, ((i+1)/len(src_vids))*20)
                self.pm.data["fps_caches"][str(fps)] = "frames_done"
                self.pm.save()

            # 2. COLMAP
            colmap_path = os.path.join(w, "cache/colmap")
            if has_colmap:
                self.log("Cache hit: COLMAP already exists. Skipping Stage 2.")
            else:
                self.set_status("Stage 2/4: SfM Reconstruction...", 20)
                os.makedirs(colmap_path, exist_ok=True)
                cmd = [sys.executable, os.path.join(SCRIPT_DIR, "run_colmap.py"), "--images", frames_dir, "--output", colmap_path]
                self.run_command(cmd, "COLMAP", 20, 45)
                self.pm.save()

            # 3. AI (If enabled)
            if self.ai_var.get():
                ai_out = os.path.join(colmap_path, "sparse/0/ai_points.ply")
                if os.path.exists(ai_out):
                    self.log("Cache hit: AI points found. Skipping Stage 3.")
                else:
                    self.set_status("Stage 3/4: AI Depth Enhancement...", 45)
                    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "align_depth.py"), "--colmap_path", f"cache/colmap/sparse/0", "--img_path", f"cache/frames/{fps}"]
                    self.run_command(cmd, "AI Depth", 45, 75)
                    self.pm.save()

            # 4. Training
            if has_training:
                self.log("Note: Training result exists. Re-training with current iterations...")
            
            self.set_status("Stage 4/4: GS Training...", 75)
            # train_gs.py writes to output/
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_gs.py"), "--colmap_path", "cache/colmap/sparse/0", "--iterations", str(self.iter_var.get())]
            self.run_command(cmd, "Training", 75, 100)
            
            self.pm.data["completed_stages"].append("training")
            self.pm.save()
            self.queue.put(("done", None))
            
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            self.set_status(f"Error: {e}", 0)

if __name__ == "__main__":
    root = tk.Tk()
    app = SplatEditor(root)
    root.mainloop()

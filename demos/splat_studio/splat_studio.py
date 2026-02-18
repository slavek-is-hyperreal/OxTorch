import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import threading
import queue
import json
from datetime import datetime
import signal
import zipfile
import shutil
import tempfile
import hashlib

# Add project root to path so we can import vulkan_nn_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vulkan_nn_lib.memory import MemoryManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

class ProjectManager:
    """Manages the .splatproj container (ZIP) and its temporary working directory."""
    def __init__(self, log_callback):
        self.log = log_callback
        self.project_path = None
        self.work_dir = None
        self.data = self._get_empty_data()
        self._temp_dir_obj = None

    def _get_empty_data(self):
        return {
            "name": "Untitled Project",
            "videos": [],
            "fps_caches": {},
            "completed_stages": [],
            "iterations": 2000,
            "ai_enabled": False,
            "research_mode": "nebula",
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
        """Opens an existing .splatproj container lazily."""
        if not zipfile.is_zipfile(splatproj_path):
            raise Exception("Selected file is not a valid .splatproj (ZIP) container.")
            
        self.project_path = splatproj_path
        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="splat_")
        self.work_dir = self._temp_dir_obj.name
        
        # Only extract metadata initially
        with zipfile.ZipFile(splatproj_path, 'r') as zip_ref:
            if "project.json" in zip_ref.namelist():
                zip_ref.extract("project.json", self.work_dir)
            
        # Ensure minimal directory structure
        os.makedirs(os.path.join(self.work_dir, "source"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "cache/frames"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "output"), exist_ok=True)
            
        # Load metadata
        proj_json = os.path.join(self.work_dir, "project.json")
        if os.path.exists(proj_json):
            with open(proj_json, "r") as f:
                self.data = json.load(f)
        
        self.log(f"Project opened (Lazy Mode): {os.path.basename(splatproj_path)}")
        return self.data

    def exists_in_zip(self, folder_prefix):
        """Checks if a folder/prefix exists in the ZIP container without extracting."""
        if not self.project_path or not os.path.exists(self.project_path):
            return False
        with zipfile.ZipFile(self.project_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.startswith(folder_prefix):
                    return True
        return False

    def get_file(self, rel_path):
        """Returns the local SSD path for a project file, extracting it if needed."""
        if not self.work_dir: return None
        dest = os.path.join(self.work_dir, rel_path)
        
        if not os.path.exists(dest):
            if self.project_path and zipfile.is_zipfile(self.project_path):
                with zipfile.ZipFile(self.project_path, 'r') as zip_ref:
                    if rel_path in zip_ref.namelist():
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        zip_ref.extract(rel_path, self.work_dir)
                        self.log(f"Streamed: {rel_path} -> SSD")
        
        return dest if os.path.exists(dest) else None

    def ensure_extracted(self, folder_prefix):
        """Ensures a specific folder is extracted from the container."""
        if not self.project_path or not os.path.exists(self.project_path):
            return
            
        target_path = os.path.join(self.work_dir, folder_prefix)
        # If the path is a file, check if it exists
        if os.path.isfile(target_path):
            return
        # If the path is a directory, check if it exists and has content
        if os.path.isdir(target_path) and os.listdir(target_path):
            return
            
        self.log(f"Extracting {folder_prefix} from project container...")
        with zipfile.ZipFile(self.project_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.startswith(folder_prefix):
                    zip_ref.extract(member, self.work_dir)

    def save(self):
        """Syncs data back, merging work_dir with original ZIP to preserve unextracted files."""
        if not self.work_dir or not self.project_path: return
        
        self.data["last_updated"] = datetime.now().isoformat()
        with open(os.path.join(self.work_dir, "project.json"), "w") as f:
            json.dump(self.data, f, indent=4)
            
        # Non-destructive save: merge work_dir with what's already in the ZIP
        temp_proj = self.project_path + ".tmp"
        
        # Files currently in work_dir
        work_files = {}
        for root, dirs, files in os.walk(self.work_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.work_dir)
                work_files[rel_path] = abs_path
                
        # Create new ZIP mirroring original + work_dir updates
        file_count = len(work_files)
        self.log(f"Syncing {file_count} files to project container...")
        
        with zipfile.ZipFile(temp_proj, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            # 1. Write everything from work_dir
            for i, (rel, abs_p) in enumerate(work_files.items()):
                new_zip.write(abs_p, rel)
                if i % 100 == 0 and i > 0:
                    self.log(f"Writing updated assets... ({i}/{file_count})")
                
            # 2. Extract and write everything from original ZIP that is NOT in work_dir
            if os.path.exists(self.project_path):
                with zipfile.ZipFile(self.project_path, 'r') as old_zip:
                    old_info = old_zip.infolist()
                    for item in old_info:
                        if item.filename not in work_files:
                            new_zip.writestr(item, old_zip.read(item.filename))
                            
        os.replace(temp_proj, self.project_path)
        self.log("Project changes synced to container (Preserving unextracted assets).")

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

class SplatEngine:
    """Handles the heavy lifting: COLMAP, Training, and Reconstruction."""
    def __init__(self, project_manager, log_callback, status_callback):
        self.pm = project_manager
        self.log = log_callback
        self.set_status = status_callback
        self.process = None
        self.is_running = False

    def run_command(self, cmd, status_prefix, progress_start, progress_end, cwd=None):
        self.log(f"Running: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, cwd=cwd or self.pm.work_dir, env=env,
            preexec_fn=os.setsid
        )
        
        for line in self.process.stdout:
            self.log(line.strip())
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

    def stop(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except:
                self.process.terminate()
            self.log("\n!!! Stopped !!!")
        self.is_running = False

    def run_pipeline(self, fps, iterations, ai_enabled):
        try:
            self.is_running = True
            self.set_status("Initializing Project State...", 0)
            self.log("Consolidating project data and merging assets...")
            self.pm.save()
            w = self.pm.work_dir
            frames_dir = self.pm.get_frames_dir(fps)
            
            # Detect existing work
            self.log("Checking project cache status...")
            has_frames = os.path.exists(frames_dir) and (os.listdir(frames_dir) or self.pm.exists_in_zip(f"cache/frames/{fps}"))
            has_colmap = os.path.exists(os.path.join(w, "cache/colmap/sparse/0/points3D.bin")) or \
                         self.pm.exists_in_zip("cache/colmap/sparse/0/points3D.bin")
            has_training = os.path.exists(os.path.join(w, "output/trained_splats.ply")) or \
                           self.pm.exists_in_zip("output/trained_splats.ply")
            
            self.log(f" - Frames cache: {'FOUND' if has_frames else 'MISSING'}")
            self.log(f" - SfM cache: {'FOUND' if has_colmap else 'MISSING'}")
            self.log(f" - Training cache: {'FOUND' if has_training else 'MISSING'}")

            # 1. Extraction
            if has_frames:
                self.log(f"Cache hit: Found frames for {fps} FPS.")
            else:
                self.pm.ensure_extracted("source")
                self.set_status("Stage 1/4: Extracting Frames...", 0)
                src_vids = [os.path.join(w, "source", v["name"]) for v in self.pm.data["videos"]]
                for i, vid in enumerate(src_vids):
                    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "extract_frames.py"), "--video", vid, "--output", frames_dir, "--fps", str(fps), "--prefix", f"v{i}"]
                    self.run_command(cmd, f"Extracting {os.path.basename(vid)}", (i/len(src_vids))*20, ((i+1)/len(src_vids))*20)
                self.pm.save()

            # 2. COLMAP
            colmap_path = os.path.join(w, "cache/colmap")
            if has_colmap:
                self.log("Cache hit: COLMAP exists.")
            else:
                self.pm.ensure_extracted(f"cache/frames/{fps}")
                self.set_status("Stage 2/4: SfM Reconstruction...", 20)
                os.makedirs(colmap_path, exist_ok=True)
                cmd = [sys.executable, os.path.join(SCRIPT_DIR, "run_colmap.py"), "--images", frames_dir, "--output", colmap_path]
                self.run_command(cmd, "COLMAP", 20, 45)
                self.pm.save()

            # 3. AI Depth
            if ai_enabled:
                ai_out = os.path.join(colmap_path, "sparse/0/ai_points.ply")
                if os.path.exists(ai_out) or self.pm.exists_in_zip("cache/colmap/sparse/0/ai_points.ply"):
                    self.log("Cache hit: AI points found.")
                else:
                    self.pm.ensure_extracted("cache/colmap")
                    self.pm.ensure_extracted(f"cache/frames/{fps}")
                    self.set_status("Stage 3/4: AI Depth Enhancement...", 45)
                    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "align_depth.py"), "--colmap_path", "cache/colmap/sparse/0", "--img_path", f"cache/frames/{fps}", "--output", ai_out]
                    self.run_command(cmd, "AI Depth", 45, 75)
                    self.pm.save()

            # 4. Training
            self.pm.ensure_extracted("cache/colmap")
            self.pm.ensure_extracted(f"cache/frames/{fps}")
            self.set_status("Stage 4/4: GS Training...", 75)
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_gs.py"), "--colmap_path", "cache/colmap/sparse/0", "--img_path", f"cache/frames/{fps}", "--iterations", str(iterations)]
            self.run_command(cmd, "Training", 75, 100)
            
            self.pm.data["completed_stages"].append("training")
            self.pm.save()
            return True
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            return False
        finally:
            self.is_running = False

    def run_reconstruction(self, mode):
        try:
            self.is_running = True
            self.set_status(f"Researching {mode}...", 0)
            self.pm.ensure_extracted("output")
            out_file = f"output/research_{mode}.ply"
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "gs_to_mesh.py"), "--input", "output/trained_splats.ply", "--mode", mode, "--output", out_file, "--live"]
            self.run_command(cmd, f"Reconstructing ({mode})", 0, 100)
            self.pm.save()
            return out_file
        except Exception as e:
            self.log(f"Failed: {e}")
            return None
        finally:
            self.is_running = False

class SplatEditor:
    def __init__(self, root, verbose=False):
        self.root = root
        self.verbose = verbose
        self.root.title("Splat Projekt")
        self.root.geometry("850x700")
        
        self.queue = queue.Queue()
        self.is_loading = False
        
        self.pm = ProjectManager(self.log)
        self.engine = SplatEngine(self.pm, self.log, self.set_status)
        
        self.setup_ui()
        self.refresh_ui()
        self.check_queue()
        self.update_hardware_health()
        
    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="SPLAT PROJEKT", font=("Helvetica", 28, "bold"), fg="#2196F3")
        header.pack(pady=10)

        # FILE OPS
        self.file_frame = tk.Frame(self.root)
        self.file_frame.pack(fill="x", padx=20)
        self.new_btn = tk.Button(self.file_frame, text="New Project", command=self.new_project)
        self.new_btn.pack(side="left", padx=5)
        self.open_btn = tk.Button(self.file_frame, text="Open Project", command=self.load_project_manual)
        self.open_btn.pack(side="left", padx=5)
        self.proj_label = tk.Label(self.file_frame, text="No Project Loaded", fg="grey")
        self.proj_label.pack(side="left", padx=20)

        # Input Section
        self.input_frame = tk.LabelFrame(self.root, text="Project Videos (Stored inside .splatproj)", padx=10, pady=10)
        self.input_frame.pack(fill="x", padx=20, pady=5)

        self.video_listbox = tk.Listbox(self.input_frame, height=4)
        self.video_listbox.pack(side="left", fill="both", expand=True)

        btn_frame = tk.Frame(self.input_frame)
        btn_frame.pack(side="right", fill="y", padx=5)

        tk.Button(btn_frame, text="Add Video", command=self.add_videos).pack(fill="x")
        tk.Button(btn_frame, text="Clear Project", command=self.clear_videos, fg="red").pack(fill="x", pady=2)

        # Config Section
        self.config_frame = tk.LabelFrame(self.root, text="Pipeline Configuration", padx=10, pady=10)
        self.config_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(self.config_frame, text="Target FPS (Multi-Cached):").grid(row=0, column=0, sticky="w")
        self.fps_var = tk.IntVar(value=24)
        self.fps_slider = tk.Scale(self.config_frame, from_=1, to=60, orient="horizontal", variable=self.fps_var, state="disabled")
        self.fps_slider.grid(row=0, column=1, sticky="ew")

        tk.Label(self.config_frame, text="Training Iterations:").grid(row=1, column=0, sticky="w")
        self.iter_var = tk.IntVar(value=2000)
        tk.Entry(self.config_frame, textvariable=self.iter_var).grid(row=1, column=1, sticky="w")

        self.ai_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.config_frame, text="Enable AI Depth Auto-Density", variable=self.ai_var).grid(row=2, column=0, columnspan=2, sticky="w")

        # Research Section
        self.research_frame = tk.LabelFrame(self.root, text="Research Reconstruction (Level 10.1 Crystal Nebula Enabled)", padx=10, pady=10)
        self.research_frame.pack(fill="x", padx=20, pady=5)
        
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
            tk.Radiobutton(self.research_frame, text=name, variable=self.reconstruct_mode, value=val).grid(row=row, column=col, sticky="w")
        
        reconstruct_btn = tk.Button(self.research_frame, text="RUN RESEARCH ON CURRENT SPLATS", command=self.run_reconstruction_manual, bg="#FF9800", fg="white", font=("Helvetica", 9, "bold"))
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

        # Hardware Health / RAM Risk (DRAS v4)
        health_frame = tk.Frame(progress_frame)
        health_frame.pack(fill="x", pady=2)
        tk.Label(health_frame, text="Hardware Health:", font=("Helvetica", 9)).pack(side="left")
        self.health_bar = ttk.Progressbar(health_frame, orient="horizontal", length=200, mode="determinate")
        self.health_bar.pack(side="left", padx=10)
        self.health_status = tk.Label(health_frame, text="Safe", font=("Helvetica", 9, "bold"), fg="green")
        self.health_status.pack(side="left")

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
        self.save_project()
        self.refresh_ui()

    def clear_videos(self):
        if messagebox.askyesno("Confirm", "Clear all videos and reset project progress?"):
            self.pm.data["videos"] = []
            self.pm.data["completed_stages"] = []
            self.pm.data["fps_caches"] = {}
            self.save_project()
            self.refresh_ui()

    def load_project_manual(self):
        path = filedialog.askopenfilename(filetypes=[("Splat Project", "*.splatproj")])
        if path:
            self.is_loading = True
            self.set_status(f"Loading {os.path.basename(path)}...", 0)
            self.refresh_ui() 
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state="disabled")
            threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _load_worker(self, path):
        try:
            self.pm.load_project(path)
            self.queue.put(("loaded", path))
        except Exception as e:
            self.queue.put(("error", str(e)))

    def save_project(self):
        """Threaded save wrapper."""
        threading.Thread(target=self._save_worker, daemon=True).start()

    def _save_worker(self):
        try:
            self.is_loading = True
            self.refresh_ui()
            self.set_status("Saving project...", None)
            self.pm.save()
            self.log("Project saved to container.")
            self.set_status("Project Saved.", 100)
        finally:
            self.is_loading = False
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
        
        has_project = self.pm.work_dir is not None
        has_data = False
        if has_project:
            splats_done = os.path.exists(os.path.join(self.pm.work_dir, "output/trained_splats.ply")) or \
                          self.pm.exists_in_zip("output/trained_splats.ply")
            has_data = splats_done

        if self.is_loading:
            self.set_ui_state(self.file_frame, "disabled")
            self.set_ui_state(self.input_frame, "disabled")
            self.set_ui_state(self.config_frame, "disabled")
            self.set_ui_state(self.research_frame, "disabled")
            self.start_btn.config(state="disabled")
            return

        self.set_ui_state(self.file_frame, "normal")
        self.set_ui_state(self.input_frame, "normal" if has_project else "disabled")
        self.set_ui_state(self.config_frame, "normal" if has_project else "disabled")
        self.set_ui_state(self.research_frame, "normal" if has_data else "disabled")
        
        is_running = self.engine.is_running
        self.start_btn.config(state="normal" if has_project and not is_running else "disabled")
        self.stop_btn.config(state="normal" if is_running else "disabled")

    def set_ui_state(self, widget, state):
        """Recursively sets state for a widget and its children."""
        try:
            widget.configure(state=state)
        except: pass
        for child in widget.winfo_children():
            self.set_ui_state(child, state)

    def log(self, message):
        self.queue.put(("log", message + "\n"))
        if self.verbose:
            print(message)

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
                elif msg_type == "loaded":
                    path = data
                    self.is_loading = False
                    self.proj_label.config(text=os.path.basename(path), fg="green")
                    self.refresh_ui()
                    self.set_status("Project loaded.", 100)
                elif msg_type == "error":
                    self.is_loading = False
                    messagebox.showerror("Error", data)
                    self.refresh_ui()
                    self.set_status(f"Error: {data}", 0)
                elif msg_type == "done":
                    self.pipeline_finished()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)
        
        has_project = self.pm.work_dir is not None
        is_running = self.engine.is_running
        self.start_btn.config(state="normal" if has_project and not is_running else "disabled")
        self.stop_btn.config(state="normal" if is_running else "disabled")

    def update_hardware_health(self):
        """Update the DRAS v4 RAM risk indicator."""
        try:
            risk = MemoryManager.get_usage_risk()
            percent = int(risk * 100)
            self.health_bar["value"] = percent
            
            if risk < 0.6:
                self.health_status.config(text="Safe", fg="green")
            elif risk < 0.85:
                self.health_status.config(text="Warning", fg="orange")
            else:
                self.health_status.config(text="CRITICAL", fg="red")
        except: pass
        self.root.after(1000, self.update_hardware_health)

    def start_pipeline(self):
        if not self.pm.work_dir or not self.pm.data["videos"]:
            messagebox.showwarning("Warning", "Project is empty.")
            return
        
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        self.pm.data.update({
            "fps": self.fps_var.get(),
            "iterations": self.iter_var.get(),
            "ai_enabled": self.ai_var.get(),
            "research_mode": self.reconstruct_mode.get()
        })
        threading.Thread(target=self._run_pipeline_worker, daemon=True).start()
        self.refresh_ui()

    def _run_pipeline_worker(self):
        fps = self.fps_var.get()
        iters = self.iter_var.get()
        ai = self.ai_var.get()
        if self.engine.run_pipeline(fps, iters, ai):
            self.queue.put(("done", None))
        else:
            self.queue.put(("error", "Pipeline failed. Check logs."))

    def pipeline_finished(self):
        self.refresh_ui()
        self.set_status("Pipeline Completed!", 100)
        self.log("Training finished. Results saved inside .splatproj container.")
        
        # Automatically launch viewer
        fps = self.fps_var.get()
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_gs.py"), "--view", "--colmap_path", "cache/colmap/sparse/0", "--img_path", f"cache/frames/{fps}"]
        subprocess.Popen(cmd, cwd=self.pm.work_dir)

    def run_reconstruction_manual(self):
        if not self.pm.work_dir: return
        splat_path = os.path.join(self.pm.work_dir, "output/trained_splats.ply")
        if not os.path.exists(splat_path) and not self.pm.exists_in_zip("output/trained_splats.ply"):
            messagebox.showwarning("Incomplete", "Run Training phase first.")
            return
        
        mode = self.reconstruct_mode.get()
        threading.Thread(target=self._run_reconstruct_worker, args=(mode,), daemon=True).start()

    def _run_reconstruct_worker(self, mode):
        out_file = self.engine.run_reconstruction(mode)
        if out_file:
            self.log(f"Reconstruction ({mode}) saved inside .splatproj.")
            # Automatically launch viewer for research result
            subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "view_mesh.py"), "--input", out_file], cwd=self.pm.work_dir)
        self.queue.put(("status", ("Reconstruction Done", 100)))

    def stop_pipeline(self):
        self.engine.stop()
        self.set_status("Stopped", 0)
        self.refresh_ui()

class SplatCLI:
    def __init__(self, project_path, verbose=False):
        self.verbose = verbose
        self.pm = ProjectManager(self.log)
        self.engine = SplatEngine(self.pm, self.log, self.set_status)
        self.pm.load_project(project_path)

    def log(self, message):
        print(message if not message.endswith("\n") else message[:-1], flush=True)

    def set_status(self, status, progress=None):
        if self.verbose:
            prog_str = f" [{progress}%]" if progress is not None else ""
            print(f"STATUS: {status}{prog_str}", flush=True)

    def start_pipeline(self):
        print(f"--- Starting Pipeline for {self.pm.data['name']} ---")
        fps = self.pm.data.get("fps", 24)
        iters = self.pm.data.get("iterations", 2000)
        ai = self.pm.data.get("ai_enabled", False)
        self.engine.run_pipeline(fps, iters, ai)
        print("--- Pipeline Finished ---")

    def run_reconstruction(self, mode=None):
        mode = mode or self.pm.data.get("research_mode", "nebula")
        print(f"--- Starting Reconstruction ({mode}) ---")
        self.engine.run_reconstruction(mode)
        print("--- Reconstruction Finished ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Splat Projekt")
    parser.add_argument("--project", type=str, help="Path to .splatproj file (Starts CLI mode)")
    parser.add_argument("--action", type=str, choices=["run", "reconstruct"], help="Action to perform in CLI mode")
    parser.add_argument("--mode", type=str, help="Research mode for reconstruction")
    parser.add_argument("--verbose", action="store_true", help="Mirror logs to terminal")
    args = parser.parse_args()

    if args.project:
        cli = SplatCLI(args.project, verbose=args.verbose)
        if args.action == "run": cli.start_pipeline()
        elif args.action == "reconstruct": cli.run_reconstruction(args.mode)
        else: print("Please specify --action [run|reconstruct]")
    else:
        root = tk.Tk()
        app = SplatEditor(root, verbose=args.verbose)
        root.mainloop()

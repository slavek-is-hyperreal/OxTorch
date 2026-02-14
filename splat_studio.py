import os
import sys
import queue
import json
from datetime import datetime

class SplatEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Splat Editor - 3D Gaussian Splatting Pipeline")
        self.root.geometry("800x650")
        
        self.queue = queue.Queue()
        self.process = None
        self.is_running = False
        self.project_data = {
            "videos": [],
            "fps": 24,
            "iterations": 2000,
            "ai_enabled": True,
            "completed_stages": []
        }
        self.project_file = "output/project.json"
        
        self.setup_ui()
        self.check_queue()
        self.load_project_auto()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Splat Editor", font=("Helvetica", 24, "bold"))
        header.pack(pady=10)

        # Input Section
        input_frame = tk.LabelFrame(self.root, text="Input Videos", padx=10, pady=10)
        input_frame.pack(fill="x", padx=20, pady=5)

        self.video_paths = []
        self.video_listbox = tk.Listbox(input_frame, height=5)
        self.video_listbox.pack(side="left", fill="both", expand=True)

        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(side="right", fill="y", padx=5)

        tk.Button(btn_frame, text="Add Videos", command=self.add_videos).pack(fill="x")
        tk.Button(btn_frame, text="Clear", command=self.clear_videos).pack(fill="x")
        tk.Button(btn_frame, text="Load Project", command=self.load_project_manual, bg="#2196F3", fg="white").pack(fill="x", pady=5)

        # Config Section
        config_frame = tk.LabelFrame(self.root, text="Configuration", padx=10, pady=10)
        config_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(config_frame, text="Frames per second:").grid(row=0, column=0, sticky="w")
        self.fps_var = tk.IntVar(value=24)
        tk.Scale(config_frame, from_=1, to=60, orient="horizontal", variable=self.fps_var).grid(row=0, column=1, sticky="ew")

        tk.Label(config_frame, text="Training Iterations:").grid(row=1, column=0, sticky="w")
        self.iter_var = tk.IntVar(value=2000)
        tk.Entry(config_frame, textvariable=self.iter_var).grid(row=1, column=1, sticky="w")

        # AI Enhancement Toggle
        self.ai_var = tk.BooleanVar(value=True)
        tk.Checkbutton(config_frame, text="Enable AI Depth Auto-Density (CPU)", variable=self.ai_var, font=("Helvetica", 10, "bold")).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        # Controls
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)

        self.start_btn = tk.Button(ctrl_frame, text="START PIPELINE", font=("Helvetica", 12, "bold"), 
                                   bg="#4CAF50", fg="white", padx=20, pady=10, command=self.start_pipeline)
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(ctrl_frame, text="STOP", font=("Helvetica", 12, "bold"), 
                                  bg="#f44336", fg="white", padx=20, pady=10, command=self.stop_pipeline, state="disabled")
        self.stop_btn.pack(side="left", padx=10)

        # Progress Section
        progress_frame = tk.LabelFrame(self.root, text="Progress & Logs", padx=10, pady=10)
        progress_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.status_label = tk.Label(progress_frame, text="Ready", font=("Helvetica", 10, "italic"))
        self.status_label.pack(anchor="w")

        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", pady=5)

        # Log with Scrollbar
        log_frame = tk.Frame(progress_frame)
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(log_frame, height=8, state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_videos(self):
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if files:
            for f in files:
                if f not in self.video_paths:
                    self.video_paths.append(f)
                    self.video_listbox.insert(tk.END, os.path.basename(f))

    def clear_videos(self):
        self.video_paths = []
        self.video_listbox.delete(0, tk.END)

    def save_project(self):
        self.project_data.update({
            "videos": self.video_paths,
            "fps": self.fps_var.get(),
            "iterations": self.iter_var.get(),
            "ai_enabled": self.ai_var.get(),
            "last_updated": datetime.now().isoformat()
        })
        os.makedirs("output", exist_ok=True)
        with open(self.project_file, "w") as f:
            json.dump(self.project_data, f, indent=4)
        self.log(f"Project state saved to {self.project_file}")

    def load_project_auto(self):
        if os.path.exists(self.project_file):
            try:
                with open(self.project_file, "r") as f:
                    data = json.load(f)
                self.apply_project_data(data)
                self.log("Automatic Recovery: Previous project session loaded.")
            except Exception as e:
                self.log(f"Failed to auto-load project: {e}")

    def load_project_manual(self):
        file_path = filedialog.askopenfilename(filetypes=[("Project JSON", "*.json")])
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                self.apply_project_data(data)
                self.project_file = file_path
                self.log(f"Project loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load project: {e}")

    def apply_project_data(self, data):
        self.project_data = data
        self.video_paths = data.get("videos", [])
        self.video_listbox.delete(0, tk.END)
        for v in self.video_paths:
            self.video_listbox.insert(tk.END, os.path.basename(v))
        self.fps_var.set(data.get("fps", 24))
        self.iter_var.set(data.get("iterations", 2000))
        self.ai_var.set(data.get("ai_enabled", True))

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

    def run_command(self, cmd, status_prefix, progress_start, progress_end):
        self.log(f"Running: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in self.process.stdout:
            self.log(line.strip())
            # Simple progress heuristics for each stage
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
        
        if messagebox.askyesno("Success", "Processing finished! Export model and launch viewer?"):
            save_path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY files", "*.ply")])
            if save_path:
                if os.path.exists("output/trained_splats.ply"):
                    import shutil
                    shutil.copy("output/trained_splats.ply", save_path)
                    self.log(f"Model saved to {save_path}")
                
            # Launch viewer
            subprocess.Popen([sys.executable, "train_gs.py", "--view"])

    def stop_pipeline(self):
        if self.process:
            self.process.terminate()
            self.log("\n!!! Pipeline stopped by user !!!")
            self.set_status("Stopped", 0)
        self.is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def start_pipeline(self):
        if not self.video_paths:
            messagebox.showwarning("Warning", "Please add at least one video.")
            return
        
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        self.save_project()
        threading.Thread(target=self.run_pipeline_thread, daemon=True).start()

    def run_pipeline_thread(self):
        try:
            # Check for completed stages
            colmap_done = os.path.exists("output/colmap/sparse/0/points3D.bin")
            frames_done = os.path.exists("output/frames") and len(os.listdir("output/frames")) > 0
            ai_done = os.path.exists("output/colmap/sparse/0/ai_points.ply")

            # 1. Extraction
            if frames_done and messagebox.askyesno("Resume", "Frames already exist. Skip extraction?"):
                self.log("Skipping Stage 1: Frames already present.")
            else:
                self.set_status("Stage 1/4: Extracting Frames...", 0)
                os.makedirs("output/frames", exist_ok=True)
                for i, vid in enumerate(self.video_paths):
                    cmd = [sys.executable, "extract_frames.py", "--video", vid, "--output", "output/frames", "--fps", str(self.fps_var.get()), "--prefix", f"v{i}"]
                    self.run_command(cmd, f"Extracting {os.path.basename(vid)}", (i/len(self.video_paths))*20, ((i+1)/len(self.video_paths))*20)
            
            self.project_data.setdefault("completed_stages", []).append("extraction")
            self.save_project()

            # 2. COLMAP
            if colmap_done and messagebox.askyesno("Resume", "COLMAP reconstruction already exists. Skip Stage 2?"):
                self.log("Skipping Stage 2: COLMAP data found.")
            else:
                self.set_status("Stage 2/4: COLMAP Reconstruction (SfM)...", 20)
                cmd = [sys.executable, "run_colmap.py", "--images", "output/frames", "--output", "output/colmap"]
                self.run_command(cmd, "COLMAP Reconstruction", 20, 45)
            
            self.project_data["completed_stages"].append("colmap")
            self.save_project()

            # 3. AI Depth Enhancement (Optional)
            if self.ai_var.get():
                if ai_done and messagebox.askyesno("Resume", "AI enhanced points already exist. Skip Stage 3?"):
                    self.log("Skipping Stage 3: AI points found.")
                else:
                    self.set_status("Stage 3/4: AI Depth Enhancement (CPU)...", 45)
                    cmd = [sys.executable, "align_depth.py", "--colmap_path", "output/colmap/sparse/0", "--img_path", "output/frames"]
                    self.run_command(cmd, "AI Depth Processing", 45, 75)
                
                self.project_data["completed_stages"].append("ai_depth")
                self.save_project()

            # 4. Training
            self.set_status("Stage 4/4: Gaussian Splatting Training...", 75)
            cmd = [sys.executable, "train_gs.py", "--colmap_path", "output/colmap/sparse/0", "--iterations", str(self.iter_var.get())]
            self.run_command(cmd, "Training GS", 75, 100)

            self.project_data["completed_stages"].append("training")
            self.save_project()
            self.queue.put(("done", None))
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            self.set_status(f"Error: {str(e)}", 0)
            self.queue.put(("error", str(e)))

if __name__ == "__main__":
    root = tk.Tk()
    app = SplatEditor(root)
    root.mainloop()

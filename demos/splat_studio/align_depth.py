import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" # Suppress NNPACK warnings
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import struct
import warnings
import taichi as ti
import sys

warnings.filterwarnings("ignore")

# Initialize Taichi for Vulkan acceleration
try:
    ti.init(arch=ti.vulkan)
except:
    ti.init(arch=ti.cpu)

# GPU Kernel for Reprojection (VRAM-Safe Streaming)
@ti.kernel
def k_reproject(
    depth: ti.types.ndarray(),
    rgb: ti.types.ndarray(),
    R: ti.types.ndarray(),
    T: ti.types.ndarray(),
    points_out: ti.types.ndarray(),
    colors_out: ti.types.ndarray(),
    W: int, H: int, focal: float, step: int
):
    for i, j in ti.ndrange(H // step, W // step):
        y = i * step
        x = j * step
        z = depth[y, x] * 0.1 # AI depth scaling
        
        if z > 1e-4:
            # Camera Space
            p_cam_x = (x - W / 2.0) * z / focal
            p_cam_y = (y - H / 2.0) * z / focal
            p_cam_z = z
            
            # Camera to World: P_world = R^T * (P_cam - T)
            # R is 3x3, T is 3
            px = p_cam_x - T[0]
            py = p_cam_y - T[1]
            pz = p_cam_z - T[2]
            
            pw_x = R[0, 0] * px + R[1, 0] * py + R[2, 0] * pz
            pw_y = R[0, 1] * px + R[1, 1] * py + R[2, 1] * pz
            pw_z = R[0, 2] * px + R[1, 2] * py + R[2, 2] * pz
            
            idx = i * (W // step) + j
            points_out[idx, 0] = pw_x
            points_out[idx, 1] = pw_y
            points_out[idx, 2] = pw_z
            
            # Color (UChar to Float 0-1 range for internal calcs, or keep as is)
            colors_out[idx, 0] = rgb[y, x, 0]
            colors_out[idx, 1] = rgb[y, x, 1]
            colors_out[idx, 2] = rgb[y, x, 2]

def read_colmap_images(path):
    # Minimal binary reader for COLMAP images.bin
    if not os.path.exists(path):
        return {}
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = f.read(1).decode("utf-8")
                if char == "\0": break
                name += char
            
            # Quat to Rot Matrix
            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
            ])
            images[img_id] = {"name": name, "R": R, "T": np.array([tx, ty, tz])}
            num_points = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points * 24) # Skip points
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/colmap/sparse/0/ai_points.ply")
    parser.add_argument("--max_res", type=int, default=512, help="Max image resolution for GPU processing")
    args = parser.parse_args()

    print(f"Loading AI Model on CPU (VRAM-Safe Mode)...")
    device = "cpu"
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    images = read_colmap_images(os.path.join(args.colmap_path, "images.bin"))
    if not images:
        print("Error: No images found in COLMAP path.")
        return

    all_points = []
    all_colors = []
    
    print(f"Starting Vulkan-Accelerated Reprojection with VRAM Streaming...")
    for i, (img_id, data) in enumerate(images.items()):
        if i % 10 != 0: continue
            
        img_file = os.path.join(args.img_path, data["name"])
        if not os.path.exists(img_file): continue
        
        image = Image.open(img_file).convert("RGB")
        w, h = image.size
        scale = args.max_res / max(w, h)
        if scale < 1.0:
            proc_img = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        else:
            proc_img = image
        
        # 1. AI Inference (CPU to save VRAM for kernels)
        inputs = processor(images=proc_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
        
        # 2. Vulkan Streaming Reprojection
        # We upload only THIS frame's data to VRAM
        cur_h, cur_w = depth_map.shape
        img_np = np.array(proc_img)
        
        step = 4 # Process every 4th pixel for speed
        out_h, out_w = cur_h // step, cur_w // step
        num_pts = out_h * out_w
        
        # Pre-allocate output buffers in RAM (Taichi will stream to these)
        pts_gpu = ti.ndarray(dtype=ti.f32, shape=(num_pts, 3))
        cols_gpu = ti.ndarray(dtype=ti.f32, shape=(num_pts, 3))
        
        # Run GPU Kernel
        # Map depth and image to GPU, reproject, and stream back
        k_reproject(
            depth_map, img_np, data["R"], data["T"],
            pts_gpu, cols_gpu,
            cur_w, cur_h, 1000.0, step
        )
        
        # Download result from VRAM to RAM
        frame_pts = pts_gpu.to_numpy()
        frame_cols = cols_gpu.to_numpy()
        
        # Filter zero points (where depth was invalid)
        mask = np.linalg.norm(frame_pts, axis=1) > 0
        all_points.append(frame_pts[mask])
        all_colors.append(frame_cols[mask])
        
        print(f"Image {i}/{len(images)}: GPU Reprojection Complete ({len(frame_pts[mask])} points)")

    # Consolidate
    final_pts = np.concatenate(all_points, axis=0)
    final_cols = np.concatenate(all_colors, axis=0)

    print(f"Total AI points: {len(final_pts)}. Exporting PLY...")
    with open(args.output, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(final_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(final_pts, final_cols):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            
    print(f"Success! AI Point Cloud ready at {args.output}")

if __name__ == "__main__":
    main()

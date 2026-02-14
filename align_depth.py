import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" # Suppress NNPACK warnings
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import struct
import warnings
warnings.filterwarnings("ignore")

def read_colmap_images(path):
    # Minimal reader for R, T and names
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
            
            # Simple Quat to Rot Matrix
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
    parser.add_argument("--output", type=str, default="output/colmap/sparse/0/merged_points.bin")
    args = parser.parse_args()

    print("Loading AI Model (Depth-Anything-V2-Small)...")
    device = "cpu" # Force CPU to save VRAM
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    images = read_colmap_images(os.path.join(args.colmap_path, "images.bin"))
    
    dense_points = []
    
    print(f"Processing every 10th image ({len(images)//10} total) for AI Depth...")
    for i, (img_id, data) in enumerate(images.items()):
        # Optimization 2: Skip redundant temporal frames
        if i % 10 != 0:
            continue
            
        img_file = os.path.join(args.img_path, data["name"])
        image = Image.open(img_file).convert("RGB")
        
        # Optimization 1: Downscale for faster CPU inference
        max_dim = 512
        w, h = image.size
        scale = max_dim / max(w, h)
        if scale < 1.0:
            low_res_img = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        else:
            low_res_img = image
        
        # AI Inference
        inputs = processor(images=low_res_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth
        
        # Resize depth to match original image size for reprojection
        depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
        depth = depth.squeeze().cpu().numpy()
        
        # Simple reprojection (simplified focal, assuming cam_id=1)
        # In a real product, we'd read cameras.bin for focal length
        W, H = image.size
        focal = 1000.0 # Default fallback
        
        img_np = np.array(image)
        
        # Downsample for speed (e.g., every 8th pixel)
        step = 16
        for y in range(0, H, step):
            for x in range(0, W, step):
                z = depth[y, x] * 0.1 # AI depth is relative, needs scaling usually
                if z <= 0: continue
                
                # Project to camera space
                p_cam = np.array([(x - W/2) * z / focal, (y - H/2) * z / focal, z])
                
                # Camera to World
                # P_world = R_inv * (P_cam - T)
                # But COLMAP T is in world coords? No, P_cam = R*P_world + T
                # So P_world = R^T * (P_cam - T)
                p_world = data["R"].T @ (p_cam - data["T"])
                
                rgb = img_np[y, x]
                dense_points.append((p_world, rgb))
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(images)} images processed")

    print(f"Generated {len(dense_points)} AI points. Merging...")
    # NOTE: In a full implementation, we would write a binary file 
    # compatible with COLMAP points3D.bin or just export a PLY 
    # and have the trainer load it. 
    # For now, let's just save as PLY for the trainer to pick up.
    
    output_ply = "output/colmap/sparse/0/ai_points.ply"
    with open(output_ply, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(dense_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, rgb in dense_points:
            f.write(f"{p[0]} {p[1]} {p[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
            
    print(f"AI Points saved to {output_ply}")

if __name__ == "__main__":
    main()

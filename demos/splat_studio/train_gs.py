import taichi as ti
import numpy as np
import argparse
import os
import struct
from PIL import Image
import random

# Initialize Taichi with Vulkan backend
ti.init(arch=ti.vulkan, device_memory_GB=1.5)

import vulkan_nn_lib.torch_shim as torch
import vulkan_nn_lib.core as vnn
from vulkan_nn_lib.optimizers import AutoAdam

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_colmap_points3D(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            id = read_next_bytes(fid, 8, "Q")[0]
            xyz = np.array(read_next_bytes(fid, 24, "ddd"))
            rgb = np.array(read_next_bytes(fid, 3, "BBB"))
            read_next_bytes(fid, 8, "d") # error
            track_len = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, 8 * track_len, "ii" * track_len)
            points3D[id] = {"xyz": xyz, "rgb": rgb}
    return points3D

def read_colmap_cameras(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            width, height = camera_properties[2], camera_properties[3]
            params = read_next_bytes(fid, 32, "dddd") # PINHOLE
            cameras[camera_id] = {"w": width, "h": height, "f": params[0]}
    return cameras

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_colmap_images(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id, qvec, tvec, camera_id = image_properties[0], np.array(image_properties[1:5]), np.array(image_properties[5:8]), image_properties[8]
            image_name = ""
            while True:
                char = fid.read(1)
                if char == b"\x00": break
                image_name += char.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            R = qvec2rotmat(qvec)
            images[image_id] = {"R": R, "T": tvec, "name": image_name, "cam_id": camera_id}
    return images

@ti.data_oriented
class GaussianModel:
    def __init__(self, num_points):
        self.num_points = num_points
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=num_points, needs_grad=True)
        self.opacity = ti.field(dtype=ti.f32, shape=num_points, needs_grad=True)
        self.sh = ti.field(dtype=ti.f32, shape=(num_points, 1, 3), needs_grad=True)
        self.display_color = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        
        # VNN Bridge: Parameters managed by the core engine for OOM-safety
        self.pos_vnn = vnn.Tensor(None, shape=(num_points, 3), requires_grad=True, device='auto')
        self.sh_vnn = vnn.Tensor(None, shape=(num_points, 1, 3), requires_grad=True, device='auto')
        self.opacity_vnn = vnn.Tensor(None, shape=(num_points,), requires_grad=True, device='auto')

    @ti.kernel
    def initialize(self, xyz: ti.types.ndarray(), rgb: ti.types.ndarray()):
        for i in range(self.num_points):
            self.pos[i] = ti.Vector([xyz[i, 0], -xyz[i, 1], -xyz[i, 2]])
            self.opacity[i] = 0.5
            self.sh[i, 0, 0] = (rgb[i, 0] / 255.0 - 0.5) / 0.28209
            self.sh[i, 0, 1] = (rgb[i, 1] / 255.0 - 0.5) / 0.28209
            self.sh[i, 0, 2] = (rgb[i, 2] / 255.0 - 0.5) / 0.28209
            self.display_color[i] = ti.Vector([rgb[i, 0]/255.0, rgb[i, 1]/255.0, rgb[i, 2]/255.0])

    @ti.kernel
    def reset_loss(self):
        self.loss[None] = 0.0

    @ti.kernel
    def train_step(self, R: ti.types.ndarray(), T: ti.types.ndarray(), focal: ti.f32, width: ti.i32, height: ti.i32, target_img: ti.types.ndarray()):
        # Simplified point-based supervision
        # Projects 3D points to 2D and compares color at that pixel
        for i in range(self.num_points):
            # Transform to camera space
            p_cam = ti.Vector([
                R[0,0]*self.pos[i].x + R[0,1]*self.pos[i].y + R[0,2]*self.pos[i].z + T[0],
                R[1,0]*self.pos[i].x + R[1,1]*self.pos[i].y + R[1,2]*self.pos[i].z + T[1],
                R[2,0]*self.pos[i].x + R[2,1]*self.pos[i].y + R[2,2]*self.pos[i].z + T[2]
            ])
            
            if p_cam.z > 0.1:
                # Project to screen
                u = int(focal * p_cam.x / p_cam.z + width / 2)
                v = int(focal * p_cam.y / p_cam.z + height / 2)
                
                if u >= 0 and u < width and v >= 0 and v < height:
                    # Point color from SH
                    p_rgb = ti.Vector([
                        ti.max(0, ti.min(1, self.sh[i, 0, 0] * 0.28209 + 0.5)),
                        ti.max(0, ti.min(1, self.sh[i, 0, 1] * 0.28209 + 0.5)),
                        ti.max(0, ti.min(1, self.sh[i, 0, 2] * 0.28209 + 0.5))
                    ])
                    # Target color from image
                    t_rgb = ti.Vector([target_img[v, u, 0]/255.0, target_img[v, u, 1]/255.0, target_img[v, u, 2]/255.0])
                    
                    # Squared error
                    self.loss[None] += ((p_rgb - t_rgb)**2).sum() / self.num_points

    @ti.kernel
    def rasterize(self, R: ti.types.ndarray(), T: ti.types.ndarray(), focal: ti.f32, width: ti.i32, height: ti.i32, canvas: ti.template(), radius_limit: ti.f32, jitter_amp: ti.f32, time: ti.f32, pulse_amp: ti.f32, density_mult: ti.i32):
        for i, j in ti.ndrange(width, height):
            canvas[i, j] = ti.Vector([0.05, 0.05, 0.05])

        for i in range(self.num_points):
            for d in range(density_mult):
                jitter = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]) * jitter_amp
                p_world = self.pos[i] + jitter
                
                p_cam = ti.Vector([
                    R[0,0]*p_world.x + R[0,1]*p_world.y + R[0,2]*p_world.z + T[0],
                    R[1,0]*p_world.x + R[1,1]*p_world.y + R[1,2]*p_world.z + T[1],
                    R[2,0]*p_world.x + R[2,1]*p_world.y + R[2,2]*p_world.z + T[2]
                ])
                
                if p_cam.z > 0.1:
                    u_c = focal * p_cam.x / p_cam.z + width / 2
                    v_c = focal * p_cam.y / p_cam.z + height / 2
                    
                    p_rgb = ti.Vector([
                        ti.max(0, ti.min(1, self.sh[i, 0, 0] * 0.28209 + 0.5)),
                        ti.max(0, ti.min(1, self.sh[i, 0, 1] * 0.28209 + 0.5)),
                        ti.max(0, ti.min(1, self.sh[i, 0, 2] * 0.28209 + 0.5))
                    ])

                    pulse = 1.0 + ti.sin(time * 5.0 + i + d) * pulse_amp
                    
                    # Distance-aware radius with clamping
                    # rad = size_in_world * focal / distance
                    rad_f = radius_limit * width / p_cam.z * pulse
                    rad = int(ti.max(1, ti.min(15, rad_f))) 
                    
                    # Brightness compensation
                    # When far away (high Z), many splats overlap, so we reduce alpha.
                    # When close (low Z), we need more alpha to maintain brightness.
                    alpha_base = 0.4 / density_mult
                    # Simple heuristic: brightness scales with distance to counteract additive overlap
                    brightness_comp = ti.min(2.0, p_cam.z / 15.0) 
                    if p_cam.z < 10.0: brightness_comp = 1.0 + (10.0 - p_cam.z) * 0.1
                    
                    for du, dv in ti.ndrange((-rad, rad), (-rad, rad)):
                        u, v = int(u_c + du), int(v_c + dv)
                        if u >= 0 and u < width and v >= 0 and v < height:
                            dist_sq = (du*du + dv*dv) / (rad*rad)
                            if dist_sq < 1.0:
                                alpha = ti.exp(-2.0 * dist_sq) * alpha_base * brightness_comp
                                canvas[u, v] += p_rgb * alpha

    @ti.kernel
    def box_blur(self, width: ti.i32, height: ti.i32, canvas: ti.template(), temp: ti.template(), blur_rad: ti.i32):
        # Optimized separable box blur (simulated gaussian)
        for i, j in ti.ndrange(width, height):
            acc = ti.Vector([0.0, 0.0, 0.0])
            count = 0
            for di in range(-blur_rad, blur_rad + 1):
                u = i + di
                if u >= 0 and u < width:
                    acc += canvas[u, j]
                    count += 1
            temp[i, j] = acc / count

        for i, j in ti.ndrange(width, height):
            acc = ti.Vector([0.0, 0.0, 0.0])
            count = 0
            for dj in range(-blur_rad, blur_rad + 1):
                v = j + dj
                if v >= 0 and v < height:
                    acc += temp[i, v]
                    count += 1
            canvas[i, j] = acc / count

    @ti.kernel
    def update_params(self, lr: ti.f32):
        # Legacy manual update (kept for reference, but we use AutoAdam)
        for i in range(self.num_points):
            self.pos[i] -= lr * self.pos.grad[i]
            for j, k in ti.static(ti.ndrange(1, 3)):
                self.sh[i, j, k] -= lr * self.sh.grad[i, j, k]

    def params(self):
        """Returns the VNN managed parameters."""
        return [self.pos_vnn, self.sh_vnn, self.opacity_vnn]

    def sync_to_vnn(self):
        """Syncs gradients from Taichi fields to VNN tensors."""
        self.pos_vnn.grad.load_from_numpy(self.pos.grad.to_numpy().reshape(self.num_points, 3))
        self.sh_vnn.grad.load_from_numpy(self.sh.grad.to_numpy().reshape(self.num_points, 1, 3))
        self.opacity_vnn.grad.load_from_numpy(self.opacity.grad.to_numpy())

    def sync_from_vnn(self):
        """Syncs updated parameters from VNN tensors back to Taichi fields."""
        self.pos.from_numpy(self.pos_vnn.to_numpy().reshape(self.num_points, 3))
        self.sh.from_numpy(self.sh_vnn.to_numpy().reshape(self.num_points, 1, 3))
        self.opacity.from_numpy(self.opacity_vnn.to_numpy())

    def export_ply(self, path):
        pos = self.pos.to_numpy()
        color = self.display_color.to_numpy()
        with open(path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {self.num_points}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(b"end_header\n")
            for i in range(self.num_points):
                f.write(struct.pack("<fffBBB", pos[i,0], pos[i,1], pos[i,2], int(color[i,0]*255), int(color[i,1]*255), int(color[i,2]*255)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", type=str, default="output/colmap/sparse/0")
    parser.add_argument("--img_path", type=str, default="output/frames")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    print("Loading COLMAP data...")
    cameras = read_colmap_cameras(os.path.join(args.colmap_path, "cameras.bin"))
    images = read_colmap_images(os.path.join(args.colmap_path, "images.bin"))
    
    # Try to load AI-generated points first
    ai_ply = os.path.join(args.colmap_path, "ai_points.ply")
    if os.path.exists(ai_ply):
        print(f"Loading AI-enhanced dense points from {ai_ply}...")
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ai_ply)
        xyz = np.asarray(pcd.points).astype(np.float32)
        rgb = (np.asarray(pcd.colors) * 255).astype(np.float32)
    else:
        points3D = read_colmap_points3D(os.path.join(args.colmap_path, "points3D.bin"))
        xyz = np.array([p["xyz"] for p in points3D.values()]).astype(np.float32)
        rgb = np.array([p["rgb"] for p in points3D.values()]).astype(np.float32)
    
    model = GaussianModel(xyz.shape[0])
    model.initialize(xyz, rgb)
    
    # Sync initial state to VNN tensors for optimization
    model.pos_vnn.load_from_numpy(model.pos.to_numpy().reshape(model.num_points, 3))
    model.sh_vnn.load_from_numpy(model.sh.to_numpy().reshape(model.num_points, 1, 3))
    model.opacity_vnn.load_from_numpy(model.opacity.to_numpy())
    
    if args.view:
        W, H = 1000, 700
        window = ti.ui.Window("3DGS Vulkan - Splat Renderer", (W, H))
        canvas_gui = window.get_canvas()
        gui, camera = window.get_gui(), ti.ui.Camera()
        
        # Correct shape for set_image (W, H)
        render_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
        
        center = np.mean(xyz, axis=0)
        view_center = np.array([center[0], -center[1], -center[2]])
        extent = np.max(np.linalg.norm(xyz - center, axis=1))
        
        camera.position(view_center[0], view_center[1], view_center[2] + extent * 1.5)
        camera.lookat(view_center[0], view_center[1], view_center[2])
        
        radius_limit, move_speed = 0.005, 0.02
        jitter_amp = 0.0
        pulse_amp = 0.0
        density_mult = 1
        blur_rad = 0
        show_splats = True
        t = 0.0
        
        # Temp buffer for blurring
        blur_temp = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
        
        while window.running:
            camera.track_user_inputs(window, movement_speed=move_speed, hold_key=ti.ui.RMB)
            t += 0.016 # Approx 60fps
            
            if show_splats:
                # View Matrix logic
                eye = np.array(camera.curr_position)
                look = np.array(camera.curr_lookat)
                up = np.array([0, 1, 0])
                
                # Z points to screen
                z_axis = look - eye
                z_axis /= np.linalg.norm(z_axis)
                x_axis = np.cross(z_axis, up)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(x_axis, z_axis)
                
                R = np.vstack([x_axis, y_axis, z_axis])
                T = -R @ eye
                
                model.rasterize(R.astype(np.float32), T.astype(np.float32), 1000.0, W, H, render_buffer, radius_limit, jitter_amp, t, pulse_amp, density_mult)
                if blur_rad > 0:
                    model.box_blur(W, H, render_buffer, blur_temp, blur_rad)
                canvas_gui.set_image(render_buffer)
            else:
                scene = window.get_scene()
                scene.set_camera(camera)
                scene.ambient_light((0.8, 0.8, 0.8))
                scene.particles(model.pos, radius=0.003, per_vertex_color=model.display_color)
                canvas_gui.scene(scene)

            with gui.sub_window("Controls", 0.02, 0.02, 0.3, 0.6):
                show_splats = gui.checkbox("Splat Mode (Real Rasterizer)", show_splats)
                radius_limit = gui.slider_float("Splat Size", radius_limit, 0.001, 0.01)
                jitter_amp = gui.slider_float("Brownian Motion", jitter_amp, 0.0, 0.05)
                density_mult = gui.slider_int("Density Mult (Fake Points)", density_mult, 1, 4)
                pulse_amp = gui.slider_float("Splat Pulse", pulse_amp, 0.0, 0.5)
                blur_rad = gui.slider_int("Blur Intensity", blur_rad, 0, 5)
                move_speed = gui.slider_float("Move Speed", move_speed, 0.001, 0.1)
                if gui.button("Reset View"): 
                    camera.position(view_center[0], view_center[1], view_center[2] + extent * 1.5)
                    camera.lookat(view_center[0], view_center[1], view_center[2])
            
            window.show()

    else:
        print(f"Starting Training on {len(images)} images...", flush=True)
        print("Note: The first few iterations may take time to start due to JIT compilation (Vulkan).", flush=True)
        img_list = list(images.values())
        optimizer = AutoAdam(model.params(), lr=1e-3) # Higher-order optimizer from VNN
        
        for i in range(args.iterations):
            img_data = random.choice(img_list)
            cam = cameras[img_data["cam_id"]]
            img_path = os.path.join(args.img_path, img_data["name"])
            
            # Load and resize image to match camera params (simplified)
            target = np.array(Image.open(img_path).resize((cam["w"], cam["h"]))).astype(np.uint8)
            
            model.reset_loss()
            with ti.ad.Tape(model.loss):
                model.train_step(img_data["R"], img_data["T"], cam["f"], cam["w"], cam["h"], target)
            
            # Use VNN's AutoAdam for memory-safe updates
            model.sync_to_vnn()
            optimizer.step()
            model.sync_from_vnn()
            
            if i % 10 == 0:
                print(f"Iteration {i}/{args.iterations} - Loss: {model.loss[None]:.10f}", flush=True)

    model.export_ply("output/trained_splats.ply")
    print("Optimization finished. Final model saved to output/trained_splats.ply")

if __name__ == "__main__":
    main()

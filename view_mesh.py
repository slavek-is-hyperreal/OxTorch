import taichi as ti
import numpy as np
import argparse
import os

# Taichi with Vulkan
ti.init(arch=ti.vulkan)

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Viewer (Vulkan)")
    parser.add_argument("--input", type=str, default="output/room_mesh.obj", help="Path to mesh (.obj or .ply)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    # Load mesh vertices and faces for visualization
    # We use Taichi's built-in OBJ loader if possible, or manual load
    # For simplicity in this viewer, we'll try to load via Taichi or basic numpy
    
    window = ti.ui.Window("3D Mesh Viewer", (1280, 720))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # Simple camera setup
    camera.position(0, 0, 5)
    camera.lookat(0, 0, 0)
    
    # Load mesh
    try:
        mesh = ti.ui.Mesh()
        # Note: ti.ui.Mesh is usually for procedural meshes or loaded ones
        # We can use ti.static_mesh if Taichi supports it directly or just scene.mesh
        pass
    except:
        print("Falling back to manual mesh rendering...")

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0, 5, 0), color=(1, 1, 1))

        # In Taichi GGUI, we usually render meshes via scene.mesh
        # but for simplicity since we have point cloud logic, 
        # let's just use open3d's own viewer if available, 
        # or implement a fast vertex-based viewer here.
        
        # For now, let's provide a script that uses Open3D for visualization too 
        # as it's more robust for meshes.
        
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    import open3d as o3d
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output/reconstruction.ply")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        print(f"Opening {args.input} in Open3D Viewer...")
        
        # Determine geometry type
        # Open3D's draw_geometries can take a list of things
        geometries = []
        
        # Try loading as mesh
        mesh = o3d.io.read_triangle_mesh(args.input)
        if mesh.has_triangles():
            mesh.compute_vertex_normals()
            geometries.append(mesh)
        else:
            # Try loading as LineSet (Graph)
            ls = o3d.io.read_line_set(args.input)
            if ls.has_lines():
                geometries.append(ls)
            else:
                # Fallback to PointCloud
                pcd = o3d.io.read_point_cloud(args.input)
                if pcd.has_points():
                    geometries.append(pcd)
        
        if not geometries:
            print("Unsupported or empty geometry file.")
        else:
            o3d.visualization.draw_geometries(geometries, window_name="Splat Research Viewer",
                                              width=1280, height=720,
                                              mesh_show_back_face=True)
    else:
        print(f"File {args.input} not found.")

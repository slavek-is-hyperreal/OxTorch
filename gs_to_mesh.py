import open3d as o3d
import numpy as np
import os
import argparse
import sys
import gc
import tempfile
import taichi as ti
from scipy.spatial import cKDTree
from skimage import measure

# Initialize Taichi
ti.init(arch=ti.vulkan if sys.platform != "darwin" else ti.cpu)

@ti.kernel
def check_edge_stability_gpu(
    points: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    edges: ti.types.ndarray(),
    counts: ti.types.ndarray(),
    iterations: int,
    radius_sq: float,
    jitter_std: float,
    planar_penalty: float
):
    for i, j in ti.ndrange(iterations, edges.shape[0]):
        p1_idx = edges[j, 0]
        p2_idx = edges[j, 1]
        
        p1 = ti.Vector([points[p1_idx, 0], points[p1_idx, 1], points[p1_idx, 2]])
        p2 = ti.Vector([points[p2_idx, 0], points[p2_idx, 1], points[p2_idx, 2]])
        n1 = ti.Vector([normals[p1_idx, 0], normals[p1_idx, 1], normals[p1_idx, 2]])
        n2 = ti.Vector([normals[p2_idx, 0], normals[p2_idx, 1], normals[p2_idx, 2]])
        
        # Jitter both points
        j1 = ti.Vector([ti.random(), ti.random(), ti.random()]) * jitter_std
        j2 = ti.Vector([ti.random(), ti.random(), ti.random()]) * jitter_std
        
        diff = (p1 + j1) - (p2 + j2)
        d_sq = diff.norm_sqr()
        
        if d_sq < radius_sq:
            # Planarity Check: Edge vector should be perpendicular to surface normals
            edge_dir = diff.normalized()
            # If edge 'perforates' the surface depth, penalty applies
            dot1 = ti.abs(edge_dir.dot(n1))
            dot2 = ti.abs(edge_dir.dot(n2))
            
            # If dots are small (near 0), edge is in tangent plane
            if dot1 < planar_penalty and dot2 < planar_penalty:
                ti.atomic_add(counts[j], 1)

def estimate_normals(points, k=15):
    """Local normal estimation via PCA on k-nearest neighbors."""
    if len(points) < 5:
        return np.zeros_like(points)
    
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=min(k, len(points)))
    
    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[idxs[i]]
        # PCA
        cov = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is the eigenvector corresponding to the smallest eigenvalue
        normals[i] = eigenvectors[:, 0]
    return normals

def orbital_field_reconstruction(points, colors, iterations=10, grid_size=32, jitter=0.01):
    """Method A: Extracts a smooth isosurface using statistical jittering."""
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    margin = (max_bound - min_bound) * 0.1
    min_bound -= margin
    max_bound += margin
    
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    color_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    
    for _ in range(iterations):
        jittered = points + np.random.normal(0, jitter, points.shape)
        indices = ((jittered - min_bound) / (max_bound - min_bound + 1e-6) * (grid_size - 1)).astype(int)
        valid = np.all((indices >= 0) & (indices < grid_size), axis=1)
        indices = indices[valid]
        c_vals = colors[valid]
        for idx, col in zip(indices, c_vals):
            grid[idx[0], idx[1], idx[2]] += 1.0
            color_grid[idx[0], idx[1], idx[2]] += col

    threshold = np.mean(grid) + np.std(grid)
    try:
        verts, faces, _, _ = measure.marching_cubes(grid, level=threshold)
        verts = verts / (grid_size - 1) * (max_bound - min_bound) + min_bound
        v_idx = np.clip(((verts - min_bound) / (max_bound - min_bound + 1e-6) * (grid_size - 1)).astype(int), 0, grid_size-1)
        v_cols = color_grid[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]] / iterations
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(v_cols, 0, 1))
        return mesh
    except: return None

def graph_statistical_connectivity_gpu(points, colors, iterations=30, radius=0.05, threshold_ratio=0.9, planar_penalty=0.2):
    """Method B: GPU-accelerated graph connectivity analysis with planar constraints."""
    if len(points) < 5: return None
    
    # 1. Estimate Surface Normals using PCA
    normals = estimate_normals(points, k=15)
    
    # 2. Find candidate edges via KDTree
    tree = cKDTree(points)
    pairs = list(tree.query_pairs(radius))
    if not pairs: return None
    
    edge_array = np.array(pairs, dtype=np.int32)
    counts = np.zeros(len(edge_array), dtype=np.int32)
    
    # 3. Run Taichi Kernel with Planar Constraints
    check_edge_stability_gpu(
        points.astype(np.float32),
        normals.astype(np.float32),
        edge_array,
        counts,
        iterations,
        radius**2,
        radius * 0.15,
        planar_penalty
    )
    
    mask = counts >= int(iterations * threshold_ratio)
    stable_edges = edge_array[mask]
    
    if len(stable_edges) == 0: return None
    
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(stable_edges)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def adaptive_octree_march(points, colors, mode, bounds, depth=0, max_pts=20000, radius=0.02, planar_penalty=0.2, temp_file=None, vis=None, live=False):
    """Level 4 Adaptive March: Recursive Octree with Planar Constraints."""
    p_min, p_max = bounds
    mask = np.all((points >= p_min) & (points <= p_max), axis=1)
    tile_pts = points[mask]
    tile_cols = colors[mask]
    
    if len(tile_pts) == 0: return []
    
    # If too many points, split into 8
    if len(tile_pts) > max_pts and depth < 6:
        center = (p_min + p_max) / 2.0
        results = []
        for i in range(8):
            oct_min = np.copy(p_min); oct_max = np.copy(p_max)
            for j in range(3):
                if (i >> j) & 1: oct_min[j] = center[j]
                else: oct_max[j] = center[j]
            results.extend(adaptive_octree_march(points, colors, mode, (oct_min, oct_max), depth+1, max_pts, radius, planar_penalty, temp_file, vis, live))
        return results
    
    # Base case: process RAM-safe leaf
    print(f"  Processing Leaf: depth={depth}, pts={len(tile_pts)}", flush=True)
    mesh_res = None
    if mode == "orbital":
        mesh_res = orbital_field_reconstruction(tile_pts, tile_cols, jitter=radius*0.5)
        if mesh_res and live: vis.add_geometry(mesh_res)
    elif mode == "graph":
        ls_res = graph_statistical_connectivity_gpu(tile_pts, tile_cols, radius=radius, threshold_ratio=0.9, planar_penalty=planar_penalty)
        if ls_res:
            # Brighten colors slightly for visibility against background
            ls_res.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(ls_res.colors) * 1.5, 0, 1))
            
            global_indices = np.where(mask)[0]
            local_edges = np.asarray(ls_res.lines)
            if len(local_edges) > 0:
                global_edges = global_indices[local_edges]
                with open(temp_file, "ab") as f:
                    np.save(f, global_edges.astype(np.int32))
                if live: vis.add_geometry(ls_res)
    
    if live:
        vis.poll_events()
        vis.update_renderer()
        
    gc.collect()
    return [mesh_res] if mesh_res else []

def main():
    parser = argparse.ArgumentParser(description="Impressionistic Reconstruction Level 3 (Clean Mode)")
    parser.add_argument("--input", type=str, default="output/trained_splats.ply")
    parser.add_argument("--mode", type=str, choices=["orbital", "graph", "impressionist"], default="orbital")
    parser.add_argument("--output", type=str, default="output/reconstruction.ply")
    parser.add_argument("--radius", type=float, default=0.04, help="Small radius for connectivity (0.01-0.1)")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input): return
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Initialize Clean Live View with High Contrast
    vis = None
    if args.live:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Live Research View (Contrast Boost)", width=1280, height=720)
        
        render_opt = vis.get_render_option()
        # Light grey background to see dark wood/furniture
        render_opt.background_color = np.asarray([0.8, 0.8, 0.8])
        render_opt.line_width = 2.0
        render_opt.point_size = 3.0

    temp_edge_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin").name
    scene_bounds = (np.min(points, axis=0), np.max(points, axis=0))
    
    print(f"Starting Level 4 Clean March (Radius={args.radius}, Planar={args.planar})...", flush=True)
    adaptive_octree_march(points, colors, args.mode, scene_bounds, radius=args.radius, planar_penalty=args.planar, temp_file=temp_edge_file, vis=vis, live=args.live)

    if args.live:
        print("Live Reconstruction Finished. Close window to save.", flush=True)
        vis.run()
        vis.destroy_window()

    if args.mode == "orbital":
        if meshes:
            combined = meshes[0]
            for m in meshes[1:]: combined += m
            o3d.io.write_triangle_mesh(args.output, combined)
            print(f"Success! Model saved to {args.output}")
            
    elif args.mode == "graph":
        final_edges = []
        if os.path.exists(temp_edge_file):
            with open(temp_edge_file, "rb") as f:
                while True:
                    try: final_edges.append(np.load(f))
                    except: break
            os.remove(temp_edge_file)
            
        if final_edges:
            merged = np.unique(np.sort(np.concatenate(final_edges), axis=1), axis=0)
            
            # --- OUTLIER REMOVAL: Keep only points that are part of an edge ---
            active_indices = np.unique(merged)
            print(f"Cleaning model: Keeping {len(active_indices)} connected points out of {len(points)}.", flush=True)
            
            clean_points = points[active_indices]
            clean_colors = colors[active_indices]
            
            # Fast index remapping
            full_map = np.full(len(points), -1, dtype=np.int32)
            full_map[active_indices] = np.arange(len(active_indices))
            merged_remapped = full_map[merged]
            
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(clean_points)
            ls.lines = o3d.utility.Vector2iVector(merged_remapped)
            ls.colors = o3d.utility.Vector3dVector(clean_colors)
            o3d.io.write_line_set(args.output, ls)
            print(f"Success! Stable graph saved to {args.output}")

if __name__ == "__main__":
    main()

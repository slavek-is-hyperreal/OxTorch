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

def estimate_normals(points, k=30):
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

def orbital_field_reconstruction(points, colors, iterations=15, grid_size=48, jitter=0.01):
    """Method A: Improved Voxel Field with robust thresholding."""
    if len(points) < 10: return None
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    margin = (max_bound - min_bound) * 0.05
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

    # Robust Threshold: 50% of peak density, but at least 2.0
    max_density = np.max(grid)
    if max_density < 2.0: return None
    
    threshold = max(2.0, max_density * 0.5)
    
    try:
        verts, faces, _, _ = measure.marching_cubes(grid, level=threshold)
        verts = verts / (grid_size - 1) * (max_bound - min_bound) + min_bound
        v_idx = np.clip(((verts - min_bound) / (max_bound - min_bound + 1e-6) * (grid_size - 1)).astype(int), 0, grid_size-1)
        v_cols = color_grid[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]] / (grid[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]] + 1e-6)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(v_cols, 0, 1))
        
        # Clean up result immediately
        mesh = filter_islands(mesh)
        return mesh
    except Exception as e:
        return None

def filter_islands(mesh):
    """Keeps only the largest connected component of the mesh."""
    if not mesh or mesh.is_empty(): return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) <= 1: return mesh
    
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh

def bpa_skinning(points, colors, radius=0.03):
    """Wraps a mesh using Ball Pivoting Algorithm (BPA)."""
    if len(points) < 10: return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    
    # BPA radii: suggest a sequence of growing radii
    radii = [radius, radius * 2]
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        mesh = filter_islands(mesh)
        return mesh
    except:
        return None

def alpha_skinning(points, colors, alpha=0.05):
    """Wraps a mesh around points using Alpha Shapes."""
    if len(points) < 4: return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals for lighting even if we use alpha shapes
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        return mesh
    except:
        return None

def gaussian_cloud_reconstruction(points, colors, radius=0.04, extinction_threshold=0.08):
    """Method E: Crystalline Nebula. Expansion logic + Optimized volumetric gradient."""
    if len(points) < 5: return None
    
    # 1. Global Point Scrub: Remove 'rays' and isolated dust before meshing
    # We use DBSCAN to find the main structure
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # eps = search distance, min_points = minimum size of a valid structure
    labels = np.array(pcd.cluster_dbscan(eps=radius*2.5, min_points=15))
    
    if len(labels) > 0:
        max_label = labels.max()
        if max_label >= 0:
            # Keep only the largest cluster (the cabinet)
            counts = np.bincount(labels[labels >= 0])
            largest_cluster = np.argmax(counts)
            mask = labels == largest_cluster
            points = points[mask]
            colors = colors[mask]
            
    if len(points) < 5: return None

    # 2. Expansion logic
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    d_nn = dists[:, 1]
    
    all_verts_list = []
    all_faces_list = []
    all_cols_list = []
    v_offset = 0
    
    # Create a template sphere for performance
    # resolution=3 makes it look organic (not a rhombus)
    template_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=3)
    t_verts = np.asarray(template_sphere.vertices)
    t_faces = np.asarray(template_sphere.triangles)
    
    # Template volumetric connectivity (shell to center 0)
    t_vol_faces = []
    for f in t_faces:
        v1, v2, v3 = f + 1
        t_vol_faces.append([0, v1, v2])
        t_vol_faces.append([0, v2, v3])
        t_vol_faces.append([0, v3, v1])
        t_vol_faces.append([v1, v2, v3])
    t_vol_faces = np.array(t_vol_faces)
    
    print(f"  Generating {len(points)} Nebula Blobs...", flush=True)
    for i in range(len(points)):
        p = points[i]
        c = colors[i]
        dist = d_nn[i]
        
        # Extinction threshold check
        if dist > extinction_threshold:
            continue
            
        r = max(dist * 0.55, 0.002) 
        
        # Transform template to world space for this point
        verts = t_verts * r + p
        all_verts = np.vstack([p, verts])
        
        # Gradient colors
        c_dark = np.clip(c * 0.3, 0, 1)
        c_glow = np.clip(c * 1.8, 0, 1)
        v_cols = np.vstack([c_dark, [c_glow] * len(verts)])
        
        # Accumulate
        all_verts_list.append(all_verts)
        all_faces_list.append(t_vol_faces + v_offset)
        all_cols_list.append(v_cols)
        v_offset += len(all_verts)
        
    if not all_verts_list: return None
    
    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(np.concatenate(all_verts_list))
    combined.triangles = o3d.utility.Vector3iVector(np.concatenate(all_faces_list))
    combined.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(all_cols_list))
    
    # FIX: Compute normals for correct lighting (avoids 'black rhombus')
    combined.compute_vertex_normals()
    return combined

def crystal_growth_reconstruction(points, colors, extinction_threshold=0.06):
    """Method D: Points grow into cubes until they hit a neighbor with Extinction logic."""
    if len(points) < 2: return None
    
    # 1. Expansion radii (half distance to nearest neighbor)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    radii = dists[:, 1] * 0.55 # 10% overlap safety
    
    # 2. Generate Boxes with Extinction Mechanic
    combined = o3d.geometry.TriangleMesh()
    for p, c, r in zip(points, colors, radii):
        # EXTINCTION: If a point is too isolated, it 'dies' (is not rendered)
        if r > extinction_threshold:
            continue
            
        # Ensure a minimum size (0.001) to avoid Open3D width <= 0 errors
        r_clamped = max(r, 0.001) 
        box = o3d.geometry.TriangleMesh.create_box(width=r_clamped*2, height=r_clamped*2, depth=r_clamped*2)
        box.translate(p - np.array([r_clamped, r_clamped, r_clamped]))
        box.paint_uniform_color(c)
        combined += box
    return combined
    """Generates oriented colourful discs (surfels) for each point."""
    if len(points) == 0: return None
    
    all_meshes = []
    # To keep it efficient, we only generate surfels for a subset or use a vectorized approach
    # For now, let's create a combined mesh of discs
    for i in range(len(points)):
        p = points[i]
        c = colors[i]
        n = normals[i]
        
        # Create a small disc (cylinder with tiny height)
        # Use a simpler proxy: a small circle-aligned mesh
        disc = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.001)
        disc.paint_uniform_color(c)
        
        # Orient disc to point normal
        # Cylinder default up is [0, 0, 1]
        z_axis = np.array([0, 0, 1])
        if not np.allclose(n, z_axis):
            v = np.cross(z_axis, n)
            s = np.linalg.norm(v)
            if s > 1e-6:
                c_val = np.dot(z_axis, n)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c_val) / (s ** 2))
                disc.rotate(rotation_matrix, center=(0, 0, 0))
        
        disc.translate(p)
        all_meshes.append(disc)
    
    if not all_meshes: return None
    
    # Combine (Warning: This can be slow for many points, usually better to use a custom Shader or PointCloud with normals)
    # For Level 6 Research, we'll combine them into one mesh.
    combined = all_meshes[0]
    for m in all_meshes[1:]:
        combined += m
    return combined

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

def adaptive_octree_march(points, colors, mode, bounds, depth=0, max_pts=20000, radius=0.04, planar_penalty=0.2, temp_file=None, vis=None, live=False):
    """Level 5 Adaptive March: Overlapping Octree for seamless manifold reconstruction."""
    p_min, p_max = bounds
    
    # Overlap logic: Each leaf processes its volume PLUS a margin to find neighbors
    e_min = p_min - radius
    e_max = p_max + radius
    
    # Select points in the extended volume
    mask = np.all((points >= e_min) & (points <= e_max), axis=1)
    tile_pts = points[mask]
    tile_cols = colors[mask]
    
    if len(tile_pts) == 0: return []
    
    # If too many points, split into 8 (only split if points inside the CORE bounds are many)
    core_mask = np.all((points >= p_min) & (points <= p_max), axis=1)
    core_count = np.sum(core_mask)
    
    if core_count > max_pts and depth < 6:
        center = (p_min + p_max) / 2.0
        results = []
        for i in range(8):
            oct_min = np.copy(p_min); oct_max = np.copy(p_max)
            for j in range(3):
                if (i >> j) & 1: oct_min[j] = center[j]
                else: oct_max[j] = center[j]
            results.extend(adaptive_octree_march(points, colors, mode, (oct_min, oct_max), depth+1, max_pts, radius, planar_penalty, temp_file, vis, live))
        return results
    
    # Base case: process leaf with overlap to ensure connectivity
    print(f"  Level 8 Leaf: depth={depth}, pts={len(tile_pts)} (seams protected)", flush=True)
    mesh_res = None
    if mode == "orbital":
        mesh_res = orbital_field_reconstruction(tile_pts, tile_cols, jitter=radius*0.5)
        if mesh_res and live: vis.add_geometry(mesh_res)
    elif mode == "crystal":
        mesh_res = crystal_growth_reconstruction(tile_pts, tile_cols, extinction_threshold=radius*1.5)
        if mesh_res and live: vis.add_geometry(mesh_res)
    elif mode == "nebula":
        mesh_res = gaussian_cloud_reconstruction(tile_pts, tile_cols, radius=radius, extinction_threshold=radius*1.5)
        if mesh_res and live: vis.add_geometry(mesh_res)
    elif mode == "graph":
        # Stability lowered to 80% to allow better surface flow
        ls_res = graph_statistical_connectivity_gpu(tile_pts, tile_cols, radius=radius, threshold_ratio=0.8, planar_penalty=planar_penalty)
        if ls_res:
            ls_res.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(ls_res.colors) * 1.5, 0, 1))
            
            global_indices = np.where(mask)[0]
            local_edges = np.asarray(ls_res.lines)
            
            if len(local_edges) > 0:
                global_edges = global_indices[local_edges]
                
                # IMPORTANT: Only save edges where AT LEAST ONE point is inside the core bounds
                # to prevent duplicate work across tiles while maintaining connectivity
                edge_p1 = points[global_edges[:, 0]]
                in_core = np.any((edge_p1 >= p_min) & (edge_p1 <= p_max), axis=1)
                valid_edges = global_edges[in_core]
                
                if len(valid_edges) > 0:
                    with open(temp_file, "ab") as f:
                        np.save(f, valid_edges.astype(np.int32))
                    if live: vis.add_geometry(ls_res)
    
    if live:
        vis.poll_events()
        vis.update_renderer()
        
    gc.collect()
    return [mesh_res] if mesh_res else []

def main():
    parser = argparse.ArgumentParser(description="Level 9 Nebula Reconstruction")
    parser.add_argument("--input", type=str, default="output/trained_splats.ply")
    parser.add_argument("--mode", type=str, choices=["orbital", "graph", "impressionist", "crystal", "nebula"], default="orbital")
    parser.add_argument("--output", type=str, default="output/reconstruction.ply")
    parser.add_argument("--radius", type=float, default=0.04, help="Small radius (0.02-0.08)")
    parser.add_argument("--planar", type=float, default=0.15, help="Planar penalty (lower=stricter)")
    parser.add_argument("--alpha", type=float, default=0.03, help="Mesh skinning tightness (Alpha-Shapes)")
    parser.add_argument("--skin_mode", type=str, choices=["bpa", "alpha"], default="bpa")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input): return
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    vis = None
    if args.live:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Level 5 Manifold Research", width=1280, height=720)
        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray([0.85, 0.85, 0.85])
        render_opt.line_width = 3.0
        render_opt.point_size = 4.0

    temp_edge_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin").name
    scene_bounds = (np.min(points, axis=0), np.max(points, axis=0))
    
    print(f"Starting Level 7 Watertight March (Radius={args.radius}, Mode={args.mode}, Skin={args.skin_mode})...", flush=True)
    meshes = adaptive_octree_march(points, colors, args.mode, scene_bounds, radius=args.radius, planar_penalty=args.planar, temp_file=temp_edge_file, vis=vis, live=args.live)

    if args.live:
        print("Live Reconstruction Finished. Close window to save.", flush=True)
        vis.run()
        vis.destroy_window()

    if args.mode in ["orbital", "crystal", "nebula"]:
        if meshes:
            print(f"Merging {len(meshes)} mesh tiles...")
            combined = meshes[0]
            for m in meshes[1:]: combined += m
            
            # Global Island Removal: ONLY for Orbital (manifold)
            # Nebula/Crystal use point-level scrubbing instead
            if args.mode == "orbital":
                print(f"Level 10: Scrubbing floating noise (Global Island Filter)...")
                combined = filter_islands(combined)
            
            o3d.io.write_triangle_mesh(args.output, combined)
            print(f"Success! {args.mode.upper()} model saved to {args.output}")
            
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
            print(f"Graph saved to {args.output}")
            
            # --- LEVEL 7: WATERTIGHT SKINNING ---
            if args.skin_mode == "bpa":
                print(f"Attempting to wrap solid 'BPA-skin' (radius={args.radius})...", flush=True)
                solid_mesh = bpa_skinning(clean_points, clean_colors, radius=args.radius)
            else:
                print(f"Attempting to wrap solid 'Alpha-skin' (tightness={args.alpha})...", flush=True)
                solid_mesh = alpha_skinning(clean_points, clean_colors, alpha=args.alpha)
                if solid_mesh: solid_mesh = filter_islands(solid_mesh)

            if solid_mesh:
                mesh_path = args.output.replace(".ply", "_solid.obj")
                o3d.io.write_triangle_mesh(mesh_path, solid_mesh)
                print(f"WATERTIGHT MODEL saved to {mesh_path}")
                if args.live:
                    print("Showing final WATERTIGHT skin in Live Viewer...", flush=True)
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Level 7 Watertight Result", width=1280, height=720)
                    vis.add_geometry(solid_mesh)
                    render_opt = vis.get_render_option()
                    render_opt.mesh_show_back_face = True
                    vis.run()
                    vis.destroy_window()
            else:
                print("Skinning failed. Try increasing radius or adjusting alpha.")

if __name__ == "__main__":
    main()

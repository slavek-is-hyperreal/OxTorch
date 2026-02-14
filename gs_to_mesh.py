import open3d as o3d
import numpy as np
import os
import argparse
from scipy.spatial import cKDTree
from skimage import measure
import struct

def orbital_field_reconstruction(points, colors, iterations=20, grid_size=128, jitter=0.01):
    """Method A: Extracts a smooth isosurface using statistical jittering into a density grid."""
    print(f"Starting Orbital Field Reconstruction ({iterations} iterations)...")
    
    # Calculate bounds
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    # Add padding
    margin = (max_bound - min_bound) * 0.1
    min_bound -= margin
    max_bound += margin
    
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    color_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    
    for i in range(iterations):
        if i % 5 == 0: print(f"  Iteration {i}/{iterations}...")
        # Brownian Jitter
        jittered_points = points + np.random.normal(0, jitter, points.shape)
        
        # Maps points to grid coordinates
        indices = ((jittered_points - min_bound) / (max_bound - min_bound) * (grid_size - 1)).astype(int)
        
        # Valid indices mask
        valid = np.all((indices >= 0) & (indices < grid_size), axis=1)
        indices = indices[valid]
        c_vals = colors[valid]
        
        # Vectorized accumulation
        # Note: In a production environment, we'd use a more sophisticated 3D Gaussian kernel
        # For our research, we use atomic addition to voxels
        for idx, col in zip(indices, c_vals):
            grid[idx[0], idx[1], idx[2]] += 1.0
            color_grid[idx[0], idx[1], idx[2]] += col

    # Extract Isosurface
    threshold = np.mean(grid) + np.std(grid) # Simple threshold heuristic
    print(f"Extracting isosurface at threshold {threshold:.2f}...")
    
    try:
        verts, faces, normals, values = measure.marching_cubes(grid, level=threshold)
        
        # Map vertices back to world space
        verts = verts / (grid_size - 1) * (max_bound - min_bound) + min_bound
        
        # Interpolate vertex colors from color_grid
        v_indices = (values / iterations).astype(int) # Placeholder for simple coloring
        # Actually we should sample color_grid at vert positions
        v_idx_raw = ((verts - min_bound) / (max_bound - min_bound) * (grid_size - 1)).astype(int)
        v_idx_raw = np.clip(v_idx_raw, 0, grid_size-1)
        vert_colors = color_grid[v_idx_raw[:, 0], v_idx_raw[:, 1], v_idx_raw[:, 2]] / iterations
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vert_colors, 0, 1))
        
        return mesh
    except Exception as e:
        print(f"Marching Cubes failed: {e}")
        return None

def graph_statistical_connectivity(points, colors, iterations=100, radius=0.05):
    """Method B: Builds a persistent graph by counting connections surviving statistical jitter."""
    print(f"Starting Graph Statistical Connectivity (R={radius})...")
    
    num_points = len(points)
    # Using a sparse adjacency count would be ideal, 
    # but for simplicity on small clouds, we use a fixed distance check.
    # We will track the "Persistence" of edges.
    
    tree = cKDTree(points)
    # Initial graph candidate edges
    pairs = tree.query_pairs(radius)
    edge_counts = {pair: 0 for pair in pairs}
    
    print(f"Analyzing {len(pairs)} candidate edges over {iterations} jitters...")
    for i in range(iterations):
        if i % 20 == 0: print(f"  Jitter {i}/{iterations}...")
        jittered = points + np.random.normal(0, radius*0.2, points.shape)
        
        for pair in edge_counts:
            d = np.linalg.norm(jittered[pair[0]] - jittered[pair[1]])
            if d < radius:
                edge_counts[pair] += 1
                
    # Filter stable edges
    threshold = int(iterations * 0.7)
    stable_edges = [pair for pair, count in edge_counts.items() if count >= threshold]
    print(f"Retained {len(stable_edges)} stable edges out of {len(pairs)} candidates.")
    
    # Create LineSet for O3D visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(stable_edges))
    line_set.colors = o3d.utility.Vector3dVector(colors) # Edge colors from points
    
    return line_set

def main():
    parser = argparse.ArgumentParser(description="Impressionistic Reconstruction Suite")
    parser.add_argument("--input", type=str, default="output/trained_splats.ply")
    parser.add_argument("--mode", type=str, choices=["orbital", "graph", "impressionist"], default="orbital")
    parser.add_argument("--output", type=str, default="output/reconstruction.ply")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if args.mode == "orbital":
        result = orbital_field_reconstruction(points, colors)
        if result: o3d.io.write_triangle_mesh(args.output, result)
        
    elif args.mode == "graph":
        result = graph_statistical_connectivity(points, colors)
        o3d.io.write_line_set(args.output, result)
        
    elif args.mode == "impressionist":
        print("Mode C: Impressionistic Hybrid. Generating both...")
        mesh = orbital_field_reconstruction(points, colors, grid_size=64)
        graph = graph_statistical_connectivity(points, colors, iterations=50)
        
        # Save both or a merged representation
        o3d.io.write_triangle_mesh("output/impresjonista_surface.ply", mesh)
        o3d.io.write_line_set("output/impresjonista_graph.ply", graph)
        print("Impressionistic components saved to output/impresjonista_*.ply")

    print(f"Done! Result saved to {args.output} (or components in output/)")

if __name__ == "__main__":
    main()

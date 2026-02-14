import open3d as o3d
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Gaussian Splatting PLY to 3D Mesh")
    parser.add_argument("--input", type=str, default="output/trained_splats.ply", help="Input PLY file")
    parser.add_argument("--output", type=str, default="output/room_mesh.obj", help="Output OBJ file")
    parser.add_argument("--depth", type=int, default=9, help="Poisson reconstruction depth (higher = more detail)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print(f"Loading point cloud from {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Points: {len(pcd.points)}")

    # 1. Statistical Outlier Removal (SOR)
    print("Step 1: Statistical Outlier Removal (SOR)...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print(f"Points after SOR: {len(pcd.points)}")

    # 2. Radius Outlier Removal (ROR)
    print("Step 2: Radius Outlier Removal (ROR)...")
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=1.0) # Adjust radius based on scene scale
    pcd = pcd.select_by_index(ind)
    print(f"Points after ROR: {len(pcd.points)}")

    # 3. Normal Estimation (Required for Poisson)
    print("Step 3: Estimating Normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

    # 4. Poisson Surface Reconstruction
    print(f"Step 4: Poisson Surface Reconstruction (depth={args.depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    print(f"Mesh generated with {len(mesh.vertices)} vertices.")

    # 5. Density-based Filtering (Remove "ghost" triangles)
    print("Step 5: Density-based Filtering...")
    densities = np.asarray(densities)
    # Filter out vertices with low density (confidence)
    vertices_to_remove = densities < np.quantile(densities, 0.1) # Remove bottom 10% density
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"Final mesh: {len(mesh.vertices)} vertices.")

    # 6. Export
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving mesh to {args.output}...")
    # OBJ is a good choice for standard apps, but PLY preserves vertex colors better in some viewers
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done!")

if __name__ == "__main__":
    main()

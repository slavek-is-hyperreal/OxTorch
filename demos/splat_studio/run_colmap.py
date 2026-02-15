import os
import argparse
import subprocess
import shutil

def run_command(command):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def run_sfm(image_path, output_path, use_gpu=0):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    database_path = os.path.join(output_path, "database.db")
    sparse_path = os.path.join(output_path, "sparse")
    
    if not os.path.exists(sparse_path):
        os.makedirs(sparse_path)

    # 1. Feature extraction
    run_command([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--SiftExtraction.use_gpu", str(use_gpu)
    ])

    # 2. Matching (Sequential for video)
    run_command([
        "colmap", "sequential_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", str(use_gpu),
        "--SequentialMatching.overlap", "10"  # Match with 10 surrounding frames
    ])

    # 3. Mapping
    run_command([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sparse_path
    ])

    # 4. Export to PLY (check first if folder 0 exists)
    model_path = os.path.join(sparse_path, "0")
    if os.path.exists(model_path):
        run_command([
            "colmap", "model_converter",
            "--input_path", model_path,
            "--output_path", os.path.join(output_path, "sparse_cloud.ply"),
            "--output_type", "PLY"
        ])
        print(f"SfM completed. Sparse cloud saved to {os.path.join(output_path, 'sparse_cloud.ply')}")
    else:
        print("Error: Reconstruction failed (no model generated).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP SfM pipeline.")
    parser.add_argument("--images", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output", type=str, default="output/colmap", help="Directory to save COLMAP output.")
    parser.add_argument("--use_gpu", type=int, default=0, help="Use GPU for SIFT (0=No, 1=Yes).")
    
    args = parser.parse_args()
    run_sfm(args.images, args.output, args.use_gpu)

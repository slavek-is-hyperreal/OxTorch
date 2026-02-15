import os
import argparse
import subprocess

def extract_frames(video_path, output_dir, fps=8, prefix="frame"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ffmpeg -i input_video.mp4 -vf "fps=8" frames/prefix_%04d.png
    output_pattern = os.path.join(output_dir, f"{prefix}_%04d.png")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality
        output_pattern
    ]
    
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)
    print(f"Extracted frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video using ffmpeg at a specific FPS.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="output/frames", help="Directory to save extracted frames.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second to extract.")
    parser.add_argument("--prefix", type=str, default="frame", help="Prefix for output filenames.")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps, args.prefix)

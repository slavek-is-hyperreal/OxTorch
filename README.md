# Splat Editor - Gaussian Splatting for Linux

Transform handheld videos into realistic 3D Gaussian Splat scenes with a single click.

## Features
- **Unified Pipeline**: Automates frame extraction, COLMAP sparse reconstruction, and Gaussian training.
- **AI Depth Auto-Density**: Uses AI (CPU-based) to fill gaps in geometry for a much denser, smoother reconstruction.
- **Low VRAM Optimization**: Designed to run on 2GB GPUs (Vulkan/Taichi).
- **Dynamic Visualization**: Brownian motion, pulsing, and blur effects for realistic results from sparse data.
- **Progress Tracking**: Real-time feedback for all processing stages.

## Installation
Ensure you have the following requirements installed:
```bash
sudo apt update
sudo apt install colmap ffmpeg python3-tk
pip install taichi open3d pillow numpy
```

## How to Use
1. Run the Editor:
   ```bash
   source venv/bin/activate
   python3 splat_studio.py
   ```
2. **Add Videos**: Select one or more `.mp4` files from your phone.
3. **Configure**: Set FPS (15-24 recommended) and Iterations (1000-10000). 
4. **AI Enhancement**: Keep the "AI Depth Auto-Density" toggle ON for best results (Surface filling).
5. **Start**: Watch the progress!
5. **Save & View**: Once done, save your `.ply` model and the viewer will launch automatically.

## Viewer Controls
- `WASD`: Move camera.
- `RMB + Mouse`: Rotate.
- `Splat Mode`: High-quality Gaussian blending.
- `Brownian Motion`: Jitter effect to fill gaps.
- `Density Mult`: Artificial point multiplier for fuller scenes.
- `Blur Intensity`: Post-filter smoothing.

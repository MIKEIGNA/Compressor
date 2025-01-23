import os
import cv2
import pywt
import numpy as np
import subprocess
from tqdm import tqdm

def wavelet_preprocess(frame, wavelet='haar', level=2):
    """Apply wavelet-based noise reduction or downscaling to a frame."""
    coeffs = pywt.wavedec2(frame, wavelet=wavelet, level=level, axes=(0, 1))
    # Keep only the approximation coefficients (reduce detail)
    processed_frame = pywt.waverec2(coeffs[:1] + [None] * (len(coeffs) - 1), wavelet=wavelet, axes=(0, 1))
    return np.clip(processed_frame, 0, 255).astype(np.uint8)

def compress_video_with_preprocessing(input_video, output_video, wavelet=None, level=2, codec="libx265", crf=28):
    """
    Compress a video with optional wavelet preprocessing and FFmpeg compression.

    Args:
        input_video: Path to the input video.
        output_video: Path to the output compressed video.
        wavelet: Optional wavelet type for preprocessing.
        level: Level of wavelet decomposition for preprocessing.
        codec: Codec to use for FFmpeg compression (default: H.265).
        crf: Constant Rate Factor for FFmpeg (lower = better quality, default: 28).

    Returns:
        None
    """
    try:
        # Setup temporary directory for processed frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        # Read input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {input_video}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pre-process and save frames
        print("Preprocessing frames...")
        for i in tqdm(range(frame_count), desc="Preprocessing"):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if wavelet:
                frame_rgb = wavelet_preprocess(frame_rgb, wavelet, level)
            preprocessed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{temp_dir}/frame_{i:05d}.png", preprocessed_frame)
        cap.release()

        # Use FFmpeg to compress processed frames
        print("Compressing video...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", f"{temp_dir}/frame_%05d.png",
            "-c:v", codec,
            "-preset", "slow",
            "-crf", str(crf),
            output_video
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Clean up temporary frames
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        print(f"Compression complete. Output saved to {output_video}")
    except Exception as e:
        print(f"Error during compression: {e}")

# Example Usage
input_video_path = "sample_video.mp4"  # Replace with your video file path
output_compressed_video = "compressed_video.mp4"

compress_video_with_preprocessing(
    input_video=input_video_path,
    output_video=output_compressed_video,
    wavelet='haar',  # Use wavelet preprocessing
    level=2,
    codec="libx265",
    crf=28  # Higher CRF for better compression
)

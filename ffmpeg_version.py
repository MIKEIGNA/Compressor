import os
import cv2
import subprocess
from tqdm import tqdm

def compress_video(input_video, output_video, target_resolution=(1280, 720), codec="libx265", crf=28):
    """
    Compress a video by resizing frames and using FFmpeg for efficient encoding.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the compressed video.
        target_resolution (tuple): Desired resolution for the compressed video (width, height).
        codec (str): Video codec for compression (default: H.265 / libx265).
        crf (int): Constant Rate Factor for FFmpeg (lower = better quality, default: 28).

    Returns:
        None
    """
    try:
        # Create a temporary directory to store resized frames
        temp_dir = "temp_compression_frames"
        os.makedirs(temp_dir, exist_ok=True)

        # Open the input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_video}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Original Resolution: {original_width}x{original_height}, Target Resolution: {target_resolution}")
        print("Resizing frames...")

        # Process frames and resize them
        for frame_idx in tqdm(range(frame_count), desc="Processing Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, resized_frame)

        cap.release()

        # Use FFmpeg to compress the resized frames into a video
        print("Compressing video...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%05d.png"),
            "-c:v", codec,
            "-preset", "medium",
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
    finally:
        # Cleanup if an error occurs
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

# Example Usage
input_video_path = "sample_video.mp4"  # Replace with your video file path
output_compressed_video = "compressed_video.mp4"
# 3840x2160 or 1280x720
compress_video(
    input_video=input_video_path,
    output_video=output_compressed_video,
    target_resolution=(3840, 2160),  # Reduce resolution for compression
    codec="libx265",  # Use H.265 codec for better compression
    crf=28  # Adjust CRF for quality (lower = higher quality, larger size)
)

print("Video compression process complete.")

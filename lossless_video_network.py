import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
import socket  # For the optional network-sending example

def is_duplicate_frame(current_frame, previous_frame, threshold=2.0):
    """
    Check if the current_frame is near-duplicate of the previous_frame by comparing mean absolute difference.

    Args:
        current_frame (np.ndarray): BGR frame from OpenCV.
        previous_frame (np.ndarray): BGR frame from OpenCV.
        threshold (float): The difference threshold below which frames are considered duplicates.

    Returns:
        bool: True if the frames are near-identical, False otherwise.
    """
    if previous_frame is None:
        return False

    # Convert frames to float32 for difference calculation
    diff = np.mean(np.abs(current_frame.astype(np.float32) - previous_frame.astype(np.float32)))
    return diff < threshold

def compress_video_lossless_no_quality(
    input_video: str,
    output_video: str,
    skip_duplicates: bool = True,
    difference_threshold: float = 2.0,
    codec: str = "libx264",
    extra_codec_flags = ("-qp", "0"),
):
    """
    Compress a video losslessly, skipping near-duplicate frames to save space.
    This results in no quality loss for the frames that are actually saved.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path for the final compressed video.
        skip_duplicates (bool): Whether to skip frames that are nearly the same as the previous one.
        difference_threshold (float): If the mean abs difference is below this, treat frames as duplicates.
        codec (str): FFmpeg codec to use for truly lossless compression (e.g. 'libx264', 'libx265', 'ffv1').
        extra_codec_flags: Additional flags to ensure lossless mode. 
                          - For H.264: ('-qp','0')
                          - For H.265: ('-x265-params','lossless=1')
                          - For ffv1: no extra flags needed typically.
    """
    temp_dir = "temp_lossless_frames"
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Opened {input_video} with {frame_count} frames at {fps:.2f} FPS.")
    prev_frame = None

    # Extract frames with skip-duplicates
    used_frames = 0
    print("Extracting frames (and optionally skipping duplicates)...")
    for i in tqdm(range(frame_count), desc="Extracting"):
        ret, frame = cap.read()
        if not ret:
            break
        if not skip_duplicates:
            # Always save frame
            pass
        else:
            if is_duplicate_frame(frame, prev_frame, difference_threshold):
                # Skip writing
                continue
        outpath = os.path.join(temp_dir, f"frame_{i:05d}.png")
        cv2.imwrite(outpath, frame)
        prev_frame = frame
        used_frames += 1
    cap.release()

    # Use FFmpeg to encode all extracted frames in lossless mode
    print("Encoding frames with FFmpeg in lossless mode...")

    # Build FFmpeg command
    # Example: 
    # ffmpeg -y -framerate FPS -i frame_%05d.png -c:v libx264 -qp 0 output_video
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-framerate", str(int(fps)),
        "-i", os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v", codec,
        "-preset", "slow",
    ]
    # Add extra flags for truly lossless
    ffmpeg_cmd.extend(list(extra_codec_flags))
    ffmpeg_cmd.append(output_video)

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Lossless compression complete. Output: {output_video}")

    # Clean up
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    print("Temporary frames removed. Done.")

def send_file_over_tcp(filepath: str, host='127.0.0.1', port=9999):
    """
    Example function to demonstrate sending a file over a simple TCP socket.
    This can be adapted to send your compressed video across the network.

    Args:
        filepath (str): Path to the file to send.
        host (str): Destination host.
        port (int): Destination port.
    """
    filesize = os.path.getsize(filepath)
    print(f"Sending {filepath} of size {filesize} bytes to {host}:{port} ...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # Send file size first
        s.sendall(f"{filesize}\n".encode('utf-8'))

        with open(filepath, 'rb') as f:
            # send file in chunks
            while True:
                data = f.read(4096)
                if not data:
                    break
                s.sendall(data)
    print("File transfer complete.")


if __name__ == "__main__":
    # Example usage
    input_video_path = "sample_video.mp4"  # The input video
    output_video_path = "lossless_out.mp4"

    # Lossless compression with H.264
    # Setting '-qp 0' ensures no quantization => truly lossless
    compress_video_lossless_no_quality(
        input_video=input_video_path,
        output_video=output_video_path,
        skip_duplicates=True,
        difference_threshold=2.0,
        codec='libx264',
        extra_codec_flags=('-qp','0')  # H.264 Lossless
    )

    # Optionally send the output video to a remote server
    # send_file_over_tcp(output_video_path, host='192.168.1.10', port=9000)

    print("All tasks complete.")

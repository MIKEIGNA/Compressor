import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
import sys

def frames_are_duplicate(frame_a, frame_b, threshold=2.0):
    """
    Determine if frame_a is near-duplicate of frame_b by measuring mean absolute difference.
    """
    if frame_b is None:
        return False
    diff = np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32)))
    return diff < threshold

def compress_video_lossless_in_memory(input_video,
                                      output_video,
                                      skip_duplicates=True,
                                      difference_threshold=2.0,
                                      codec="libx264",
                                      ffmpeg_codec_flags=("-qp", "0")):
    """
    Compress a video in a truly lossless manner, skipping near-duplicate frames,
    all in-memory without writing temp frames.

    Args:
        input_video (str): Path to the input video.
        output_video (str): Path to the output compressed video.
        skip_duplicates (bool): If True, skip frames that appear nearly identical to the previous one.
        difference_threshold (float): Threshold for deciding duplicates (lower => less skipping).
        codec (str): FFmpeg codec to use in a truly lossless mode (e.g. 'libx264').
        ffmpeg_codec_flags (tuple): Extra flags for the codec (e.g., '-qp', '0') for H.264 lossless.

    Returns:
        None
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Opened video: {input_video}")
    print(f" - Detected {frame_count} frames at {fps:.2f} FPS.")
    print("Start reading frames and piping to FFmpeg...")

    # Build FFmpeg command for reading images from pipe
    # -f image2pipe => read images from STDIN
    # -c:v {codec}, plus extra flags => ensure lossless
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                # overwrite
        "-f", "image2pipe",  # read image sequence from pipe
        "-framerate", str(int(fps)),
        "-i", "pipe:0",      # input from stdin
        "-c:v", codec,
        "-preset", "medium", # or "slow" for potentially smaller file
    ]
    ffmpeg_cmd.extend(list(ffmpeg_codec_flags))
    ffmpeg_cmd.append(output_video)

    # Start FFmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    prev_frame = None
    used_frames = 0

    # We read frames and pass them to ffmpeg via pipe
    # Encoding each frame as PNG in-memory for truly lossless transfer
    for i in tqdm(range(frame_count), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        if skip_duplicates and frames_are_duplicate(frame, prev_frame, threshold=difference_threshold):
            continue

        # Convert BGR -> PNG in memory
        ret2, png_data = cv2.imencode(".png", frame)
        if not ret2:
            print(f"Warning: Could not encode frame {i} as PNG.")
            continue

        # Write PNG bytes to ffmpeg STDIN
        ffmpeg_process.stdin.write(png_data.tobytes())
        prev_frame = frame
        used_frames += 1

    cap.release()

    # Finally, close STDIN => signals FFmpeg no more frames
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    print(f"Lossless compression done. {used_frames} frames used. Output => {output_video}")

# Example usage
if __name__ == "__main__":
    input_video_path = "sample_video.mp4"   # Input
    output_video_path = "output_lossless.mp4"

    compress_video_lossless_in_memory(
        input_video=input_video_path,
        output_video=output_video_path,
        skip_duplicates=True,         # skip near-identical frames
        difference_threshold=2.0,     # tune as needed
        codec="libx264",              # or 'libx265'
        ffmpeg_codec_flags=("-qp","0")# for true lossless in H.264
    )

    print("All done.")

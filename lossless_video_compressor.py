import subprocess
import os

def compress_video_optimized(
    input_video: str,
    output_video: str,
    codec: str = "libx265",
    crf: int = 28,
    preset: str = "veryslow",
    extra_ffmpeg_flags: list = None
):
    """
    Compress a video efficiently using FFmpeg.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output compressed video file.
        codec (str): Video codec ('libx264' for H.264 or 'libx265' for H.265).
        crf (int): Compression level (higher = smaller file, lower = better quality).
        preset (str): Speed/quality trade-off ('ultrafast', 'medium', 'veryslow').
        extra_ffmpeg_flags (list): Additional FFmpeg parameters.

    Returns:
        None
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",  # Ensures better compatibility and size reduction
        "-b:v", "1M",  # Limit the bitrate to further compress
        "-bf", "2",  # Enable adaptive B-frames
        "-tune", "fastdecode"  # Optimize for smaller size and fast decoding
    ]

    if extra_ffmpeg_flags:
        ffmpeg_cmd.extend(extra_ffmpeg_flags)

    ffmpeg_cmd.append(output_video)

    print("Running FFmpeg command:")
    print(" ".join(ffmpeg_cmd))

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Compression complete. Output saved as {output_video}")

# Example usage
if __name__ == "__main__":
    input_video_path = "sample_video.mp4"  # Replace with your actual video file
    output_video_path = "compressed_output.mp4"

    compress_video_optimized(
        input_video=input_video_path,
        output_video=output_video_path,
        codec="libx265",  # Switch to H.265 for better compression
        crf=28,  # Adjust CRF for more compression (20-28 recommended)
        preset="veryslow",  # Use 'veryslow' for better compression efficiency
        extra_ffmpeg_flags=None
    )

    print("All done!")

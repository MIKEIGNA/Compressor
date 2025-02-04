import subprocess
import os

def compress_video_simple(
    input_video: str,
    output_video: str,
    codec: str = "libx264",
    crf: int = 20,
    preset: str = "medium",
    extra_ffmpeg_flags: list = None
):
    """
    Compress a video in a single pass using FFmpeg, minimal overhead, suitable for low-end PCs.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output compressed video file.
        codec (str): FFmpeg codec (e.g., 'libx264' for H.264, 'libx265' for H.265).
        crf (int): Constant Rate Factor [0-51], lower => better quality + bigger size.
        preset (str): Speed/quality trade-off. e.g., 'ultrafast', 'fast', 'medium', 'slow'.
        extra_ffmpeg_flags (list): Additional arguments to pass to FFmpeg if needed (e.g. ['-pix_fmt','yuv420p']).

    Returns:
        None
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Basic FFmpeg command:
    #   ffmpeg -i input_video -c:v codec -preset preset -crf crf output_video
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_video,
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
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
    # Adjust these paths as needed
    input_video_path = "sample_video.mp4"
    output_video_path = "compressed_output.mp4"

    # For near-lossless H.264 on a low-end PC:
    # - Use a moderate CRF (18-24 is typical for near transparency)
    # - Use a faster preset if your CPU is very limited (e.g. 'fast' or 'ultrafast')

    compress_video_simple(
        input_video=input_video_path,
        output_video=output_video_path,
        codec="libx264",
        crf=20,          # Good balance of quality & size (approx 50% reduction typical)
        preset="medium", # For slower PCs, pick "fast" or "ultrafast"
        extra_ffmpeg_flags=None
        # e.g. extra_ffmpeg_flags=["-pix_fmt", "yuv420p"]
    )

    print("All done!")

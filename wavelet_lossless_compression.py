import os
import cv2
import pywt
import numpy as np
import subprocess
from tqdm import tqdm

def wavelet_transform_lossless(frame, wavelet='haar', level=1):
    """
    Convert a frame into a wavelet-based color representation without discarding information.
    We do not threshold or quantize, ensuring lossless transform.

    Args:
        frame (np.ndarray): The input frame as uint8 RGB.
        wavelet (str): Type of wavelet to apply. 'haar' is simplest.
        level (int): Decomposition level. Usually 1 for minimal overhead in a lossless pipeline.

    Returns:
        transformed_frame (np.ndarray): A float32 array that reorganizes color data in wavelet form.
    """
    # Split into color channels, wavelet transform each channel
    # Keep all subbands intact, ensuring no threshold => no data loss
    channels = cv2.split(frame)
    wavelet_coeffs = []
    for ch in channels:
        # Convert channel to float32 to handle wavelet transform
        ch_float = ch.astype(np.float32)
        coeffs = pywt.wavedec2(ch_float, wavelet=wavelet, level=level)
        wavelet_coeffs.append(coeffs)

    # Combine all subbands from each channel into a single 2D array (for storage)
    # We'll store them by flattening each subband and stacking them. 
    # The shape info is needed to reverse the transform exactly.
    flattened_subbands = []
    shape_info = []

    for c in wavelet_coeffs:
        # c[0] is the approximation subband
        # c[1:] are detail subbands (tuples of horizontal, vertical, diagonal)
        # We store shape info for each subband, then flatten it
        all_subbands = []
        for subband in c:
            if isinstance(subband, tuple):
                for sb in subband:
                    shape_info.append(sb.shape)
                    all_subbands.append(sb.flatten())
            else:
                shape_info.append(subband.shape)
                all_subbands.append(subband.flatten())

        # stack and keep track
        flattened_subbands.append(np.concatenate(all_subbands))

    # Combine flattened data from each color channel
    combined = np.concatenate(flattened_subbands)

    # We'll form a single 1D array of float32, plus we store shape_info to reconstruct
    transformed_frame = combined.astype(np.float32)
    return transformed_frame, wavelet_coeffs, shape_info

def inverse_wavelet_transform_lossless(transformed_frame, shape_info, wavelet_coeffs, wavelet='haar', level=1):
    """
    Reverse the wavelet transform to restore the original frame.

    Args:
        transformed_frame (np.ndarray): The combined flattened wavelet data (float32).
        shape_info (list of tuples): The shapes of each subband to reconstruct them accurately.
        wavelet_coeffs (list): A template with the same structure as the original wavelet coeffs
                               (used to determine how many subbands per channel).
        wavelet (str): The wavelet used.
        level (int): The wavelet level used.

    Returns:
        reconstructed_frame (np.ndarray): The restored uint8 RGB image.
    """
    # We'll sequentially carve up the flattened array based on shape_info
    idx = 0
    full_channels = []

    for channel_coeffs in wavelet_coeffs:
        # channel_coeffs is something like [approx, (h1, v1, d1), (h2, v2, d2), ...]
        # We'll replicate that structure with shapes from shape_info
        channel_subbands = []
        for c_subband in channel_coeffs:
            if isinstance(c_subband, tuple):
                sub_tuple = []
                for _ in c_subband:
                    shape = shape_info[idx]
                    size = np.prod(shape)
                    subband_data = transformed_frame[idx : idx + size].reshape(shape)
                    idx += size
                    sub_tuple.append(subband_data)
                channel_subbands.append(tuple(sub_tuple))
            else:
                # single subband
                shape = shape_info[idx]
                size = np.prod(shape)
                subband_data = transformed_frame[idx : idx + size].reshape(shape)
                idx += size
                channel_subbands.append(subband_data)

        # Now channel_subbands is a wavelet-like structure
        # We can run pywt.waverec2 to get back the original channel
        ch_float = pywt.waverec2(channel_subbands, wavelet=wavelet)
        ch_uint8 = np.clip(ch_float, 0, 255).astype(np.uint8)
        full_channels.append(ch_uint8)

    # Merge the channels to form an RGB image
    reconstructed_frame = cv2.merge(full_channels)
    return reconstructed_frame

def compress_video_lossless_wavelet(
    input_video, output_video, wavelet='haar', level=1, codec='ffv1'
):
    """
    Perform a wavelet transform on each frame (losslessly, no threshold),
    store them as PNG, then encode with a truly lossless codec (FFV1) using FFmpeg.

    Args:
        input_video (str): Path to the input video.
        output_video (str): Path to the output video file.
        wavelet (str): Type of wavelet to use (default: haar).
        level (int): Wavelet decomposition level. 1 is minimal overhead.
        codec (str): A lossless codec for FFmpeg (e.g., 'ffv1', 'libx264 -qp 0', etc.).
    """
    import pickle
    import numpy as np
    import subprocess
    import os

    temp_dir = "temp_wavelet_frames"
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # For shape consistency
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("No frames to read in video.")
    frame_height, frame_width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start

    # Step 1: Convert each frame to wavelet form and store them as frames for pipeline
    # We'll store them as PNG to preserve data, but with a slight overhead. Another approach is to store as raw data.
    # For demonstration, let's do PNG.

    print("Applying wavelet transform and storing frames as PNG...")
    for idx in tqdm(range(frame_count), desc="Wavelet Transforming"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wavelet transform (no threshold => no data loss)
        transformed_data, wavelet_coeffs, shape_info = wavelet_transform_lossless(frame_rgb, wavelet, level)

        # We store (transformed_data, shape_info, wavelet_coeffs) in memory, then reconstruct a wavelet-coded image for PNG
        # Because we are going for 100% lossless, let's do a test reconstruction here
        # so we keep a single representation (like a wavelet-coded image)...

        # Reconstruct the wavelet-coded frame to an actual RGB
        # This might be identical to the original if truly no threshold
        recon_frame = inverse_wavelet_transform_lossless(transformed_data, shape_info, wavelet_coeffs, wavelet, level)

        # Convert recon_frame back to BGR for consistent storing
        recon_bgr = cv2.cvtColor(recon_frame, cv2.COLOR_RGB2BGR)
        outpath = os.path.join(temp_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(outpath, recon_bgr)

    cap.release()

    # Step 2: Use FFmpeg to encode these PNG frames with a lossless codec
    print(f"Encoding with {codec} in lossless mode (FFmpeg)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v", codec,
        # Some codecs need extra flags for true lossless, e.g. for H.264: '-qp', '0'
        # For 'ffv1', it's inherently lossless
        output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Lossless wavelet-based compression completed. Output: {output_video}")

    # Step 3: Cleanup
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print("All temporary data removed.")

# Additional Helper Routines for Wavelet
def wavelet_transform_lossless(frame, wavelet='haar', level=1):
    """
    Convert a frame into wavelet-based representation with no data loss.
    We do no thresholding => 'lossless' approach in wavelet domain.

    Returns:
        transformed_frame (np.ndarray): Flattened wavelet data
        wavelet_coeffs (list): A template structure to re-inject shape
        shape_info (list): Shapes of each subband
    """
    import pywt
    import numpy as np

    # Split channels
    # Validate input
    if frame.dtype != np.uint8:
        raise ValueError("Frame must be uint8.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be an HxWx3 color image.")

    # Convert each channel to float32
    channels = cv2.split(frame)
    wavelet_coeffs = []
    for ch in channels:
        ch_float = ch.astype(np.float32)
        coeffs = pywt.wavedec2(ch_float, wavelet=wavelet, level=level)
        wavelet_coeffs.append(coeffs)

    # We'll flatten them but keep shape info
    flattened_subbands = []
    shape_info = []

    for c in wavelet_coeffs:
        for subband in c:
            if isinstance(subband, tuple):
                # detail tuple
                for sb in subband:
                    shape_info.append(sb.shape)
                    flattened_subbands.append(sb.flatten())
            else:
                # approximation
                shape_info.append(subband.shape)
                flattened_subbands.append(subband.flatten())

    combined = np.concatenate(flattened_subbands).astype(np.float32)
    return combined, wavelet_coeffs, shape_info

def inverse_wavelet_transform_lossless(transformed_frame, shape_info, wavelet_coeffs, wavelet='haar', level=1):
    """
    Rebuild the original frame from wavelet-coded data.

    reversed from wavelet_transform_lossless
    """
    import pywt
    import numpy as np

    idx = 0
    out_channels = []

    for c in wavelet_coeffs:
        # Reconstruct the subbands
        new_subbands = []
        for subband in c:
            if isinstance(subband, tuple):
                detail_tuple = []
                for _ in subband:
                    shape = shape_info[idx]
                    size = np.prod(shape)
                    data_chunk = transformed_frame[idx : idx+size].reshape(shape)
                    idx += size
                    detail_tuple.append(data_chunk)
                new_subbands.append(tuple(detail_tuple))
            else:
                # single array
                shape = shape_info[idx]
                size = np.prod(shape)
                data_chunk = transformed_frame[idx : idx+size].reshape(shape)
                idx += size
                new_subbands.append(data_chunk)

        # new_subbands is the wavelet structure
        ch_float = pywt.waverec2(new_subbands, wavelet=wavelet)
        ch_uint8 = np.clip(ch_float, 0, 255).astype(np.uint8)
        out_channels.append(ch_uint8)

    # Merge
    rec_frame = cv2.merge(out_channels)
    return rec_frame

if __name__ == "__main__":
    # Example usage with wavelet-based approach for each frame, then lossless encode
    input_video_path = "sample_video.mp4"
    output_video_path = "output_lossless_wavelet.mp4"

    # 'ffv1' is a truly lossless codec. You can also try 'libx264' with '-qp 0' flags, 
    # but that might require extra options in ffmpeg_cmd if you'd prefer x264:
    # e.g.:
    #    ffmpeg_cmd = ['ffmpeg', '-i', 'frame_%05d.png', '-c:v', 'libx264', '-qp', '0', 'output.mp4']

    compress_video_lossless_wavelet(
        input_video=input_video_path,
        output_video=output_video_path,
        wavelet='haar',
        level=1,
        codec='ffv1'  # you can use 'libx264 -qp 0' if you want H.264 lossless
    )

    print("Done.")

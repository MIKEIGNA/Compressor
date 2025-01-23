# import os
# import cv2
# import pywt
# import numpy as np
# import subprocess
# from tqdm import tqdm

# def wavelet_preprocess(frame, wavelet='haar', level=2):
#     """Apply wavelet-based noise reduction or downscaling to a frame."""
#     coeffs = pywt.wavedec2(frame, wavelet=wavelet, level=level, axes=(0, 1))
#     # Keep only the approximation coefficients (reduce detail)
#     processed_frame = pywt.waverec2(coeffs[:1] + [None] * (len(coeffs) - 1), wavelet=wavelet, axes=(0, 1))
#     return np.clip(processed_frame, 0, 255).astype(np.uint8)

# def compress_video_with_preprocessing(input_video, output_video, wavelet=None, level=2, codec="libx265", crf=28):
#     """
#     Compress a video with optional wavelet preprocessing and FFmpeg compression.

#     Args:
#         input_video: Path to the input video.
#         output_video: Path to the output compressed video.
#         wavelet: Optional wavelet type for preprocessing.
#         level: Level of wavelet decomposition for preprocessing.
#         codec: Codec to use for FFmpeg compression (default: H.265).
#         crf: Constant Rate Factor for FFmpeg (lower = better quality, default: 28).

#     Returns:
#         None
#     """
#     try:
#         # Setup temporary directory for processed frames
#         temp_dir = "temp_frames"
#         os.makedirs(temp_dir, exist_ok=True)

#         # Read input video
#         cap = cv2.VideoCapture(input_video)
#         if not cap.isOpened():
#             raise Exception(f"Cannot open video: {input_video}")
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Pre-process and save frames
#         print("Preprocessing frames...")
#         for i in tqdm(range(frame_count), desc="Preprocessing"):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             if wavelet:
#                 frame_rgb = wavelet_preprocess(frame_rgb, wavelet, level)
#             preprocessed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(f"{temp_dir}/frame_{i:05d}.png", preprocessed_frame)
#         cap.release()

#         # Use FFmpeg to compress processed frames
#         print("Compressing video...")
#         ffmpeg_cmd = [
#             "ffmpeg",
#             "-y",
#             "-framerate", str(fps),
#             "-i", f"{temp_dir}/frame_%05d.png",
#             "-c:v", codec,
#             "-preset", "slow",
#             "-crf", str(crf),
#             output_video
#         ]
#         subprocess.run(ffmpeg_cmd, check=True)

#         # Clean up temporary frames
#         for file in os.listdir(temp_dir):
#             os.remove(os.path.join(temp_dir, file))
#         os.rmdir(temp_dir)

#         print(f"Compression complete. Output saved to {output_video}")
#     except Exception as e:
#         print(f"Error during compression: {e}")

# # Example Usage
# input_video_path = "sample_video.mp4"  # Replace with your video file path
# output_compressed_video = "compressed_video.mp4"

# compress_video_with_preprocessing(
#     input_video=input_video_path,
#     output_video=output_compressed_video,
#     wavelet='haar',  # Use wavelet preprocessing
#     level=2,
#     codec="libx265",
#     crf=28  # Higher CRF for better compression
# )

# print('done with compress_video_with_preprocessing')


import os
import cv2
import numpy as np
import pywt
import pickle
import bz2
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compress_image(image, wavelet='haar', level=3, quantization_factor=100):
    """Compress an image using wavelet transform."""
    try:
        assert image.ndim == 3, "Image must have three dimensions (H, W, C)."
        assert image.shape[2] == 3, "Image must have three color channels."
        assert image.dtype == np.uint8, "Image data type must be uint8."

        coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
        if len(coeffs) < 2 or not all(isinstance(c, tuple) for c in coeffs[1:]):
            raise ValueError("Unexpected detail coefficient type. Check input image and wavelet parameters.")

        compressed_coeffs = [
            tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
            for c_tuple in coeffs[1:]
        ]
        compressed_data = bz2.compress(pickle.dumps((coeffs[0], compressed_coeffs)))
        return compressed_data
    except Exception as e:
        print(f"Error during compression: {e}")
        return None

def decompress_image(compressed_data, wavelet='haar', quantization_factor=100):
    """Decompress an image from compressed wavelet data."""
    try:
        data = pickle.loads(bz2.decompress(compressed_data))
        decompressed_coeffs = [
            tuple((arr * quantization_factor).astype(np.float32) for arr in c_tuple)
            for c_tuple in data[1]
        ]
        coeffs = [data[0]] + decompressed_coeffs
        return pywt.waverec2(coeffs, wavelet=wavelet, axes=(0, 1)).clip(0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error during decompression: {e}")
        return None

def preprocess_frame(frame, target_width=None, target_height=None):
    """Resize and validate frame."""
    try:
        if target_width and target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return frame
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def compress_video(input_video, output_dir, wavelet='haar', level=3, quantization_factor=100):
    """Compress a video frame by frame."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {input_video}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        def process_frame(frame_idx):
            local_cap = cv2.VideoCapture(input_video)
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = local_cap.read()
            local_cap.release()
            if not ret:
                return None

            frame = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_width, frame_height)
            compressed_data = compress_image(frame, wavelet, level, quantization_factor)
            if compressed_data is None:
                return None

            compressed_filename = f"{output_dir}/frame_{frame_idx}.bz2"
            with open(compressed_filename, 'wb') as f:
                f.write(compressed_data)
            return compressed_filename

        with ThreadPoolExecutor() as executor:
            compressed_files = list(tqdm(executor.map(process_frame, range(frame_count)), total=frame_count, desc="Compressing video"))

        return [f for f in compressed_files if f is not None]
    except Exception as e:
        print(f"Error compressing video: {e}")
        return None

def decompress_video(compressed_dir, output_video, wavelet='haar', quantization_factor=100, fps=30):
    """Decompress frames and reassemble them into a video."""
    try:
        frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.bz2')])
        if not frame_files:
            raise Exception("No compressed frames found in directory.")

        with open(f"{compressed_dir}/{frame_files[0]}", 'rb') as f:
            compressed_data = f.read()
            first_frame = decompress_image(compressed_data, wavelet, quantization_factor)
            if first_frame is None:
                raise Exception("Could not decompress first frame to get shape.")
            frame_size = (first_frame.shape[1], first_frame.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        for frame_file in tqdm(frame_files, desc="Decompressing video"):
            with open(f"{compressed_dir}/{frame_file}", 'rb') as f:
                compressed_data = f.read()
                frame = decompress_image(compressed_data, wavelet, quantization_factor)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                else:
                    print(f"Error decoding frame {frame_file}")

        out.release()
        print(f"Decompressed video saved to {output_video}.")
    except Exception as e:
        print(f"Error decompressing video: {e}")

# Example usage
input_video_path = "sample_video.mp4"  # Replace with your video path
output_compressed_dir = "compressed_frames"
output_video_path = "output.mp4"

compressed_files = compress_video(input_video_path, output_compressed_dir)
if compressed_files:
    decompress_video(output_compressed_dir, output_video_path)

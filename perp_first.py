import os
import cv2
import numpy as np
import pywt
import pickle
import bz2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

def compress_image(image, wavelet='haar', level=3, quantization_factor=50):
    """Compress an image using wavelet transform."""
    try:
        coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
        compressed_coeffs = [
            tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
            for c_tuple in coeffs[1:]
        ]
        compressed_data = bz2.compress(pickle.dumps((coeffs[0], compressed_coeffs)))
        return compressed_data
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

def decompress_image(compressed_data, wavelet='haar', quantization_factor=50):
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
        print(f"Error decompressing image: {e}")
        return None

def compress_video(input_video, output_dir, wavelet='haar', level=3, quantization_factor=50):
    """Compress a video frame by frame with improved efficiency."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {input_video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use a single VideoCapture instance for all frames
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        def process_frame(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            compressed_data = compress_image(frame, wavelet, level, quantization_factor)
            if compressed_data is None:
                return None
            return compressed_data

        with ThreadPoolExecutor() as executor:
            compressed_data_list = list(tqdm(executor.map(process_frame, frames), total=len(frames), desc="Compressing video"))

        # Save compressed frames
        for idx, compressed_data in enumerate(compressed_data_list):
            if compressed_data is not None:
                compressed_filename = f"{output_dir}/frame_{idx}.bz2"
                with open(compressed_filename, 'wb') as f:
                    f.write(compressed_data)
        
        return [f"{output_dir}/frame_{idx}.bz2" for idx, data in enumerate(compressed_data_list) if data is not None]
    except Exception as e:
        print(f"Error compressing video: {e}")
        return None

def decompress_video(compressed_dir, output_video, wavelet='haar', quantization_factor=50, fps=30):
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
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
                    continue

        out.release()
        print(f"Decompressed video saved to {output_video}.")
    except Exception as e:
        print(f"Error decompressing video: {e}")

def evaluate_compression(original_frame, decompressed_frame, compressed_size):
    """Evaluate compression performance for a single frame."""
    try:
        psnr_value = psnr(original_frame, decompressed_frame, data_range=255)
        compression_ratio = original_frame.nbytes / compressed_size
        return psnr_value, compression_ratio
    except Exception as e:
        print(f"Error evaluating compression: {e}")
        return None, None

# Example usage
input_video_path = "sample_video.mp4"  # Replace with your video path
output_compressed_dir = "compressed_frames"
output_video_path = "output.mp4"

compressed_files = compress_video(input_video_path, output_compressed_dir)
if compressed_files:
    decompress_video(output_compressed_dir, output_video_path)

    cap = cv2.VideoCapture(input_video_path)
    ret, original_frame = cap.read()
    cap.release()
    if ret:
        with open(compressed_files[0], 'rb') as f:
            compressed_data = f.read()
        decompressed_frame = decompress_image(compressed_data)

        if decompressed_frame is not None:
            psnr_val, cr = evaluate_compression(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), decompressed_frame, len(compressed_data))
            if psnr_val and cr:
                print(f"PSNR: {psnr_val:.2f} dB, Compression Ratio: {cr:.2f}")
else:
    print("Video compression failed. Cannot proceed with decompression or evaluation.")

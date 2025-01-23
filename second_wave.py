import cv2
import numpy as np
import pywt
import pickle
import bz2
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

# Constants
WAVELET = 'haar'
LEVEL = 3
QUANTIZATION_FACTOR = 50  # Adjust based on quality vs. compression needs

def compress_image(image, wavelet=WAVELET, level=LEVEL, quantization_factor=QUANTIZATION_FACTOR):
    """Compress an image using wavelet transform and quantization."""
    try:
        # Apply wavelet transform
        coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
        
        # Quantize coefficients (except the approximation coefficients)
        quantized_coeffs = [
            tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
            for c_tuple in coeffs[1:]
        ]
        
        # Compress using bz2
        compressed_data = bz2.compress(pickle.dumps((coeffs[0], quantized_coeffs)))
        return compressed_data
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

def decompress_image(compressed_data, wavelet=WAVELET, quantization_factor=QUANTIZATION_FACTOR):
    """Decompress an image from compressed wavelet data."""
    try:
        # Decompress using bz2
        data = pickle.loads(bz2.decompress(compressed_data))
        
        # Dequantize coefficients
        dequantized_coeffs = [
            tuple((arr * quantization_factor).astype(np.float32) for arr in c_tuple)
            for c_tuple in data[1]
        ]
        
        # Reconstruct image using inverse wavelet transform
        coeffs = [data[0]] + dequantized_coeffs
        reconstructed_image = pywt.waverec2(coeffs, wavelet=wavelet, axes=(0, 1))
        return np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error decompressing image: {e}")
        return None

def compress_video(input_video, output_dir, wavelet=WAVELET, level=LEVEL, quantization_factor=QUANTIZATION_FACTOR):
    """Compress a video frame by frame with motion estimation."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {input_video}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        def process_frame(frame_idx):
            """Process a single frame for compression."""
            local_cap = cv2.VideoCapture(input_video)
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = local_cap.read()
            local_cap.release()
            if not ret:
                return None
            
            # Convert to RGB and compress
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            compressed_data = compress_image(frame, wavelet, level, quantization_factor)
            if compressed_data is None:
                return None
            
            # Save compressed frame
            compressed_filename = f"{output_dir}/frame_{frame_idx}.bz2"
            with open(compressed_filename, 'wb') as f:
                f.write(compressed_data)
            return compressed_filename

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            compressed_files = list(tqdm(executor.map(process_frame, range(frame_count)), total=frame_count, desc="Compressing video"))

        return [f for f in compressed_files if f is not None], fps, (width, height)
    except Exception as e:
        print(f"Error compressing video: {e}")
        return None, None, None

def decompress_video(compressed_dir, output_video, fps, frame_size, wavelet=WAVELET, quantization_factor=QUANTIZATION_FACTOR):
    """Decompress frames and reassemble them into a video."""
    try:
        frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.bz2')])
        if not frame_files:
            raise Exception("No compressed frames found in directory.")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        # Decompress and write frames
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

# Compress video
compressed_files, fps, frame_size = compress_video(input_video_path, output_compressed_dir)
if compressed_files:
    # Decompress video
    decompress_video(output_compressed_dir, output_video_path, fps, frame_size)

    # Evaluate compression
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
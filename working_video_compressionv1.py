import os
import cv2
import numpy as np
import pywt
import pickle
import bz2  # For better compression
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def wavelet_transform(image, wavelet='haar', level=3):
    coeffs_per_channel = []
    for channel in cv2.split(image):
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        coeffs_per_channel.append(coeffs)
    return coeffs_per_channel

def quantize_coefficients(coeffs, quantization_factor=50):
    quantized_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):
            quantized_coeffs.append(tuple(np.round(subband / quantization_factor).astype(np.int16) for subband in c))
        else:
            quantized_coeffs.append(np.round(c / quantization_factor).astype(np.int16))
    return quantized_coeffs

def dequantize_coefficients(quantized_coeffs, quantization_factor=50):
    dequantized_coeffs = []
    for c in quantized_coeffs:
        if isinstance(c, tuple):
            dequantized_coeffs.append(tuple(subband * quantization_factor for subband in c))
        else:
            dequantized_coeffs.append(c * quantization_factor)
    return dequantized_coeffs

def compress_image(image, wavelet='haar', level=3, quantization_factor=50):
    coeffs_per_channel = wavelet_transform(image, wavelet, level)
    quantized_coeffs = [quantize_coefficients(coeffs, quantization_factor) for coeffs in coeffs_per_channel]
    compressed_data = {'quantized_coeffs': quantized_coeffs, 'wavelet': wavelet, 'quantization_factor': quantization_factor}
    return bz2.compress(pickle.dumps(compressed_data))

def decompress_image(compressed_data):
    data = pickle.loads(bz2.decompress(compressed_data))
    quantized_coeffs = data['quantized_coeffs']
    wavelet = data['wavelet']
    quantization_factor = data['quantization_factor']
    dequantized_coeffs = [dequantize_coefficients(coeffs, quantization_factor) for coeffs in quantized_coeffs]
    reconstructed_channels = [pywt.waverec2(coeffs, wavelet) for coeffs in dequantized_coeffs]
    reconstructed_image = cv2.merge([np.clip(channel, 0, 255).astype(np.uint8) for channel in reconstructed_channels])
    return reconstructed_image

def compress_video(input_video, output_dir, wavelet='haar', level=3, quantization_factor=50):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {input_video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    def process_frame(frame_idx):
        local_cap = cv2.VideoCapture(input_video)
        local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = local_cap.read()
        local_cap.release()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

def decompress_video(compressed_dir, output_video, wavelet='haar', quantization_factor=50, fps=30):
    frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.bz2')])
    if not frame_files:
        raise Exception("No compressed frames found in directory.")

    with open(f"{compressed_dir}/{frame_files[0]}", 'rb') as f:
        compressed_data = f.read()
        first_frame = decompress_image(compressed_data)
        if first_frame is None:
            raise Exception("Could not decompress first frame to get shape.")
        frame_size = (first_frame.shape[1], first_frame.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for frame_file in tqdm(frame_files, desc="Decompressing video"):
        with open(f"{compressed_dir}/{frame_file}", 'rb') as f:
            compressed_data = f.read()
            frame = decompress_image(compressed_data)
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            else:
                print(f"Error decoding frame {frame_file}")
                continue

    out.release()
    print(f"Decompressed video saved to {output_video}.")

def evaluate_compression(original_frame, decompressed_frame, compressed_size):
    psnr_value = psnr(original_frame, decompressed_frame, data_range=255)
    compression_ratio = original_frame.nbytes / compressed_size
    return psnr_value, compression_ratio

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
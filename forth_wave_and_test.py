import cv2
import numpy as np
import pywt
import pickle
import heapq
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import itertools

# Constants (default values)
WAVELET = 'haar'
LEVEL = 3
BASE_QUANTIZATION_FACTOR = 50
MOTION_THRESHOLD = 5

# Huffman Coding Functions
def build_huffman_tree(freq):
    """Build a Huffman tree from a frequency dictionary."""
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0][1:]

def build_huffman_codes(tree):
    """Build Huffman codes from a Huffman tree."""
    codes = {}
    for symbol, code in tree:
        codes[symbol] = code
    return codes

def huffman_encode(data, codes):
    """Encode data using Huffman codes."""
    return ''.join(codes[symbol] for symbol in data)

def huffman_decode(encoded_data, tree):
    """Decode Huffman-encoded data."""
    decoded_data = []
    current_node = tree
    for bit in encoded_data:
        if bit == '0':
            current_node = current_node[0]
        else:
            current_node = current_node[1]
        if isinstance(current_node, str):
            decoded_data.append(current_node)
            current_node = tree
    return decoded_data

# Motion Estimation
def motion_estimation(prev_frame, current_frame):
    """Estimate motion vectors between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, None, None)[0]
    return flow

def motion_compensation(prev_frame, flow):
    """Compensate for motion using flow vectors."""
    h, w = prev_frame.shape[:2]
    flow_map = -flow
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
    compensated_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_LINEAR)
    return compensated_frame

# Adaptive Quantization
def adaptive_quantization(coeffs, base_factor):
    """Apply adaptive quantization to wavelet coefficients."""
    quantized_coeffs = []
    for i, c in enumerate(coeffs):
        if isinstance(c, tuple):
            # Higher quantization for higher-frequency subbands
            factor = base_factor * (2 ** i)
            quantized_coeffs.append(tuple((arr / factor).astype(np.int16) for arr in c))
        else:
            quantized_coeffs.append((c / base_factor).astype(np.int16))
    return quantized_coeffs

def adaptive_dequantization(quantized_coeffs, base_factor):
    """Reverse adaptive quantization."""
    dequantized_coeffs = []
    for i, c in enumerate(quantized_coeffs):
        if isinstance(c, tuple):
            factor = base_factor * (2 ** i)
            dequantized_coeffs.append(tuple((arr * factor).astype(np.float32) for arr in c))
        else:
            dequantized_coeffs.append((c * base_factor).astype(np.float32))
    return dequantized_coeffs

# Compression and Decompression
def compress_image(image, prev_frame=None, wavelet=WAVELET, level=LEVEL, base_factor=BASE_QUANTIZATION_FACTOR):
    """Compress an image using wavelet transform, motion compensation, and adaptive quantization."""
    try:
        if prev_frame is not None:
            # Motion estimation and compensation
            flow = motion_estimation(prev_frame, image)
            compensated_frame = motion_compensation(prev_frame, flow)
            residual = image.astype(np.int16) - compensated_frame.astype(np.int16)
        else:
            residual = image.astype(np.int16)

        # Apply wavelet transform
        coeffs = pywt.wavedec2(residual, wavelet=wavelet, level=level, axes=(0, 1))

        # Adaptive quantization
        quantized_coeffs = adaptive_quantization(coeffs, base_factor)

        # Huffman encoding
        flat_coeffs = [item for sublist in quantized_coeffs for item in (sublist if isinstance(sublist, tuple) else [sublist])]
        freq = defaultdict(int)
        for coeff in flat_coeffs:
            freq[coeff] += 1
        huffman_tree = build_huffman_tree(freq)
        huffman_codes = build_huffman_codes(huffman_tree)
        encoded_data = huffman_encode(flat_coeffs, huffman_codes)

        # Pack data for storage
        compressed_data = pickle.dumps((huffman_tree, encoded_data, flow if prev_frame is not None else None))
        return compressed_data
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

def decompress_image(compressed_data, prev_frame=None, wavelet=WAVELET, base_factor=BASE_QUANTIZATION_FACTOR):
    """Decompress an image from compressed data."""
    try:
        # Unpack data
        huffman_tree, encoded_data, flow = pickle.loads(compressed_data)

        # Huffman decoding
        flat_coeffs = huffman_decode(encoded_data, huffman_tree)
        quantized_coeffs = []
        idx = 0
        for i in range(LEVEL + 1):
            if i == 0:
                quantized_coeffs.append(np.array(flat_coeffs[idx], dtype=np.int16))
                idx += 1
            else:
                quantized_coeffs.append(tuple(
                    np.array(flat_coeffs[idx + j], dtype=np.int16) for j in range(3)
                ))
                idx += 3

        # Adaptive dequantization
        dequantized_coeffs = adaptive_dequantization(quantized_coeffs, base_factor)

        # Inverse wavelet transform
        residual = pywt.waverec2(dequantized_coeffs, wavelet=wavelet, axes=(0, 1))

        # Motion compensation
        if prev_frame is not None and flow is not None:
            compensated_frame = motion_compensation(prev_frame, flow)
            reconstructed_image = (compensated_frame + residual).clip(0, 255).astype(np.uint8)
        else:
            reconstructed_image = residual.clip(0, 255).astype(np.uint8)

        return reconstructed_image
    except Exception as e:
        print(f"Error decompressing image: {e}")
        return None

# Video Compression and Decompression
def compress_video(input_video, output_dir, wavelet=WAVELET, level=LEVEL, base_factor=BASE_QUANTIZATION_FACTOR):
    """Compress a video using motion compensation, wavelet transform, and Huffman coding."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {input_video}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        prev_frame = None
        compressed_files = []

        for frame_idx in tqdm(range(frame_count), desc="Compressing video"):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            compressed_data = compress_image(frame, prev_frame, wavelet, level, base_factor)
            if compressed_data is None:
                continue

            compressed_filename = f"{output_dir}/frame_{frame_idx}.pkl"
            with open(compressed_filename, 'wb') as f:
                f.write(compressed_data)
            compressed_files.append(compressed_filename)
            prev_frame = frame

        cap.release()
        return compressed_files, fps, (width, height)
    except Exception as e:
        print(f"Error compressing video: {e}")
        return None, None, None

def decompress_video(compressed_dir, output_video, fps, frame_size, wavelet=WAVELET, base_factor=BASE_QUANTIZATION_FACTOR):
    """Decompress a video and save it as a new file."""
    try:
        frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.pkl')])
        if not frame_files:
            raise Exception("No compressed frames found in directory.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        prev_frame = None
        for frame_file in tqdm(frame_files, desc="Decompressing video"):
            with open(f"{compressed_dir}/{frame_file}", 'rb') as f:
                compressed_data = f.read()
            frame = decompress_image(compressed_data, prev_frame, wavelet, base_factor)
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                prev_frame = frame
            else:
                print(f"Error decoding frame {frame_file}")

        out.release()
        print(f"Decompressed video saved to {output_video}.")
    except Exception as e:
        print(f"Error decompressing video: {e}")

# Automated Testing and Analysis
def test_video_compression(input_video_path, output_compressed_dir, output_video_path, parameters):
    """Test video compression with different parameters and analyze results."""
    results = []
    for params in parameters:
        print(f"Testing parameters: {params}")
        BASE_QUANTIZATION_FACTOR, MOTION_THRESHOLD, WAVELET, LEVEL = params

        # Compress video
        compressed_files, fps, frame_size = compress_video(input_video_path, output_compressed_dir, WAVELET, LEVEL, BASE_QUANTIZATION_FACTOR)
        if compressed_files:
            # Decompress video
            decompress_video(output_compressed_dir, output_video_path, fps, frame_size, WAVELET, BASE_QUANTIZATION_FACTOR)

            # Evaluate metrics
            original_size = os.path.getsize(input_video_path)
            compressed_size = sum(os.path.getsize(f) for f in compressed_files)
            compression_ratio = original_size / compressed_size

            cap = cv2.VideoCapture(input_video_path)
            ret, original_frame = cap.read()
            cap.release()
            if ret:
                with open(compressed_files[0], 'rb') as f:
                    compressed_data = f.read()
                decompressed_frame = decompress_image(compressed_data, None, WAVELET, BASE_QUANTIZATION_FACTOR)
                if decompressed_frame is not None:
                    psnr_val = psnr(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), decompressed_frame, data_range=255)
                    results.append((params, compression_ratio, psnr_val))

    # Print results
    for params, cr, psnr_val in results:
        print(f"Parameters: {params}, Compression Ratio: {cr:.2f}, PSNR: {psnr_val:.2f} dB")

# Example usage
input_video_path = "sample_video.mp4"
output_compressed_dir = "compressed_frames"
output_video_path = "output.mp4"

# Define parameter grid
parameters = list(itertools.product(
    [20, 50, 100],  # BASE_QUANTIZATION_FACTOR
    [2, 5, 10],     # MOTION_THRESHOLD
    ['haar', 'db2', 'sym2'],  # WAVELET
    [2, 3, 4]       # LEVEL
))

# Run tests
test_video_compression(input_video_path, output_compressed_dir, output_video_path, parameters)
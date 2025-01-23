""""
New Approach: Block-Based Motion Compensation + DCT

    Motion Estimation and Compensation:

        Divide each frame into small blocks (e.g., 16x16 pixels).

        For each block in the current frame, find the most similar block in the previous frame (motion estimation).

        Store only the motion vectors and the residual (difference between the current block and the matched block).

    Discrete Cosine Transform (DCT):

        Apply DCT to the residual blocks to transform them into frequency domains.

        Quantize the DCT coefficients to reduce precision and discard less important information.

    Entropy Coding:

        Use Huffman coding or Arithmetic coding to further compress the quantized DCT coefficients and motion vectors.

    Frame Types:

        I-frames: Keyframes compressed independently (no motion compensation).

        P-frames: Predicted frames that rely on previous frames for compression.
"""
import cv2
import numpy as np
import pickle
import zlib
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor

# Constants
BLOCK_SIZE = 16  # Size of blocks for motion estimation
SEARCH_RANGE = 8  # Reduced search range for motion estimation
QUANTIZATION_SCALE = 10  # Quantization scale for DCT coefficients
I_FRAME_INTERVAL = 10  # Interval for I-frames (keyframes)
DOWNSCALE_FACTOR = 2  # Downsample frames for faster motion estimation

def motion_estimation(prev_frame, current_frame):
    """Estimate motion vectors between two frames using block matching."""
    height, width = current_frame.shape[:2]
    motion_vectors = np.zeros((height // BLOCK_SIZE, width // BLOCK_SIZE, 2), dtype=np.int16)

    # Downsample frames for faster motion estimation
    prev_frame_small = cv2.resize(prev_frame, (width // DOWNSCALE_FACTOR, height // DOWNSCALE_FACTOR))
    current_frame_small = cv2.resize(current_frame, (width // DOWNSCALE_FACTOR, height // DOWNSCALE_FACTOR))

    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            min_diff = float('inf')
            best_vector = (0, 0)

            # Search for the best match in the previous frame
            for dy in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
                for dx in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
                    y_prev = (y // DOWNSCALE_FACTOR) + dy
                    x_prev = (x // DOWNSCALE_FACTOR) + dx

                    if 0 <= y_prev <= (height // DOWNSCALE_FACTOR) - (BLOCK_SIZE // DOWNSCALE_FACTOR) and 0 <= x_prev <= (width // DOWNSCALE_FACTOR) - (BLOCK_SIZE // DOWNSCALE_FACTOR):
                        prev_block = prev_frame_small[y_prev:y_prev + (BLOCK_SIZE // DOWNSCALE_FACTOR), x_prev:x_prev + (BLOCK_SIZE // DOWNSCALE_FACTOR)]
                        current_block = current_frame_small[(y // DOWNSCALE_FACTOR):(y // DOWNSCALE_FACTOR) + (BLOCK_SIZE // DOWNSCALE_FACTOR), (x // DOWNSCALE_FACTOR):(x // DOWNSCALE_FACTOR) + (BLOCK_SIZE // DOWNSCALE_FACTOR)]
                        diff = np.sum(np.abs(prev_block - current_block))

                        if diff < min_diff:
                            min_diff = diff
                            best_vector = (dy * DOWNSCALE_FACTOR, dx * DOWNSCALE_FACTOR)

            motion_vectors[y // BLOCK_SIZE, x // BLOCK_SIZE] = best_vector

    return motion_vectors

def motion_compensation(prev_frame, motion_vectors):
    """Compensate for motion using motion vectors."""
    height, width = prev_frame.shape[:2]
    compensated_frame = np.zeros_like(prev_frame)

    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            dy, dx = motion_vectors[y // BLOCK_SIZE, x // BLOCK_SIZE]
            y_prev = y + dy
            x_prev = x + dx

            if 0 <= y_prev <= height - BLOCK_SIZE and 0 <= x_prev <= width - BLOCK_SIZE:
                compensated_frame[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = prev_frame[y_prev:y_prev + BLOCK_SIZE, x_prev:x_prev + BLOCK_SIZE]

    return compensated_frame

def dct_transform(block):
    """Apply DCT to a block."""
    return cv2.dct(block.astype(np.float32))

def inverse_dct_transform(block):
    """Apply inverse DCT to a block."""
    return cv2.idct(block)

def quantize(block, scale):
    """Quantize DCT coefficients."""
    return np.round(block / scale).astype(np.int16)

def dequantize(block, scale):
    """Dequantize DCT coefficients."""
    return block * scale

def compress_frame(frame, prev_frame, is_keyframe=False):
    """Compress a frame using motion compensation and DCT."""
    if is_keyframe:
        # Compress as an I-frame (no motion compensation)
        compressed_blocks = []
        for y in range(0, frame.shape[0], BLOCK_SIZE):
            for x in range(0, frame.shape[1], BLOCK_SIZE):
                block = frame[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
                dct_block = dct_transform(block)
                quantized_block = quantize(dct_block, QUANTIZATION_SCALE)
                compressed_blocks.append(quantized_block)
        return {'type': 'I', 'blocks': compressed_blocks}
    else:
        # Compress as a P-frame (motion compensation)
        motion_vectors = motion_estimation(prev_frame, frame)
        compensated_frame = motion_compensation(prev_frame, motion_vectors)
        residual = frame - compensated_frame

        compressed_blocks = []
        for y in range(0, frame.shape[0], BLOCK_SIZE):
            for x in range(0, frame.shape[1], BLOCK_SIZE):
                block = residual[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
                dct_block = dct_transform(block)
                quantized_block = quantize(dct_block, QUANTIZATION_SCALE)
                compressed_blocks.append(quantized_block)

        return {'type': 'P', 'motion_vectors': motion_vectors, 'blocks': compressed_blocks}

def decompress_frame(compressed_frame, prev_frame):
    """Decompress a frame."""
    if compressed_frame['type'] == 'I':
        # Decompress an I-frame
        frame = np.zeros_like(prev_frame)
        idx = 0
        for y in range(0, frame.shape[0], BLOCK_SIZE):
            for x in range(0, frame.shape[1], BLOCK_SIZE):
                quantized_block = compressed_frame['blocks'][idx]
                dequantized_block = dequantize(quantized_block, QUANTIZATION_SCALE)
                block = inverse_dct_transform(dequantized_block)
                frame[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = block
                idx += 1
        return frame
    else:
        # Decompress a P-frame
        motion_vectors = compressed_frame['motion_vectors']
        compensated_frame = motion_compensation(prev_frame, motion_vectors)

        residual = np.zeros_like(prev_frame)
        idx = 0
        for y in range(0, residual.shape[0], BLOCK_SIZE):
            for x in range(0, residual.shape[1], BLOCK_SIZE):
                quantized_block = compressed_frame['blocks'][idx]
                dequantized_block = dequantize(quantized_block, QUANTIZATION_SCALE)
                block = inverse_dct_transform(dequantized_block)
                residual[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = block
                idx += 1

        return compensated_frame + residual

def compress_video(input_video, output_file):
    """Compress a video."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {input_video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    compressed_frames = []
    prev_frame = None

    for frame_idx in tqdm(range(frame_count), desc="Compressing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_keyframe = (frame_idx % I_FRAME_INTERVAL == 0)

        compressed_frame = compress_frame(frame, prev_frame, is_keyframe)
        compressed_frames.append(compressed_frame)
        prev_frame = frame

    cap.release()

    # Save compressed data
    with open(output_file, 'wb') as f:
        pickle.dump({'fps': fps, 'width': width, 'height': height, 'frames': compressed_frames}, f)

def decompress_video(input_file, output_video):
    """Decompress a video."""
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    fps = data['fps']
    width = data['width']
    height = data['height']
    compressed_frames = data['frames']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)

    prev_frame = None

    for compressed_frame in tqdm(compressed_frames, desc="Decompressing video"):
        frame = decompress_frame(compressed_frame, prev_frame)
        out.write(frame.clip(0, 255).astype(np.uint8))
        prev_frame = frame

    out.release()

# Example usage
input_video_path = "sample_video.mp4"
compressed_file = "compressed_video.pkl"
output_video_path = "output.mp4"

# Compress video
compress_video(input_video_path, compressed_file)

# Decompress video
decompress_video(compressed_file, output_video_path)

print("Video compression and decompression complete.")
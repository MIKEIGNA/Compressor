
import cv2
import numpy as np
import pywt
import pickle
import zlib

def wavelet_transform(image, wavelet='haar', level=2):
    """
    Apply wavelet transform to each channel of a color image.

    Args:
        image: Input image as a NumPy array (H, W, C).
        wavelet: Type of wavelet to use.
        level: Number of decomposition levels.

    Returns:
        coeffs_per_channel: List of wavelet coefficients for each channel.
    """
    coeffs_per_channel = []
    for channel in cv2.split(image):  # Split image into color channels
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        coeffs_per_channel.append(coeffs)
    return coeffs_per_channel

def quantize_coefficients(coeffs, quantization_factor=10):
    """
    Quantize wavelet coefficients to reduce storage size.

    Args:
        coeffs: Wavelet coefficients for a single channel.
        quantization_factor: Factor to scale and reduce precision.

    Returns:
        Quantized coefficients.
    """
    quantized_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):
            quantized_coeffs.append(tuple(np.round(subband / quantization_factor).astype(np.int16) for subband in c))
        else:
            quantized_coeffs.append(np.round(c / quantization_factor).astype(np.int16))
    return quantized_coeffs

def dequantize_coefficients(quantized_coeffs, quantization_factor=10):
    """
    Dequantize wavelet coefficients to reconstruct the original values.

    Args:
        quantized_coeffs: Quantized coefficients for a single channel.
        quantization_factor: Factor used for quantization.

    Returns:
        Dequantized coefficients.
    """
    dequantized_coeffs = []
    for c in quantized_coeffs:
        if isinstance(c, tuple):
            dequantized_coeffs.append(tuple(subband * quantization_factor for subband in c))
        else:
            dequantized_coeffs.append(c * quantization_factor)
    return dequantized_coeffs

def compress_image(image, wavelet='haar', level=2, quantization_factor=10):
    """
    Compress a color image using wavelet transform and quantization.

    Args:
        image: Input image as a NumPy array (H, W, C).
        wavelet: Type of wavelet to use.
        level: Number of decomposition levels.
        quantization_factor: Factor for quantizing wavelet coefficients.

    Returns:
        compressed_data: Compressed coefficients and metadata.
    """
    coeffs_per_channel = wavelet_transform(image, wavelet, level)
    quantized_coeffs = [quantize_coefficients(coeffs, quantization_factor) for coeffs in coeffs_per_channel]
    compressed_data = {'quantized_coeffs': quantized_coeffs, 'wavelet': wavelet, 'quantization_factor': quantization_factor}
    return zlib.compress(pickle.dumps(compressed_data))  # Compress the data further with zlib

def decompress_image(compressed_data):
    """
    Decompress a color image using wavelet transform and dequantization.

    Args:
        compressed_data: Compressed coefficients and metadata.

    Returns:
        image: Reconstructed color image as a NumPy array.
    """
    data = pickle.loads(zlib.decompress(compressed_data))
    quantized_coeffs = data['quantized_coeffs']
    wavelet = data['wavelet']
    quantization_factor = data['quantization_factor']
    dequantized_coeffs = [dequantize_coefficients(coeffs, quantization_factor) for coeffs in quantized_coeffs]
    reconstructed_channels = [pywt.waverec2(coeffs, wavelet) for coeffs in dequantized_coeffs]
    reconstructed_image = cv2.merge([np.clip(channel, 0, 255).astype(np.uint8) for channel in reconstructed_channels])
    return reconstructed_image

# Example usage
image = cv2.imread('sample.png')  # Load the color image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

compressed_data = compress_image(image, wavelet='haar', level=3, quantization_factor=20)

# Save and Load Compressed Data
with open('compressed_image.pkl', 'wb') as f:
    f.write(compressed_data)

with open('compressed_image.pkl', 'rb') as f:
    loaded_data = f.read()

# Decompression
decompressed_image = decompress_image(loaded_data)

# Save and compare
decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
cv2.imwrite('decompressed_image.png', decompressed_image)
print("Compression complete.")

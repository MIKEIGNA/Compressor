# import os
# import cv2
# import numpy as np
# import pywt
# import pickle
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from concurrent.futures import ThreadPoolExecutor

# # def compress_image(image, wavelet='haar', level=2, quantization_factor=20):
# #     """
# #     Compress an image using wavelet transform.
# #     """
# #     coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
    
# #     # Compress each detail coefficient tuple (cH, cV, cD)
# #     compressed_coeffs = [
# #         tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
# #         for c_tuple in coeffs[1:]
# #     ]
    
# #     # Combine the approximation coefficients with compressed detail coefficients
# #     return pickle.dumps((coeffs[0], compressed_coeffs))

# def decompress_image(compressed_data, wavelet='haar', quantization_factor=20):
#     """
#     Decompress an image from wavelet compressed data.
#     """
#     data = pickle.loads(compressed_data)
    
#     # Decompress each detail coefficient tuple (cH, cV, cD)
#     decompressed_coeffs = [
#         tuple((arr * quantization_factor).astype(np.float32) for arr in c_tuple)
#         for c_tuple in data[1]
#     ]
    
#     # Combine the approximation coefficients with decompressed detail coefficients
#     coeffs = [data[0]] + decompressed_coeffs
#     return pywt.waverec2(coeffs, wavelet=wavelet, axes=(0, 1)).clip(0, 255).astype(np.uint8)

# # def compress_video(input_video_path, output_dir, wavelet='haar', level=2, quantization_factor=20):
# #     """
# #     Compress a video frame by frame using wavelet compression.
# #     """
# #     os.makedirs(output_dir, exist_ok=True)
# #     cap = cv2.VideoCapture(input_video_path)
# #     frame_idx = 0

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         compressed_data = compress_image(frame, wavelet, level, quantization_factor)

# #         with open(os.path.join(output_dir, f"frame_{frame_idx}.pkl"), 'wb') as f:
# #             f.write(compressed_data)
        
# #         frame_idx += 1

# #     cap.release()
# #     print(f"Compressed {frame_idx} frames and saved to {output_dir}.")
# def compress_image(image, wavelet='haar', level=2, quantization_factor=20):
#     coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
#     compressed_coeffs = [
#         tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
#         for c_tuple in coeffs[1:]
#     ]
#     return pickle.dumps((coeffs[0], compressed_coeffs))

# def compress_video(input_video, output_dir, wavelet='haar', level=2, quantization_factor=20):
#     cap = cv2.VideoCapture(input_video)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     compressed_files = []

#     def process_frame(frame_idx):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         if not ret:
#             return None
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         compressed_data = compress_image(frame, wavelet, level, quantization_factor)
#         compressed_filename = f"{output_dir}/frame_{frame_idx}.pkl"
#         with open(compressed_filename, 'wb') as f:
#             f.write(compressed_data)
#         return compressed_filename

#     with ThreadPoolExecutor() as executor:
#         compressed_files = list(executor.map(process_frame, range(frame_count)))

#     cap.release()
#     return [f for f in compressed_files if f is not None]

# # def decompress_video(output_dir, output_video_path, frame_shape, fps, wavelet='haar', quantization_factor=20):
# #     """
# #     Decompress frames and reassemble them into a video.
# #     """
# #     frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])
# #     height, width, channels = frame_shape
# #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
# #     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# #     for frame_file in frame_files:
# #         with open(os.path.join(output_dir, frame_file), 'rb') as f:
# #             compressed_data = f.read()

# #         frame = decompress_image(compressed_data, wavelet, quantization_factor)
# #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# #         out.write(frame)

# #     out.release()
# #     print(f"Decompressed video saved to {output_video_path}.")
# def decompress_video(compressed_dir, output_video, wavelet='haar', quantization_factor=20):
#     import os
#     from tqdm import tqdm

#     frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.pkl')])
#     frame_size = None
#     out = None

#     for frame_file in tqdm(frame_files, desc="Decompressing video"):
#         with open(f"{compressed_dir}/{frame_file}", 'rb') as f:
#             compressed_data = f.read()
#         frame = decompress_image(compressed_data, wavelet, quantization_factor)

#         if out is None:
#             frame_size = (frame.shape[1], frame.shape[0])
#             out = cv2.VideoWriter(
#                 output_video,
#                 cv2.VideoWriter_fourcc(*'mp4v'),
#                 30,  # Assuming 30 FPS
#                 frame_size,
#             )

#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         out.write(frame_bgr)

#     out.release()

# def evaluate_compression(original_frame, decompressed_frame, compressed_size):
#     """
#     Evaluate compression performance for a single frame.
#     """
#     psnr_value = psnr(original_frame, decompressed_frame, data_range=255)
#     compression_ratio = original_frame.nbytes / compressed_size
#     return psnr_value, compression_ratio


# import os
# os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"  # Force single thread for video writing
# import cv2
# import numpy as np
# import pywt
# import pickle
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# def compress_image(image, wavelet='haar', level=2, quantization_factor=20):
#     """Compress an image using wavelet transform."""
#     try:
#         coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
#         compressed_coeffs = [
#             tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
#             for c_tuple in coeffs[1:]
#         ]
#         return pickle.dumps((coeffs[0], compressed_coeffs))
#     except Exception as e:
#         print(f"Error compressing image: {e}")
#         return None

# def compress_video(input_video, output_dir, wavelet='haar', level=2, quantization_factor=20):
#     """Compress a video frame by frame using wavelet compression and multithreading (thread-safe)."""
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         cap = cv2.VideoCapture(input_video)
#         if not cap.isOpened():
#             raise Exception(f"Could not open video: {input_video}")
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS) # Get FPS from the video
#         cap.release() #Release the main capture object

#         def process_frame(frame_idx):
#             local_cap = cv2.VideoCapture(input_video) #Create capture object in the thread
#             local_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_idx / fps) * 1000) # Seek using milliseconds for accuracy
#             ret, frame = local_cap.read()
#             local_cap.release()
#             if not ret:
#                 return None
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             compressed_data = compress_image(frame, wavelet, level, quantization_factor)
#             if compressed_data is None:
#                 return None
#             compressed_filename = f"{output_dir}/frame_{frame_idx}.pkl"
#             with open(compressed_filename, 'wb') as f:
#                 f.write(compressed_data)
#             return compressed_filename

#         with ThreadPoolExecutor() as executor:
#             compressed_files = list(tqdm(executor.map(process_frame, range(frame_count)), total=frame_count, desc="Compressing video"))

#         return [f for f in compressed_files if f is not None]
#     except Exception as e:
#         print(f"Error compressing video: {e}")
#         return None
    
# def decompress_image(compressed_data, wavelet='haar', quantization_factor=20):
#     """Decompress an image from wavelet compressed data."""
#     try:
#         data = pickle.loads(compressed_data)
#         decompressed_coeffs = [
#             tuple((arr * quantization_factor).astype(np.float32) for arr in c_tuple)
#             for c_tuple in data[1]
#         ]
#         coeffs = [data[0]] + decompressed_coeffs
#         return pywt.waverec2(coeffs, wavelet=wavelet, axes=(0, 1)).clip(0, 255).astype(np.uint8)
#     except Exception as e:
#         print(f"Error decompressing image: {e}")
#         return None

# def decompress_video(compressed_dir, output_video, wavelet='haar', quantization_factor=20):
#     """Decompress frames and reassemble them into a video."""
#     try:
#         frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.pkl')])
#         if not frame_files:
#             raise Exception("No compressed frames found in directory.")

#         with open(f"{compressed_dir}/{frame_files[0]}", 'rb') as f:
#             compressed_data = f.read()
#             first_frame = decompress_image(compressed_data, wavelet, quantization_factor)
#             if first_frame is None:
#                 raise Exception("Could not decompress first frame to get shape.")
#             frame_size = (first_frame.shape[1], first_frame.shape[0])

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
#         fps = 30.0 #set fps
#         out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

#         for frame_file in tqdm(frame_files, desc="Decompressing video"):
#             with open(f"{compressed_dir}/{frame_file}", 'rb') as f:
#                 compressed_data = f.read()
#                 frame = decompress_image(compressed_data, wavelet, quantization_factor)
#                 if frame is not None:
#                     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                     out.write(frame_bgr)
#                 else:
#                     print(f"Error decoding frame {frame_file}")
#                     continue

#         out.release()
#         print(f"Decompressed video saved to {output_video}.")
#     except Exception as e:
#         print(f"Error decompressing video: {e}")

# def evaluate_compression(original_frame, decompressed_frame, compressed_size):
#     """Evaluate compression performance for a single frame."""
#     try:
#         psnr_value = psnr(original_frame, decompressed_frame, data_range=255)
#         compression_ratio = original_frame.nbytes / compressed_size
#         return psnr_value, compression_ratio
#     except Exception as e:
#         print(f"Error evaluating compression: {e}")
#         return None, None

# # Example usage:
# input_video_path = "sample_video.mp4"  # Replace with your video path
# output_compressed_dir = "compressed_frames"
# output_video_path = "output.mp4"

# compressed_files = compress_video(input_video_path, output_compressed_dir)
# if compressed_files:
#     decompress_video(output_compressed_dir, output_video_path)

#     # Example evaluation (using the first frame)
#     cap = cv2.VideoCapture(input_video_path)
#     ret, original_frame = cap.read()
#     cap.release()
#     if ret:
#         with open(compressed_files[0], 'rb') as f:
#             compressed_data = f.read()
#         decompressed_frame = decompress_image(compressed_data)

#         if decompressed_frame is not None:
#             psnr_val, cr = evaluate_compression(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), decompressed_frame, len(compressed_data))
#             if psnr_val and cr:
#                 print(f"PSNR: {psnr_val:.2f} dB, Compression Ratio: {cr:.2f}")
# else:
#     print("Video compression failed. Cannot proceed with decompression or evaluation.")



#GPT

# import os
# os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"  # Force single thread for video writing
# import cv2
# import numpy as np
# import pywt
# import pickle
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from tqdm import tqdm


# def compress_image(image, wavelet='haar', level=2, quantization_factor=50):
#     """Compress an image using wavelet transform."""
#     coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
#     # Apply quantization
#     compressed_coeffs = [
#         tuple((arr / quantization_factor).astype(np.int8) for arr in c_tuple)
#         for c_tuple in coeffs[1:]
#     ]
#     # Compress approximation coefficients
#     compressed_approx = (coeffs[0] / quantization_factor).astype(np.int8)
#     return pickle.dumps((compressed_approx, compressed_coeffs))


# def compress_video(input_video, output_dir, wavelet='haar', level=2, quantization_factor=50):
#     """Compress a video frame by frame using wavelet compression."""
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(input_video)
#     if not cap.isOpened():
#         raise Exception(f"Could not open video: {input_video}")

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     compressed_files = []
#     prev_frame = None

#     for frame_idx in tqdm(range(frame_count), desc="Compressing video"):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Skip compression if frame is similar to the previous one
#         if prev_frame is not None and np.mean(np.abs(frame_rgb - prev_frame)) < 5:
#             compressed_files.append(None)  # Mark as duplicate
#             continue

#         # Compress current frame
#         compressed_data = compress_image(frame_rgb, wavelet, level, quantization_factor)
#         compressed_filename = f"{output_dir}/frame_{frame_idx}.pkl"
#         with open(compressed_filename, 'wb') as f:
#             f.write(compressed_data)
#         compressed_files.append(compressed_filename)
#         prev_frame = frame_rgb

#     cap.release()
#     return compressed_files


# def decompress_image(compressed_data, wavelet='haar', quantization_factor=50):
#     """Decompress an image from wavelet compressed data."""
#     compressed_approx, compressed_coeffs = pickle.loads(compressed_data)
#     decompressed_approx = (compressed_approx * quantization_factor).astype(np.float32)
#     decompressed_coeffs = [
#         tuple((arr * quantization_factor).astype(np.float32) for arr in c_tuple)
#         for c_tuple in compressed_coeffs
#     ]
#     coeffs = [decompressed_approx] + decompressed_coeffs
#     return pywt.waverec2(coeffs, wavelet=wavelet, axes=(0, 1)).clip(0, 255).astype(np.uint8)


# def decompress_video(compressed_dir, output_video, wavelet='haar', quantization_factor=50):
#     """Decompress frames and reassemble them into a video."""
#     frame_files = sorted([f for f in os.listdir(compressed_dir) if f.endswith('.pkl')])
#     if not frame_files:
#         raise Exception("No compressed frames found in directory.")

#     # Read the first frame for size
#     with open(f"{compressed_dir}/{frame_files[0]}", 'rb') as f:
#         first_frame = decompress_image(f.read(), wavelet, quantization_factor)
#     frame_size = (first_frame.shape[1], first_frame.shape[0])
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = 30.0
#     out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

#     for frame_file in tqdm(frame_files, desc="Decompressing video"):
#         if frame_file is None:  # Skip duplicate frames
#             out.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
#             continue
#         with open(f"{compressed_dir}/{frame_file}", 'rb') as f:
#             compressed_data = f.read()
#             frame = decompress_image(compressed_data, wavelet, quantization_factor)
#             if frame is not None:
#                 first_frame = frame
#                 out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     out.release()


# # Example usage
# input_video_path = "sample_video.mp4"  # Replace with your video path
# output_compressed_dir = "compressed_frames"
# output_video_path = "output.mp4"

# compressed_files = compress_video(input_video_path, output_compressed_dir)
# if compressed_files:
#     decompress_video(output_compressed_dir, output_video_path)


# GEMINI

# import os
# os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"  # Force single thread for video writing
# import cv2
# import numpy as np
# import pywt
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# def compress_image(image, wavelet='haar', level=2, quantization_factor=20):
#     """Compress an image using wavelet transform."""
#     try:
#         coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0, 1))
#         compressed_coeffs = [
#             tuple((arr / quantization_factor).astype(np.int16) for arr in c_tuple)
#             for c_tuple in coeffs[1:]
#         ]
#         filename = f"{image.shape[0]}_{image.shape[1]}.npy"  # Use shape for unique filename
#         with open(filename, 'wb') as f:
#             np.save(f, (coeffs[0], compressed_coeffs))  # Use numpy.save
#         return filename
#     except Exception as e:
#         print(f"Error compressing image: {e}")
#         return None

# def compress_video(input_video, output_dir, wavelet='haar', level=2, quantization_factor=20):
#     """Compress a video frame by frame using wavelet compression and multithreading."""
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         cap = cv2.VideoCapture(input_video)
#         if not cap.isOpened():
#             raise Exception(f"Could not open video: {input_video}")
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS) # Get FPS from the video
#         cap.release() #Release the main capture object

#         def process_frame(frame_idx):
#             local_cap = cv2.VideoCapture(input_video) #Create capture object in the thread
#             local_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_idx / fps) * 1000) # Seek using milliseconds for accuracy
#             ret, frame = local_cap.read()
#             local_cap.release()
#             if not ret:
#                 return None
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             compressed_filename = compress_image(frame, wavelet, level, quantization_factor)
#             if compressed_filename is None:
#                 return None
#             return compressed_filename

#         with ThreadPoolExecutor() as executor:
#             compressed_files = list(tqdm(executor.map(process_frame, range(frame_count)), total=frame_count, desc="Compressing video"))
#         return [os.path.join(output_dir, f) for f in compressed_files if f is not None] #join path here
#     except Exception as e:
#         print(f"Error compressing video: {e}")
#         return None

# def decompress_image(compressed_filename, wavelet='haar', quantization_factor=20): #take filename as input
#     """Decompress an image from wavelet compressed data."""
#     try:
#         with open(compressed_filename, 'rb') as f:
#             data = np.load(f, allow_pickle=True) #Use numpy.load
#         decompressed_coeffs = [
#             tuple((arr * quantization_factor).astype(np.float32) for arr in c_tuple)
#             for c_tuple in data[1]
#         ]
#         coeffs = [data[0]] + decompressed_coeffs
#         return pywt.waverec2(coeffs, wavelet=wavelet, axes=(0, 1)).clip(0, 255).astype(np.uint8)
#     except Exception as e:
#         print(f"Error decompressing image: {e}")
#         return None

# def decompress_video(compressed_files, output_video, wavelet='haar', quantization_factor=20): #take list of files as input
#     """Decompress frames and reassemble them into a video."""
#     try:
#         if not compressed_files:
#             raise Exception("No compressed frames provided.")
#         first_frame = decompress_image(compressed_files[0], wavelet, quantization_factor)
#         if first_frame is None:
#             raise Exception("Could not decompress first frame to get shape.")
#         frame_size = (first_frame.shape[1], first_frame.shape[0])

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
#         fps = 30.0 #set fps
#         out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
        
#         for frame_file in tqdm(compressed_files, desc="Decompressing video"):
#             frame = decompress_image(frame_file, wavelet, quantization_factor)
#             if frame is not None:
#                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 out.write(frame_bgr)
#             else:
#                 print(f"Error decoding frame {frame_file}")
#                 continue
#         out.release()
#         print(f"Decompressed video saved to {output_video}.")
#     except Exception as e:
#         print(f"Error decompressing video: {e}")

# def evaluate_compression(original_frame, decompressed_frame, compressed_size):
#     """Evaluate compression performance for a single frame."""
#     try:
#         psnr_value = psnr(original_frame, decompressed_frame, data_range=255)
#         compression_ratio = original_frame.nbytes / compressed_size
#         return psnr_value, compression_ratio
#     except Exception as e:
#         print(f"Error evaluating compression: {e}")
#         return None, None

# # Example usage:
# input_video_path = "sample_video.mp4"  # Replace with your video path
# output_compressed_dir = "compressed_frames"
# output_video_path = "output.mp4"

# compressed_files = compress_video(input_video_path, output_compressed_dir)
# if compressed_files:
#     decompress_video(compressed_files, output_video_path)
#     # Example evaluation (using the first frame)
#     cap = cv2.VideoCapture(input_video_path)
#     ret, original_frame = cap.read()
#     cap.release()
#     if ret:
#         decompressed_frame = decompress_image(compressed_files[0])
#         if decompressed_frame is not None:
#             with open(compressed_files[0], 'rb') as f:
#                 compressed_size = os.path.getsize(f.name)
#             psnr_val, cr = evaluate_compression(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), decompressed_frame, compressed_size)
#             if psnr_val and cr:
#                 print(f"PSNR: {psnr_val:.2f} dB, Compression Ratio: {cr:.2f}")
# else:
#     print("Video compression failed. Cannot proceed with decompression or evaluation.")



# CHATGPT

import os
import cv2
import numpy as np
import pywt
import pickle
import bz2  # For better compression
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compress_image(image, wavelet='haar', level=3, quantization_factor=100):
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
        print(f"Error decompressing image: {e}")
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

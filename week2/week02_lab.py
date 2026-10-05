# Week 02 Lab: Image Processing Through a Four-Step Pipeline
# Complete Python version of the notebook
import numpy as np


def rgb_to_grayscale(r: int, g: int, b: int) -> int:
    """Convert one RGB pixel to a rounded grayscale luminosity value."""
    luminosity = 0.299 * r + 0.587 * g + 0.114 * b
    return int(round(luminosity))

sample_result = rgb_to_grayscale(180, 120, 60)
print("Sample result:", sample_result)
assert sample_result == 131

# Extra test: equal channels should remain unchanged.
assert rgb_to_grayscale(80, 80, 80) == 80
print("Sample 1 passed.")

import numpy as np

def adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Apply O = alpha * I + beta and clip the result to [0, 255]."""
    image_float = image.astype(np.float32)
    adjusted = alpha * image_float + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)

sample_image = np.array([[100, 150], [200, 50]], dtype=np.uint8)
sample_output = adjust_brightness_contrast(sample_image, alpha=1.5, beta=-20)
expected_output = np.array([[130, 205], [255, 55]], dtype=np.uint8)
np.testing.assert_array_equal(sample_output, expected_output)
print(sample_output)

# Extra test: a large positive beta must clip at 255.
extra = adjust_brightness_contrast(np.array([[240]], dtype=np.uint8), alpha=1.0, beta=30)
assert extra[0, 0] == 255
print("Sample 2 passed.")

# Problem 1: Thresholding
def threshold_image(image: np.ndarray, threshold: int) -> np.ndarray:
    """Convert a grayscale image to black and white."""
    return np.where(image >= threshold, 255, 0).astype(np.uint8)

problem_1_input = np.array([[20, 128, 200], [100, 150, 250]], dtype=np.uint8)
problem_1_expected = np.array([[0, 255, 255], [0, 255, 255]], dtype=np.uint8)
np.testing.assert_array_equal(threshold_image(problem_1_input, 128), problem_1_expected)
print("Problem 1 passed")

# Problem 2: Image Memory
def display_memory_bytes(width: int, height: int, bpp: int) -> int:
    """Return image memory in bytes."""
    return width * height * bpp // 8

expected_bytes = 1920 * 1080 * 24 // 8
assert display_memory_bytes(1920, 1080, 24) == expected_bytes
print("Problem 2 passed")

# Problem 3: Average 9 Pixels
def mean_filter_3x3(neighborhood: np.ndarray) -> int:
    """Return the rounded average of 9 pixels."""
    return int(round(np.sum(neighborhood) / 9))

problem_3_input = np.array([[10, 20, 10], [30, 50, 30], [10, 20, 10]], dtype=np.uint8)
assert mean_filter_3x3(problem_3_input) == 21
print("Problem 3 passed")

# Problem 4: Change Image Contrast
def contrast_stretch(image: np.ndarray, in_min: float, in_max: float, out_min: float = 0.0, out_max: float = 255.0) -> np.ndarray:
    """Change image values from one range to another."""
    result = (image - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    return np.clip(result, out_min, out_max)

problem_4_input = np.array([50, 100, 150], dtype=np.float32)
expected = np.array([0.0, 127.5, 255.0], dtype=np.float32)
np.testing.assert_allclose(contrast_stretch(problem_4_input, 50, 150), expected)
print("Problem 4 passed")

# Problem 5: Find an Edge with Sobel
def sobel_response(block: np.ndarray) -> tuple[float, float, float]:
    """Return gx, gy, and edge strength."""
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = float(np.sum(block * gx_kernel))
    gy = float(np.sum(block * gy_kernel))
    strength = float(np.sqrt(gx * gx + gy * gy))
    return gx, gy, strength

problem_5_input = np.array([[20, 20, 200], [20, 20, 200], [20, 20, 200]], dtype=np.float32)
gx, gy, strength = sobel_response(problem_5_input)
assert gx == 720.0 and gy == 0.0
print("Problem 5 passed")

import math
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = plt.imread("sample.jpg")

# Remove alpha channel if image has RGBA
if image.shape[2] == 4:
    image = image[:, :, :3]

# Make sure image is uint8
image = image.astype(np.uint8)


# ---------------- Task 4 ----------------
N = 8

downsampled = image[::N, ::N, :]

expanded = np.repeat(np.repeat(downsampled, N, axis=0), N, axis=1)
expanded = expanded[:image.shape[0], :image.shape[1], :]

dimension_reduction = (1 - downsampled.shape[0] / image.shape[0]) * 100
memory_saving = (1 - downsampled.nbytes / image.nbytes) * 100

print("Original:", image.shape, image.nbytes)
print("Downsampled:", downsampled.shape, downsampled.nbytes)
print("Re-expanded:", expanded.shape)
print("Dimension reduction:", round(dimension_reduction, 2), "%")
print("Memory saving:", round(memory_saving, 2), "%")
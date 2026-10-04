import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Task 2 ----------------
img = np.zeros((300, 400, 3), dtype=np.uint8)

img[:150, :200] = [255, 0, 0]
img[:150, 200:] = [0, 255, 0]
img[150:, :200] = [0, 0, 255]
img[150:, 200:] = [255, 255, 255]

print("Shape:", img.shape)
print("Data type:", img.dtype)
print("Total elements:", img.size)
print("Memory:", img.nbytes, "bytes")

plt.imshow(img)
plt.axis("off")
plt.show()
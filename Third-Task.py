import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Task 3 ----------------
# Replace sample.jpg with the required input image if needed.
image = plt.imread("sample.jpg")
if image.shape[2] == 4:
    image = image[:, :, :3]

image = image.astype(np.uint8)

red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

red_only = np.zeros_like(image)
green_only = np.zeros_like(image)
blue_only = np.zeros_like(image)

red_only[:, :, 0] = red
green_only[:, :, 1] = green
blue_only[:, :, 2] = blue

fig, axes = plt.subplots(2, 3)
axes[0,0].imshow(red_only); axes[0,0].set_title("Red-Only")
axes[0,1].imshow(green_only); axes[0,1].set_title("Green-Only")
axes[0,2].imshow(blue_only); axes[0,2].set_title("Blue-Only")
axes[1,0].imshow(red, cmap="gray"); axes[1,0].set_title("Red Intensity")
axes[1,1].imshow(green, cmap="gray"); axes[1,1].set_title("Green Intensity")
axes[1,2].imshow(blue, cmap="gray"); axes[1,2].set_title("Blue Intensity")

for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()


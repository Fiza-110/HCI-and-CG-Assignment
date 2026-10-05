import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width, channels = image.shape

    pixel_count = width * height
    estimated_bytes = pixel_count * channels

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": list(image.shape),
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": "BGR"
    }


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # OpenCV loads images as BGR
    blue, green, red = cv2.split(image)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Half width and half height
    height, width = image.shape[:2]

    new_width = width // 2
    new_height = height // 2

    downsampled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create channel images
    red_rgb = np.zeros_like(image_rgb)
    green_rgb = np.zeros_like(image_rgb)
    blue_rgb = np.zeros_like(image_rgb)

    red_rgb[:, :, 0] = red
    green_rgb[:, :, 1] = green
    blue_rgb[:, :, 2] = blue

    downsampled_rgb = cv2.cvtColor(
        downsampled,
        cv2.COLOR_BGR2RGB
    )

    # Create figure
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(red_rgb)
    plt.title("Red Channel")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(green_rgb)
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(blue_rgb)
    plt.title("Blue Channel")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(downsampled_rgb)
    plt.title("Half-Size Image")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "pixel_views.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "original_size": [width, height],
        "downsampled_size": [new_width, new_height],
        "output_path": output_path
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""

    os.makedirs(output_dir, exist_ok=True)

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness
    brighter = cv2.add(
        gray,
        np.full(gray.shape, brightness_delta, dtype=np.uint8)
    )

    # Contrast
    contrast = gray.astype(np.float32) * contrast_factor
    contrast = np.clip(contrast, 0, 255).astype(np.uint8)

    # Threshold
    _, thresholded = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # Create figure
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(brighter, cmap="gray")
    plt.title(f"Brighter (+{brightness_delta})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(contrast, cmap="gray")
    plt.title(f"Higher Contrast (×{contrast_factor})")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(thresholded, cmap="gray")
    plt.title(f"Threshold ({threshold})")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "adjustments.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""

    os.makedirs(output_dir, exist_ok=True)

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mean blur
    blurred = cv2.blur(
        gray,
        (kernel_size, kernel_size)
    )

    # Sobel edges from original grayscale
    sobel_x_original = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y_original = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel_original = cv2.magnitude(
        sobel_x_original,
        sobel_y_original
    )

    sobel_original = cv2.convertScaleAbs(
        sobel_original
    )

    # Sobel edges from blurred grayscale
    sobel_x_blurred = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y_blurred = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel_blurred = cv2.magnitude(
        sobel_x_blurred,
        sobel_y_blurred
    )

    sobel_blurred = cv2.convertScaleAbs(
        sobel_blurred
    )

    # Create figure
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(blurred, cmap="gray")
    plt.title(f"Mean Blur ({kernel_size}×{kernel_size})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(sobel_original, cmap="gray")
    plt.title("Sobel Edges - Original")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(sobel_blurred, cmap="gray")
    plt.title("Sobel Edges - Blurred")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "blur_and_edges.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }


def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1–4 and return their results together."""

    task1 = inspect_image(image_path)

    task2 = create_pixel_views(
        image_path,
        output_dir
    )

    task3 = create_adjustments(
        image_path,
        output_dir
    )

    task4 = create_blur_and_edges(
        image_path,
        output_dir
    )

    return {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4
    }


def main() -> None:
    """Run the lab using the required repository paths."""

    image_path = "images/original.jpg"
    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)

    results = run_lab(
        image_path,
        output_dir
    )

    print("Image Processing Lab completed successfully.")
    print()
    print("Image information:")
    print(results["task1"])
    print()
    print("Created:")
    print(results["task2"]["output_path"])
    print(results["task3"]["output_path"])
    print(results["task4"]["output_path"])


if __name__ == "__main__":
    main()
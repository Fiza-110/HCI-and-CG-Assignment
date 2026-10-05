# Weeks 1–3 Image Processing Lab

## Overview

This project is a Weeks 1–3 Image Processing Lab for the HCI and Computer Graphics assignment.

The program uses Python, OpenCV, NumPy, and Matplotlib to inspect an image and perform basic image processing operations.

The lab covers:

1. Image data inspection
2. RGB channels, grayscale, and downsampling
3. Brightness, contrast, and thresholding
4. Mean blur and Sobel edge detection

The input image is:

```text
images/original.jpg
```

The generated results are saved in:

```text
outputs/
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Requirements

The required Python packages are:

```text
opencv-python
numpy
matplotlib
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
HCI-and-CG-Assignment/
│
├── weeks01-03_image_lab/
│   ├── images/
│   │   └── original.jpg
│   │
│   ├── outputs/
│   │   ├── pixel_views.png
│   │   ├── adjustments.png
│   │   └── blur_and_edges.png
│   │
│   ├── weeks01-03_image_lab.py
│   └── requirements.txt
│
├── README.md
└── ...
```

---

# How to Run

Open PowerShell in the `weeks01-03_image_lab` folder.

Run:

```powershell
python weeks01-03_image_lab.py
```

The program will:

1. Load `images/original.jpg`
2. Inspect the image data
3. Create RGB channel views
4. Create a grayscale image
5. Create a half-size image
6. Apply brightness adjustment
7. Apply contrast adjustment
8. Apply thresholding
9. Apply mean blur
10. Detect edges using Sobel
11. Save the results in the `outputs` folder

---

# Task 1 – Image Data Inspection

The function used for Task 1 is:

```python
inspect_image(image_path: str) -> dict
```

This function loads the image using OpenCV and measures its basic properties.

## Image Measurements

The current `original.jpg` image has the following measurements:

| Property | Value |
|---|---:|
| Width | 640 pixels |
| Height | 480 pixels |
| Channels | 3 |
| NumPy Shape | `[480, 640, 3]` |
| Pixel Count | 307,200 |
| Estimated Bytes | 921,600 bytes |
| Color Order | BGR |

### Pixel Count

The total number of pixels is:

```text
Width × Height
= 640 × 480
= 307,200 pixels
```

### Estimated Memory

The estimated image size is:

```text
Width × Height × Channels
= 640 × 480 × 3
= 921,600 bytes
```

This estimate assumes 8 bits, or 1 byte, per color channel.

### OpenCV Color Order

OpenCV loads color images in **BGR** order:

```text
Blue → Green → Red
```

The code converts the image from BGR to RGB before displaying it with Matplotlib.

---

# Task 2 – Pixel Views

The function used for Task 2 is:

```python
create_pixel_views(image_path: str, output_dir: str) -> dict
```

This task creates six views:

1. Original image
2. Red channel
3. Green channel
4. Blue channel
5. Grayscale image
6. Half-size image

The output is saved as:

```text
outputs/pixel_views.png
```

## RGB Channels

The original image contains three color channels:

- Red
- Green
- Blue

The code separates the channels using:

```python
blue, green, red = cv2.split(image)
```

Because OpenCV uses BGR order, the channels are first obtained as blue, green, and red.

The separate channel views show the intensity contribution of each color.

## Grayscale

The image is converted to grayscale using:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

The grayscale image represents the intensity of the image without separate color information.

## Downsampling

The program creates a half-size image using:

```python
new_width = width // 2
new_height = height // 2
```

For the current image:

```text
Original size:
640 × 480 pixels

Half-size:
320 × 240 pixels
```

The program uses `cv2.INTER_AREA` interpolation for resizing.

### Dimensions

| Image | Width | Height |
|---|---:|---:|
| Original | 640 | 480 |
| Half-size | 320 | 240 |

### Observation

The half-size image contains fewer pixels than the original image. Because the resolution is reduced, some fine details can be lost.

The RGB channel views show the contribution of red, green, and blue information separately.

---

# Task 3 – Brightness, Contrast, and Threshold

The function used for Task 3 is:

```python
create_adjustments(
    image_path,
    output_dir,
    brightness_delta=40,
    contrast_factor=1.5,
    threshold=127
)
```

The image is first converted to grayscale.

The task creates:

1. Original grayscale
2. Brighter image
3. Higher contrast image
4. Thresholded image

The output is saved as:

```text
outputs/adjustments.png
```

## Brightness

The brightness adjustment is:

```text
+40
```

The code increases the grayscale pixel values by 40.

This makes the image appear brighter while keeping the values within the valid image range.

The relevant operation is:

```python
brighter = cv2.add(
    gray,
    np.full(gray.shape, brightness_delta, dtype=np.uint8)
)
```

## Contrast

The contrast factor is:

```text
1.5
```

The grayscale pixel values are multiplied by 1.5.

The result is clipped to the valid range:

```text
0 – 255
```

This increases the difference between darker and brighter areas.

## Threshold

The threshold value is:

```text
127
```

Binary thresholding is applied using:

```python
cv2.threshold(
    gray,
    threshold,
    255,
    cv2.THRESH_BINARY
)
```

Pixels above the threshold become white (`255`), while pixels below the threshold become black (`0`).

### Observation

The brightness operation makes the grayscale image lighter.

The contrast operation makes the difference between dark and bright regions stronger.

The threshold operation changes the grayscale image into a black-and-white binary image.

---

# Task 4 – Blur and Edge Detection

The function used for Task 4 is:

```python
create_blur_and_edges(
    image_path,
    output_dir,
    kernel_size=5
)
```

This task creates:

1. Grayscale image
2. Mean blurred image
3. Sobel edges from original grayscale
4. Sobel edges from blurred grayscale

The output is saved as:

```text
outputs/blur_and_edges.png
```

## Mean Blur

The kernel size is:

```text
5 × 5
```

The code applies:

```python
blurred = cv2.blur(
    gray,
    (kernel_size, kernel_size)
)
```

Mean blur averages neighboring pixels.

This smooths the image and reduces small details.

## Sobel Edge Detection

The Sobel operator detects changes in image intensity.

The program calculates horizontal and vertical gradients.

For the original grayscale image:

```python
sobel_x_original = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)
```

```python
sobel_y_original = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)
```

The horizontal and vertical gradients are combined using `cv2.magnitude()`.

The same process is performed on the blurred grayscale image.

### Observation

The Sobel result from the original grayscale image shows intensity changes and edges.

After applying the 5×5 mean blur, small details are reduced. Therefore, the Sobel result from the blurred image is smoother and may contain fewer small edges.

---

# Output Files

After running the program, the following files are created.

## `pixel_views.png`

Location:

```text
outputs/pixel_views.png
```

Contains:

- Original image
- Red channel
- Green channel
- Blue channel
- Grayscale
- Half-size image

## `adjustments.png`

Location:

```text
outputs/adjustments.png
```

Contains:

- Original grayscale
- Brighter image
- Higher contrast image
- Threshold image

## `blur_and_edges.png`

Location:

```text
outputs/blur_and_edges.png
```

Contains:

- Grayscale
- Mean blur
- Sobel edges from original
- Sobel edges from blurred image

---

# Functions Used

The Python program contains the following functions.

## `inspect_image()`

Loads the image and returns:

- Width
- Height
- Channels
- NumPy shape
- Pixel count
- Estimated bytes
- Color order

## `create_pixel_views()`

Creates:

- Original image
- Red channel
- Green channel
- Blue channel
- Grayscale
- Half-size image

## `create_adjustments()`

Creates:

- Original grayscale
- Brighter image
- Higher contrast image
- Thresholded image

## `create_blur_and_edges()`

Creates:

- Grayscale
- Mean blur
- Sobel edges from original
- Sobel edges from blurred image

## `run_lab()`

Runs Tasks 1–4 and returns all results together.

## `main()`

Uses the required repository paths:

```python
image_path = "images/original.jpg"
output_dir = "outputs"
```

and runs the complete lab.

---

# Observations Summary

### Task 1 – Image Inspection

The image is 640 × 480 pixels with 3 color channels. It contains 307,200 pixels and uses BGR color ordering when loaded by OpenCV.

### Task 2 – Pixel Views

The RGB channel views show the individual contribution of each color. Grayscale removes color information and represents the image using intensity values. Reducing the image from 640 × 480 to 320 × 240 decreases the number of pixels and can remove fine details.

### Task 3 – Image Adjustments

Increasing brightness by 40 makes the grayscale image lighter. A contrast factor of 1.5 increases the difference between darker and brighter regions. A threshold of 127 produces a binary black-and-white image.

### Task 4 – Blur and Edges

The 5 × 5 mean blur smooths the image by averaging neighboring pixels. Sobel edge detection highlights areas where image intensity changes. Applying Sobel after blurring reduces some small details and produces a smoother edge result.

---

# Conclusion

This lab demonstrates basic digital image processing using Python, OpenCV, NumPy, and Matplotlib.

The project shows how an image can be represented using pixels and color channels and how different processing techniques change the image.

The main operations demonstrated are:

- Image inspection
- RGB channel separation
- Grayscale conversion
- Image resizing
- Brightness adjustment
- Contrast adjustment
- Thresholding
- Mean blurring
- Sobel edge detection

These operations provide a basic understanding of how digital images can be analyzed and processed using Python.

---

# Source

This implementation was created for the Weeks 1–3 Image Processing Lab assignment.

Libraries used:

- OpenCV
- NumPy
- Matplotlib

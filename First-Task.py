import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Task 1 ----------------
def display_metrics(w, h, diagonal):
    total_pixels = w * h
    g = math.gcd(w, h)
    aspect_ratio = f"{w//g}:{h//g}"
    diagonal_pixels = math.sqrt(w**2 + h**2)
    ppi = diagonal_pixels / diagonal

    if ppi < 100:
        category = "Low Density (Standard Monitor)"
    elif ppi <= 200:
        category = "Medium Density (HD Display)"
    else:
        category = "High Density (Retina / Mobile)"

    return total_pixels, aspect_ratio, ppi, category

# Desktop
print(display_metrics(1920, 1080, 24))

# Smartphone
print(display_metrics(1170, 2532, 6.1))

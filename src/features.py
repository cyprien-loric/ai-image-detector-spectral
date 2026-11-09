from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_images(path, sample_size=None):
    path = Path(path)
    files = list(path.glob("*.jpg"))
    if sample_size is not None:
        files = files[:sample_size]

    x = []
    for file in files:
        img = Image.open(file).convert("L")
        x.append(np.asarray(img))
    x = np.array(x)
    return x


def noise_residual(image, sigma=1.0):
    image = image.astype(float)
    smoothed = gaussian_filter(image, sigma=sigma)
    residual = image - smoothed
    return residual


def rapsd(image):
    f = np.fft.fftshift(np.fft.fft2(image))
    energy = np.abs(f) ** 2
    h, w = image.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    ring_sum = np.bincount(radius.ravel(), weights=energy.ravel())
    ring_count = np.bincount(radius.ravel())
    return ring_sum / ring_count

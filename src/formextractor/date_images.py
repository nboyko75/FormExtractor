"""Alternative OCR inputs for failed date crops; never modify the source image."""

import cv2
import numpy as np
from PIL import Image

from .field_images import field_image_variants as date_image_variants


def date_part_variants(image: Image.Image) -> list[tuple[str, list[np.ndarray]]]:
    """Split only crops with two separate, aligned slash-shaped ink components."""
    variants = []
    for name, pixels in date_image_variants(image):
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, labels, stats, centers = cv2.connectedComponentsWithStats(ink)
        height, width = gray.shape
        components, slashes = [], []
        for label, (x, _y, w, h, area) in enumerate(stats[1:], 1):
            if area < 8 or h < height * 0.12 or w < 2:
                continue
            if w / h < 0.14 and (x < width * 0.15 or x + w > width * 0.85):
                continue  # Residual vertical box rules.
            components.append(label)
            if h < height * 0.45 or not 0.18 <= w / h <= 0.7:
                continue
            ys, xs = np.nonzero(labels == label)
            if np.corrcoef(xs, ys)[0, 1] <= -0.90:
                slashes.append(label)
        if len(slashes) != 2:
            continue
        slashes.sort(key=lambda label: centers[label, 0])
        first, second = slashes
        a, b = stats[first], stats[second]
        if min(a[3], b[3]) < max(a[3], b[3]) * 0.65:
            continue
        if abs(centers[first, 1] - centers[second, 1]) > max(a[3], b[3]) * 0.3:
            continue
        groups = [[], [], []]
        for label in components:
            if label not in slashes:
                group = sum(centers[label, 0] > centers[slash, 0] for slash in slashes)
                groups[group].append(label)
        if not all(groups):
            continue
        parts = []
        for group in groups:
            selected = np.isin(labels, group)
            x, y, w, h = cv2.boundingRect(selected.astype(np.uint8))
            part = np.where(selected, gray, 255).astype(np.uint8)[y : y + h, x : x + w]
            part = cv2.resize(part, None, fx=48 / h, fy=48 / h, interpolation=cv2.INTER_CUBIC)
            part = cv2.copyMakeBorder(part, 8, 8, 12, 12, cv2.BORDER_CONSTANT, value=255)
            parts.append(cv2.cvtColor(part, cv2.COLOR_GRAY2BGR))
        variants.append((name, parts))
    return variants

"""Alternative OCR inputs for failed date crops; never modify the source image."""

import cv2
import numpy as np
from PIL import Image


def date_image_variants(image: Image.Image) -> list[tuple[str, np.ndarray]]:
    """Remove edge rules and provide padded grayscale and binary date readings."""
    gray = np.asarray(image.convert("L")).copy()
    height, width = gray.shape
    if min(height, width) < 8:
        return []
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if not np.any(ink) or np.all(ink):
        return []
    mask = np.zeros_like(gray)
    left, top, right, bottom = 0, 0, width, height
    thickness = max(3, round(min(height, width) * 0.06))
    # Long, nearly horizontal/vertical strokes at the crop edges are form rules.
    # Interior strokes (including slashes and handwritten ones) are not removed.
    lines = cv2.HoughLinesP(
        ink,
        1,
        np.pi / 180,
        threshold=max(10, round(min(height, width) * 0.4)),
        minLineLength=max(8, round(min(height, width) * 0.6)),
        maxLineGap=3,
    )
    for x1, y1, x2, y2 in [] if lines is None else lines[:, 0]:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        horizontal = dx >= width * 0.65 and dy <= dx * 0.12
        vertical = dy >= height * 0.65 and dx <= dy * 0.12
        edge = False
        if horizontal and max(y1, y2) < height * 0.3:
            top = max(top, min(y1, y2))
            edge = True
        elif horizontal and min(y1, y2) > height * 0.7:
            bottom = min(bottom, max(y1, y2) + 1)
            edge = True
        elif vertical and max(x1, x2) < width * 0.15:
            left = max(left, min(x1, x2))
            edge = True
        elif vertical and min(x1, x2) > width * 0.85:
            right = min(right, max(x1, x2) + 1)
            edge = True
        if edge:
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    cleaned = gray.copy()
    cleaned[mask != 0] = 255
    cleaned = cleaned[top:bottom, left:right]
    _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(255 - binary)
    if points is None:
        return []
    x, y, w, h = cv2.boundingRect(points)
    if w < 4 or h < 4:
        return []
    cleaned = cleaned[y : y + h, x : x + w]
    scale = min(4.0, 64 / h, 1600 / w)
    resized = cv2.resize(cleaned, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return [
        (
            name,
            cv2.cvtColor(
                cv2.copyMakeBorder(
                    value,
                    8,
                    8,
                    16,
                    16,
                    cv2.BORDER_CONSTANT,
                    value=255,
                ),
                cv2.COLOR_GRAY2BGR,
            ),
        )
        for name, value in (("borderless_gray", resized), ("borderless_binary", binary))
    ]


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

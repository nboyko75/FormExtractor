"""Border cleanup for direct recognition of selected form fields."""

import cv2
import numpy as np
from PIL import Image


def remove_area_borders(image: Image.Image) -> Image.Image:
    """Erase long cell rules on a copy, retaining the crop size and crossing strokes."""
    pixels = np.asarray(image.convert("RGB")).copy()
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if min(height, width) < 8:
        return Image.fromarray(pixels)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if not np.any(ink) or np.all(ink):
        return Image.fromarray(pixels)
    lines = cv2.HoughLinesP(
        ink, 1, np.pi / 180,
        threshold=max(12, round(min(height, width) * 0.35)),
        minLineLength=max(12, round(min(height, width) * 0.5)),
        maxLineGap=3,
    )
    candidates = [] if lines is None else lines[:, 0].tolist()
    # Dense horizontal rules can consume the corner pixels before probabilistic
    # Hough finds a vertical edge. Directional opening recovers straight sides.
    for axis, length in ((0, width), (1, height)):
        size = max(24, round(length * .55))
        kernel = np.ones((1, size) if axis == 0 else (size, 1), np.uint8)
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if axis == 0 and w >= size and h <= max(7, w * .12):
                candidates.append((x, y + h // 2, x + w - 1, y + h // 2))
            elif axis == 1 and h >= size and w <= max(7, h * .12):
                candidates.append((x + w // 2, y, x + w // 2, y + h - 1))
    horizontal, vertical = [], []
    for x1, y1, x2, y2 in candidates:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx >= max(24, width * 0.55) and dy <= dx * 0.12:
            horizontal.append((int(x1), int(y1), int(x2), int(y2)))
        elif dy >= max(24, height * 0.55) and dx <= dy * 0.12:
            vertical.append((int(x1), int(y1), int(x2), int(y2)))

    def intersects(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return (
            max(min(ax1, ax2), min(bx1, bx2)) <= min(max(ax1, ax2), max(bx1, bx2)) + 5
            and max(min(ay1, ay2), min(by1, by2)) <= min(max(ay1, ay2), max(by1, by2)) + 5
        )

    masks = [np.zeros_like(gray), np.zeros_like(gray)]
    thickness = max(3, min(7, round(min(height, width) * .06)))
    for mask, candidates, perpendicular, axis, size in (
        (masks[0], horizontal, vertical, 1, height),
        (masks[1], vertical, horizontal, 0, width),
    ):
        for line in candidates:
            position = (line[axis] + line[axis + 2]) / 2
            # Interior rules must join a grid; avoid removing long ones, slashes,
            # or dashes in the middle of an otherwise unboxed value.
            if size * 0.2 < position < size * 0.8 and not any(
                intersects(line, other) for other in perpendicular
            ):
                continue
            cv2.line(mask, line[:2], line[2:], 255, thickness)
    mask = cv2.bitwise_or(*masks)
    if not np.any(mask):
        return Image.fromarray(pixels)

    # Interpolate across thin rules where a stroke enters from both sides.
    # This retains descenders and slashes crossing a border without cropping
    # away ink outside the box. The surrounding image is left untouched.
    repaired = cv2.inpaint(pixels, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(repaired)


def text_line_variants(image: Image.Image) -> list[tuple[str, np.ndarray]]:
    """Retry wide, single-line fields only; reject blank/rule-only and multiline crops."""
    if image.width < image.height * 2:
        return []
    variants = field_image_variants(image)
    if not variants:
        return []
    binary = variants[1][1][:, :, 0]
    rows = np.flatnonzero(np.count_nonzero(binary < 128, axis=1) >= 2)
    if not len(rows):
        return []
    # Separate text lines have a substantial white gap. Keep small gaps around
    # accents and dots within a line; do not flatten a paragraph into one value.
    if np.any(np.diff(rows) > max(3, (rows[-1] - rows[0] + 1) * 0.2)):
        return []
    return variants


def field_image_variants(image: Image.Image) -> list[tuple[str, np.ndarray]]:
    """Remove edge rules and provide padded grayscale and binary field readings."""
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


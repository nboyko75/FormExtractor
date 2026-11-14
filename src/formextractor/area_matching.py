"""Locate image templates and return upright crops from the target page."""

import cv2
import numpy as np
from PIL import Image

from .errors import ExtractionError


def _nontext_mask(shape, text_regions) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    for polygon in text_regions:
        points = np.asarray(polygon, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
        ):
            raise ExtractionError("Invalid text detection polygon")
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 0)
    # Exclude antialiased pixels just outside the detected boundary as well.
    return cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))


def _features(gray, mask, count):
    clean = np.where(mask, gray, 0).astype(np.uint8)
    points, descriptors = cv2.SIFT_create(nfeatures=count).detectAndCompute(clean, mask)
    if descriptors is None or np.all(mask):
        return points, descriptors
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # SIFT descriptors sample a neighbourhood, not just the keypoint's centre.
    keep = [
        index
        for index, point in enumerate(points)
        if distance[round(point.pt[1]), round(point.pt[0])] > 6 * point.size
    ]
    return [points[index] for index in keep], descriptors[keep] if keep else None


def _resize_mask(mask, size):
    return (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) == 255).astype(np.uint8) * 255


def _match_scores(search, needle, search_mask, needle_mask):
    if np.all(search_mask) and np.all(needle_mask):
        return cv2.matchTemplate(search, needle, cv2.TM_CCOEFF_NORMED)
    # Normalized correlation over only pixels valid in BOTH images. Filling
    # text boxes with white would otherwise let their shapes influence matching.
    valid = (search_mask != 0).astype(np.float32)
    weight = (needle_mask != 0).astype(np.float32)
    target = search.astype(np.float32) * valid / 255
    source = needle.astype(np.float32) * weight / 255

    def correlate(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCORR)

    count = correlate(valid, weight)
    denominator = np.maximum(count, 1)
    target_sum = correlate(target, weight)
    source_sum = correlate(valid, source)
    target_variance = np.maximum(
        correlate(target * target, weight) - target_sum**2 / denominator, 0
    )
    source_variance = np.maximum(correlate(valid, source * source) - source_sum**2 / denominator, 0)
    covariance = correlate(target, source) - target_sum * source_sum / denominator
    scores = covariance / np.maximum(np.sqrt(target_variance * source_variance), 1e-8)
    reliable = (
        (count >= max(weight.sum() * 0.6, weight.size * 0.1))
        & (target_variance / denominator > (3 / 255) ** 2)
        & (source_variance / denominator > (3 / 255) ** 2)
    )
    return np.where(reliable, np.clip(scores, -1, 1), -1)


class AreaTemplate:
    def __init__(self, page: Image.Image, area, *, text_regions=()) -> None:
        self.area = area
        self.page_size = page.size
        page_mask = _nontext_mask((page.height, page.width), text_regions)
        self.has_text = not np.all(page_mask)
        x1, y1, x2, y2 = area.pixel_box(*page.size)
        self.mask = page_mask[y1:y2, x1:x2].copy()
        with page.crop(area.pixel_box(*page.size)) as crop:
            self.gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        self.page_thumbnail = cv2.resize(np.asarray(page), (128, 128), interpolation=cv2.INTER_AREA)
        self.points, self.descriptors = _features(self.gray, self.mask, 2000)


class PageMatcher:
    def __init__(self, image: Image.Image, *, text_regions=()) -> None:
        self.rgb = np.asarray(image)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self.mask = _nontext_mask(self.gray.shape, text_regions)
        self.has_text = not np.all(self.mask)
        self.features = None

    def crop(self, template: AreaTemplate) -> Image.Image:
        # Identical pages (including changed resolution and quarter turns) have
        # a known alignment, even when the selected area itself is blank.
        # With detected text, this shortcut could use letters as alignment cues.
        for turns in range(4) if not (self.has_text or template.has_text) else ():
            oriented = np.rot90(self.rgb, turns)
            thumbnail = cv2.resize(oriented, (128, 128), interpolation=cv2.INTER_AREA)
            if np.array_equal(thumbnail, template.page_thumbnail):
                h, w = oriented.shape[:2]
                x1, y1, x2, y2 = template.area.pixel_box(w, h)
                return Image.fromarray(oriented[y1:y2, x1:x2].copy())
        crop = self._feature_crop(template)
        if crop is not None:
            return crop
        crop = self._template_crop(template)
        if crop is not None:
            return crop
        raise ExtractionError("No reliable image match found")

    def _feature_crop(self, template: AreaTemplate) -> Image.Image | None:
        if template.descriptors is None or len(template.points) < 6:
            return None
        if self.features is None:
            self.features = _features(self.gray, self.mask, 6000)
        points, descriptors = self.features
        if descriptors is None or len(points) < 6:
            return None
        pairs = cv2.BFMatcher().knnMatch(template.descriptors, descriptors, k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        # A target feature must not vote more than once for a transformation.
        matches = list({match.trainIdx: match for match in matches}.values())
        if len(matches) < 6:
            return None
        source = np.float32([template.points[m.queryIdx].pt for m in matches])
        target = np.float32([points[m.trainIdx].pt for m in matches])
        transform, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if transform is None or inliers is None:
            return None
        accepted = inliers.ravel().astype(bool)
        if accepted.sum() < 6 or accepted.mean() < 0.6:
            return None
        h, w = template.gray.shape
        # Reject accidental matches concentrated on a single character or line.
        spread = np.ptp(source[accepted], axis=0)
        if spread[0] < w * 0.2 or spread[1] < h * 0.2:
            return None
        scale = float(np.linalg.norm(transform[:, 0]))
        if not 0.25 <= scale <= 4:
            return None
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        mapped = cv2.transform(corners[None], transform)[0]
        height, width = self.gray.shape
        if (mapped < -1).any() or (mapped[:, 0] > width).any() or (mapped[:, 1] > height).any():
            return None
        # Preserve the target scan's resolution while undoing its rotation.
        output_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        inverse = cv2.invertAffineTransform(transform) * scale
        upright = cv2.warpAffine(
            self.rgb, inverse, output_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return Image.fromarray(upright)

    def _template_crop(self, template: AreaTemplate) -> Image.Image | None:
        # Small templates sometimes lack enough features. Search all quarter
        # turns at the original and expected scan resolutions in that case.
        visible = template.gray[template.mask != 0]
        if visible.size < template.gray.size * 0.1 or visible.std() < 3:
            return None
        search_scale = min(1.0, 1000 / max(self.gray.shape))
        best = None
        for turns in range(4):
            oriented = np.rot90(np.where(self.mask, self.gray, 0).astype(np.uint8), turns)
            h, w = oriented.shape
            search = cv2.resize(
                oriented,
                (round(w * search_scale), round(h * search_scale)),
                interpolation=cv2.INTER_AREA,
            )
            search_mask = _resize_mask(
                np.rot90(self.mask, turns), (search.shape[1], search.shape[0])
            )
            expected = min(w / template.page_size[0], h / template.page_size[1])
            for scale in sorted({1.0, expected * 0.9, expected, expected * 1.1}):
                th, tw = template.gray.shape
                size = (round(tw * scale * search_scale), round(th * scale * search_scale))
                if min(size) < 4 or size[0] > search.shape[1] or size[1] > search.shape[0]:
                    continue
                needle = cv2.resize(
                    np.where(template.mask, template.gray, 0).astype(np.uint8),
                    size,
                    interpolation=cv2.INTER_AREA,
                )
                needle_mask = _resize_mask(template.mask, size)
                visible = needle[needle_mask != 0]
                if visible.size < needle.size * 0.1 or visible.std() < 3:
                    continue
                scores = _match_scores(search, needle, search_mask, needle_mask)
                _, score, _, location = cv2.minMaxLoc(scores)
                if score < 0.88 or (best is not None and score <= best[0]):
                    continue
                # Repeated identical fields are ambiguous; do not choose one
                # merely because it occurs earlier on the page.
                x, y = location
                scores[
                    max(0, y - size[1] // 2) : y + size[1] // 2 + 1,
                    max(0, x - size[0] // 2) : x + size[0] // 2 + 1,
                ] = -1
                if float(scores.max()) > score - 0.04:
                    continue
                best = (score, turns, x, y, size)
        if best is None:
            return None
        _, turns, x, y, size = best
        oriented = np.rot90(self.rgb, turns)
        left, top = round(x / search_scale), round(y / search_scale)
        right, bottom = round((x + size[0]) / search_scale), round((y + size[1]) / search_scale)
        return Image.fromarray(oriented[top:bottom, left:right].copy())

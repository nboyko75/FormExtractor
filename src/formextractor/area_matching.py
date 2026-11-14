"""Locate image templates and return upright crops from the target page."""

import cv2
import numpy as np
from PIL import Image

from .errors import ExtractionError


class AreaTemplate:
    def __init__(self, page: Image.Image, area) -> None:
        self.area = area
        self.page_size = page.size
        with page.crop(area.pixel_box(*page.size)) as crop:
            self.gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        self.page_thumbnail = cv2.resize(np.asarray(page), (128, 128), interpolation=cv2.INTER_AREA)
        self.points, self.descriptors = cv2.SIFT_create(nfeatures=2000).detectAndCompute(
            self.gray, None
        )


class PageMatcher:
    def __init__(self, image: Image.Image) -> None:
        self.rgb = np.asarray(image)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self.features = None

    def crop(self, template: AreaTemplate) -> Image.Image:
        # Identical pages (including changed resolution and quarter turns) have
        # a known alignment, even when the selected area itself is blank.
        for turns in range(4):
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
            self.features = cv2.SIFT_create(nfeatures=6000).detectAndCompute(self.gray, None)
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
        if template.gray.std() < 3:
            return None
        search_scale = min(1.0, 1000 / max(self.gray.shape))
        best = None
        for turns in range(4):
            oriented = np.rot90(self.gray, turns)
            h, w = oriented.shape
            search = cv2.resize(
                oriented,
                (round(w * search_scale), round(h * search_scale)),
                interpolation=cv2.INTER_AREA,
            )
            expected = min(w / template.page_size[0], h / template.page_size[1])
            for scale in sorted({1.0, expected * 0.9, expected, expected * 1.1}):
                th, tw = template.gray.shape
                size = (round(tw * scale * search_scale), round(th * scale * search_scale))
                if min(size) < 4 or size[0] > search.shape[1] or size[1] > search.shape[0]:
                    continue
                needle = cv2.resize(template.gray, size, interpolation=cv2.INTER_AREA)
                if needle.std() < 3:
                    continue
                scores = cv2.matchTemplate(search, needle, cv2.TM_CCOEFF_NORMED)
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

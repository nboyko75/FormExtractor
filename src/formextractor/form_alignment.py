"""Register photographed forms by their paper outline and internal ruling."""

import cv2
import numpy as np


class RuledPage:
    """A small, text-free line image with a conservative paper boundary seed."""

    def __init__(self, gray, mask):
        height, width = gray.shape
        factor = min(1.0, 1000 / max(height, width))
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        self.scale = np.diag([size[0] / width, size[1] / height, 1.0])
        self.quad = None
        if min(size) < 32:
            return
        reduced = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        _, paper = cv2.threshold(
            cv2.GaussianBlur(reduced, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        contour = max(contours, key=cv2.contourArea)
        quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if (
            len(quad) != 4
            or not cv2.isContourConvex(quad)
            or not 0.25 < cv2.contourArea(quad) / reduced.size < 0.98
        ):
            return
        quad = quad.reshape(4, 2).astype(np.float32)
        centered = quad - quad.mean(axis=0)
        quad = quad[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]

        # Keep the paper edge and background out of verification: two unrelated
        # sheets have the same outline. Only their internal form lines can agree.
        interior = np.zeros_like(reduced)
        cv2.fillConvexPoly(interior, quad.astype(np.int32), 255)
        margin = max(3, round(max(size) * 0.025)) | 1
        interior = cv2.erode(interior, np.ones((margin, margin), np.uint8))
        self.mask = (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) == 255) & (interior > 0)
        structure = np.zeros_like(reduced)
        lines = cv2.createLineSegmentDetector().detect(reduced)[0]
        if lines is None:
            return
        for line in lines[:, 0]:
            start, end = line[:2], line[2:]
            if np.linalg.norm(end - start) >= max(size) * 0.04:
                cv2.line(structure, tuple(start.astype(int)), tuple(end.astype(int)), 255, 2)
        structure[~self.mask] = 0
        y, x = np.nonzero(structure)
        if len(x) < max(size) * 4 or np.ptp(x) < size[0] * 0.5 or np.ptp(y) < size[1] * 0.5:
            return
        self.lines = structure
        self.quad = quad

    def align(self, target):
        """Return a verified source-to-target homography, or no alignment."""
        if self.quad is None or target.quad is None:
            return None
        candidates = []
        for turns in range(4):
            destination = np.roll(target.quad, turns, axis=0)
            transform = cv2.getPerspectiveTransform(self.quad, destination)
            for factor, sigma in ((0.25, 3), (0.5, 2), (1.0, 2)):
                scale = np.diag([factor, factor, 1.0])
                warp = (scale @ transform @ np.linalg.inv(scale)).astype(np.float32)
                source_lines = cv2.GaussianBlur(
                    cv2.resize(self.lines, None, fx=factor, fy=factor).astype(np.float32) / 255,
                    (0, 0),
                    sigma,
                )
                target_lines = cv2.GaussianBlur(
                    cv2.resize(target.lines, None, fx=factor, fy=factor).astype(np.float32) / 255,
                    (0, 0),
                    sigma,
                )
                try:
                    score, warp = cv2.findTransformECC(
                        source_lines,
                        target_lines,
                        warp,
                        cv2.MOTION_HOMOGRAPHY,
                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                        None,
                        1,
                    )
                except cv2.error:
                    break
                transform = np.linalg.inv(scale) @ warp @ scale
                projected = cv2.perspectiveTransform(self.quad[None], transform)[0]
                # Refinement must stay near the independently detected paper.
                # This also rejects singular warps and slipping by table rows.
                if (
                    score < 0.65
                    or not np.isfinite(transform).all()
                    or not np.isfinite(projected).all()
                    or np.max(np.linalg.norm(projected - destination, axis=1))
                    > max(target.lines.shape) * 0.06
                ):
                    break
            else:
                # Ignore text in BOTH scans when validating the final alignment.
                size = (self.lines.shape[1], self.lines.shape[0])
                inverse = np.linalg.inv(transform)
                warped = cv2.warpPerspective(target_lines, inverse, size)
                valid = self.mask & (
                    cv2.warpPerspective(
                        target.mask.astype(np.uint8), inverse, size, flags=cv2.INTER_NEAREST
                    )
                    != 0
                )
                first, second = source_lines[valid], warped[valid]
                if first.size < self.lines.size * 0.1:
                    continue
                first, second = first - first.mean(), second - second.mean()
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                score = float(first @ second / denominator) if denominator > 0 else 0
                if score >= 0.8:
                    candidates.append((score, transform))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.04):
            return None
        return np.linalg.inv(target.scale) @ candidates[0][1] @ self.scale

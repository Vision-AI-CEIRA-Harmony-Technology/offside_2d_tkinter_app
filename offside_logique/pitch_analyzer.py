"""
Pitch Analysis for Offside Detection System
"""

import cv2
import numpy as np
import random
from typing import Tuple, Optional, List

from .config import (
    GREEN_HSV_LOWER, GREEN_HSV_UPPER,
    HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
    HOUGH_MIN_LENGTH, HOUGH_MAX_GAP, WHITE_LINE_THRESHOLD,
    VP_N_ITERATIONS, VP_DISTANCE_THRESHOLD, VP_EARLY_STOP_RATIO,
    VP_MIN_DISTANCE_FROM_CENTER
)
from .utils import GeometryUtils


class FieldSegmenter:
    """Segments the football pitch from frame."""
    
    @staticmethod
    def segment_field(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment field from frame.
        
        Returns:
            (masked_frame, binary_mask)
        """
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame.copy(), mask
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
        masked = cv2.bitwise_and(frame, frame, mask=clean_mask)
        return masked, clean_mask


class LineDetector:
    """Detects white lines on the pitch."""
    
    @staticmethod
    def detect_white_lines(masked_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """
        Detect white lines on pitch.
        
        Returns:
            (raw_white_lines, dominant_angles)
        """
        H, W = masked_frame.shape[:2]
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        field_pixels_v = V[masked_frame[:, :, 0] > 0]
        field_pixels_s = S[masked_frame[:, :, 0] > 0]
        v_threshold = np.percentile(field_pixels_v, 98)
        s_threshold = np.percentile(field_pixels_s, 5)
        white_mask = ((V > v_threshold) & (S < s_threshold)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        raw_white_lines = cv2.HoughLinesP(
            cleaned,
            rho=1,
            theta=np.pi / 180,
            threshold=WHITE_LINE_THRESHOLD,
            minLineLength=int(W * 0.001),
            maxLineGap=15
        )
        if raw_white_lines is None:
            return None, None
        angles = []
        white_lines = []
        for line in raw_white_lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            if 15 <= abs(angle) <= 89:
                angles.append(angle)
                white_lines.append(line)
        if not angles:
            return white_lines if white_lines else None, None
        hist, bin_edges = np.histogram(angles, bins=18, range=(-90, 90))
        top_indices = np.argsort(hist)[-2:][::-1]
        dominant_angles = []
        for idx in top_indices:
            if hist[idx] > 0:
                bin_center = (bin_edges[idx] + bin_edges[idx + 1]) / 2
                dominant_angles.append(float(bin_center))
        print(f"Dominant angles: {[f'{a:.1f}°' for a in dominant_angles]}")
        return np.array(white_lines) if white_lines else None, dominant_angles if dominant_angles else None

    @staticmethod
    def detect_pitch_lines(masked_frame: np.ndarray) -> List[Tuple]:
        """
        Detect pitch lines using saturation channel.
        
        Returns:
            List of lines as ((x1,y1),(x2,y2))
        """
        H, W = masked_frame.shape[:2]
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        sat = cv2.equalizeHist(sat)
        k = max(5, int(min(H, W) * 0.006) | 1)
        blurred = cv2.GaussianBlur(sat, (k + 4, k + 4), 0)
        median = np.median(sat)
        lower = int(max(0, 0.33 * median))
        upper = int(min(255, 1.66 * median))
        edges = cv2.Canny(blurred, lower, upper)
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LENGTH,
            maxLineGap=HOUGH_MAX_GAP,
        )
        white_lines, dominant_angles = LineDetector.detect_white_lines(masked_frame)
        if raw_lines is None and white_lines is None:
            return []
        if white_lines is not None:
            if raw_lines is not None:
                raw_lines = np.append(raw_lines, white_lines, axis=0)
            else:
                raw_lines = white_lines
        if raw_lines is None:
            return []
        lines = []
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if dominant_angles is None or any(abs(angle - dom) <= 5 for dom in dominant_angles):
                lines.append(((x1, y1), (x2, y2)))
        print(f"Detected {len(lines)} pitch lines")
        return lines


class VanishingPointEstimator:
    """Estimates vanishing point using RANSAC."""
    
    @staticmethod
    def compute_vanishing_point(
        lines: List[Tuple],
        frame: np.ndarray,
        n_iter: int = VP_N_ITERATIONS,
        threshold: float = VP_DISTANCE_THRESHOLD,
        early_stop: float = VP_EARLY_STOP_RATIO
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate vanishing point using RANSAC.
        
        Args:
            lines: List of lines
            frame: Reference frame for dimensions
            n_iter: Number of RANSAC iterations
            threshold: Distance threshold for inliers
            early_stop: Early stop ratio
            
        Returns:
            Vanishing point (x, y) or None
        """
        h, w = frame.shape[:2]
        if len(lines) < 2:
            return None
        intersections = []
        for i, _ in enumerate(lines):
            for j in range(i + 1, len(lines)):
                pt = GeometryUtils.line_intersection(lines[i], lines[j])
                if pt is not None:
                    intersections.append(pt)
        print(f"Computed {len(intersections)} pairwise intersections")
        if len(intersections) < 2:
            return None
        pts = np.array(intersections)
        middle = np.array([w / 2, h / 2])
        dists = np.linalg.norm(pts - middle, axis=1)
        pts = pts[dists > w * VP_MIN_DISTANCE_FROM_CENTER]
        pts = pts[pts[:, 1] < h]
        if len(pts) < 2:
            return None
        best_inliers = []
        for _ in range(n_iter):
            idx = random.sample(range(len(pts)), 2)
            p1, p2 = pts[idx[0]], pts[idx[1]]
            d = p2 - p1
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                continue
            diffs = pts - p1
            dists = np.abs(diffs[:, 0] * d[1] - diffs[:, 1] * d[0]) / norm
            inliers = pts[dists < threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                if len(best_inliers) / len(pts) >= early_stop:
                    break
        if len(best_inliers) == 0:
            return None
        vp = best_inliers.mean(axis=0)
        return (float(vp[0]), float(vp[1]))


class PitchAnalyzer:
    """Main pitch analysis orchestrator."""
    
    def __init__(self):
        """Initialize pitch analyzer."""
        self.field_segmenter = FieldSegmenter()
        self.line_detector = LineDetector()
        self.vp_estimator = VanishingPointEstimator()
    
    def analyze_pitch(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple], Optional[Tuple]]:
        """
        Full pitch analysis pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            (masked_frame, pitch_lines, vanishing_point)
        """
        masked, _ = self.field_segmenter.segment_field(frame)
        pitch_lines = self.line_detector.detect_pitch_lines(masked)
        vp = self.vp_estimator.compute_vanishing_point(pitch_lines, frame) if pitch_lines else None
        return masked, pitch_lines, vp

"""
Utility functions for Offside Detection System
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .config import OFFSIDE_KP_IDX


class ColorUtils:
    """Utility class for color operations."""

    @staticmethod
    def parse_color(color_str: Optional[str]) -> Optional[Tuple[int, int, int]]:
        """
        Parse color string to BGR tuple.
        
        Args:
            color_str: Color string in format "R,G,B"
            
        Returns:
            BGR tuple or None if parsing fails
        """
        if color_str is None:
            return None
        try:
            return tuple(int(c) for c in color_str.split(","))
        except Exception:
            return None

    @staticmethod
    def bgr_to_display(color: np.ndarray) -> Tuple[int, int, int]:
        """Convert KMeans BGR float color to int tuple."""
        return tuple(int(c) for c in color)


class GeometryUtils:
    """Utility class for geometric operations."""
    @staticmethod
    def clip_line_to_pitch_only_on_vp_side(
        line_pts,
        mask,
        vp
    ):

        import numpy as np
        import cv2

        H, W = mask.shape[:2]

        p1, p2 = line_pts

        # determine which endpoint is closer to VP
        d1 = np.linalg.norm(np.array(p1) - np.array(vp))
        d2 = np.linalg.norm(np.array(p2) - np.array(vp))

        if d1 < d2:
            vp_side = p1
            far_side = p2
        else:
            vp_side = p2
            far_side = p1

        # sample points along line
        n = 2000

        xs = np.linspace(vp_side[0], far_side[0], n)
        ys = np.linspace(vp_side[1], far_side[1], n)

        inside_pts = []

        for x, y in zip(xs, ys):

            xi = int(np.clip(x, 0, W - 1))
            yi = int(np.clip(y, 0, H - 1))

            if mask[yi, xi] > 0:
                inside_pts.append((xi, yi))

            elif len(inside_pts) > 0:
                break

        if len(inside_pts) == 0:
            return line_pts

        clipped_vp_side = inside_pts[0]

        return (
            clipped_vp_side,
            far_side
        )
    @staticmethod
    def compute_angle(line: Tuple) -> float:
        """
        Returns angle in degrees in range [0, 180).
        
        Args:
            line: Tuple of ((x1,y1), (x2,y2))
            
        Returns:
            Angle in degrees
        """
        (x1, y1), (x2, y2) = line
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    @staticmethod
    def compute_iou(box1: Tuple, box2: Tuple) -> float:
        """
        Compute Intersection over Union between two boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU score in [0, 1]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def line_intersection(l1: Tuple, l2: Tuple) -> Optional[Tuple[float, float]]:
        """
        Compute intersection of two lines.
        
        Args:
            l1: ((x1,y1),(x2,y2))
            l2: ((x3,y3),(x4,y4))
            
        Returns:
            Intersection point or None
        """
        (x1, y1), (x2, y2) = l1
        (x3, y3), (x4, y4) = l2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    @staticmethod
    def extend_line_to_frame(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        W: int,
        H: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Extends the line through p1->p2 to the frame borders.
        
        Args:
            p1: First point (x1, y1)
            p2: Second point (x2, y2)
            W: Frame width
            H: Frame height
            
        Returns:
            Tuple of two endpoints on frame border
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if abs(x2 - x1) < 1e-6:
            return (int(x1), 0), (int(x1), H)
        
        slope = (y2 - y1) / (x2 - x1)
        pts = []
        
        for x in [0, W]:
            y = slope * (x - x1) + y1
            if 0 <= y <= H:
                pts.append((int(x), int(y)))

        for y in [0, H]:
            x = (y - y1) / slope + x1 if abs(slope) > 1e-6 else x1
            if 0 <= x <= W:
                pts.append((int(x), int(y)))

        if len(pts) < 2:
            return (int(x2), int(y2)), (int(x2), int(y2))
        return pts[0], pts[-1]


class BBoxUtils:
    """Utility class for bounding box operations."""

    @staticmethod
    def extract_roi(frame: np.ndarray, bbox: Tuple, vertical_range: Tuple[float, float], 
                   horizontal_range: Tuple[float, float] = (0.0, 1.0)) -> Optional[np.ndarray]:
        """
        Extract region of interest from frame using bbox.
        
        Args:
            frame: Input image
            bbox: (x1, y1, x2, y2)
            vertical_range: (top_fraction, bottom_fraction) of bbox height
            horizontal_range: (left_fraction, right_fraction) of bbox width
            
        Returns:
            ROI or None if invalid
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        
        ry1 = int(y1 + h * vertical_range[0])
        ry2 = int(y1 + h * vertical_range[1])
        rx1 = int(x1 + w * horizontal_range[0])
        rx2 = int(x2 - w * (1 - horizontal_range[1]))
        
        roi = frame[ry1:ry2, rx1:rx2]
        return roi if roi.size > 0 else None

    @staticmethod
    def get_bbox_center(bbox: Tuple) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def filter_boxes_by_iou(boxes: list, scores: list, iou_threshold: float = 0.5) -> list:
        """
        Filter overlapping boxes using IoU threshold.
        
        Args:
            boxes: List of boxes (x1, y1, x2, y2)
            scores: List of confidence scores
            iou_threshold: IoU threshold for filtering
            
        Returns:
            Indices of kept boxes
        """
        if not boxes:
            return []
        
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        
        while idxs:
            cur = idxs.pop(0)
            keep.append(cur)
            idxs = [i for i in idxs if GeometryUtils.compute_iou(boxes[cur], boxes[i]) < iou_threshold]
        
        return keep


class KeypointUtils:
    """Utility class for keypoint operations."""

    @staticmethod
    def get_offside_keypoint_x(det: dict, direction: str) -> float:
        """
        Get the most advanced keypoint x-coordinate for a detection.
        
        Args:
            det: Detection dict with 'keypoints'
            direction: 'left' or 'right'
            
        Returns:
            X-coordinate of most advanced keypoint
        """
        kp_xs = [kp[0] for kp in det["keypoints"].values()]
        if not kp_xs:
            x1, _, x2, _ = det["bbox"]
            kp_xs = [(x1 + x2) / 2.0]
        
        return max(kp_xs) if direction == "right" else min(kp_xs)

    @staticmethod
    def extract_offside_keypoints(all_kps: np.ndarray, conf_threshold: float) -> dict:
        """
        Extract only offside-relevant keypoints from all keypoints.
        
        Args:
            all_kps: Array of shape (17, 3) with x, y, confidence
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary of keypoint names to (x, y) tuples
        """
        offside_kps = {}
        for name, idx in OFFSIDE_KP_IDX.items():
            kx, ky, kc = all_kps[idx]
            if kc >= conf_threshold:
                offside_kps[name] = (float(kx), float(ky))
        return offside_kps

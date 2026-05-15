"""
Team Classification for Offside Detection System
"""

import cv2
import numpy as np
from typing import Tuple, List

from .config import JERSEY_SAMPLE_TOP, JERSEY_SAMPLE_BOTTOM, JERSEY_SAMPLE_LEFT, JERSEY_SAMPLE_RIGHT
from .utils import BBoxUtils


class TeamClassifier:
    """Classifies players into teams based on jersey color."""
    
    @staticmethod
    def extract_jersey_color(frame: np.ndarray, bbox: Tuple) -> np.ndarray:

        roi = BBoxUtils.extract_roi(
            frame,
            bbox,
            vertical_range=(JERSEY_SAMPLE_TOP, JERSEY_SAMPLE_BOTTOM),
            horizontal_range=(JERSEY_SAMPLE_LEFT, 1.0 - JERSEY_SAMPLE_RIGHT)
        )

        if roi is None or roi.size < 5:
            return None

        pixels = roi.reshape(-1, 3).astype(np.float32)

        return np.mean(pixels, axis=0)

    @staticmethod
    def classify_teams_with_priors(
        frame,
        detections,
        center0,
        center1,
    ):

        labels = []

        for det in detections:

            c = TeamClassifier.extract_jersey_color(
                frame,
                det["bbox"]
            )

            if c is None:
                labels.append(-1)
                continue

            d0 = np.linalg.norm(c - center0)
            d1 = np.linalg.norm(c - center1)

            labels.append(0 if d0 < d1 else 1)

        return labels, center0, center1
    
    @staticmethod
    def classify_teams(
        frame: np.ndarray,
        detections: List[dict],
    ):

        colors = []
        valid_idx = []

        for i, det in enumerate(detections):

            c = TeamClassifier.extract_jersey_color(
                frame,
                det["bbox"]
            )

            if c is not None:
                colors.append(c)
                valid_idx.append(i)

        labels = [-1] * len(detections)

        c0 = np.array([200, 200, 200], dtype=np.float32)
        c1 = np.array([100, 100, 100], dtype=np.float32)

        if len(colors) < 2:
            return labels, c0, c1

        arr = np.array(colors)

        brightness = arr.mean(axis=1)

        threshold = np.median(brightness)

        for i, vi in enumerate(valid_idx):

            labels[vi] = 0 if brightness[i] < threshold else 1

        team0 = arr[brightness < threshold]
        team1 = arr[brightness >= threshold]

        if len(team0) > 0:
            c0 = np.mean(team0, axis=0)

        if len(team1) > 0:
            c1 = np.mean(team1, axis=0)

        return labels, c0, c1

class AttackDirectionDetector:
    """Detects attacking team and direction."""
    
    @staticmethod
    def detect_attack_direction(
        detections: List[dict],
        team_labels: List[int],
        frame_w: int,
    ) -> dict:
        """
        Detect attacking team and direction.
        
        Heuristic: the team whose players are concentrated in the
        opponent's half is the attacking team.
        
        Args:
            detections: List of detection dicts
            team_labels: List of team labels
            frame_w: Frame width
            
        Returns:
            {
                'attacking_team': 0|1,
                'defending_team': 0|1,
                'direction': 'left'|'right'
            }
        """
        if not detections:
            return {
                "attacking_team": 0,
                "defending_team": 1,
                "direction": "right"
            }
        team_centers = {0: [], 1: []}
        for det, lbl in zip(detections, team_labels):
            if lbl < 0:
                continue
            x1, _, x2, _ = det["bbox"]
            cx = (x1 + x2) / 2.0
            team_centers[lbl].append(cx)
        median = {}
        for t in [0, 1]:
            if team_centers[t]:
                median[t] = float(np.median(team_centers[t]))
            else:
                median[t] = frame_w / 2.0
        dist0 = abs(median[0] - frame_w / 2.0)
        dist1 = abs(median[1] - frame_w / 2.0)
        attacking_team = 0 if dist0 >= dist1 else 1
        defending_team = 1 - attacking_team
        def_xs = [
            (det["bbox"][0] + det["bbox"][2]) / 2.0
            for det, lbl in zip(detections, team_labels)
            if lbl == defending_team
        ]
        if def_xs:
            def_median = float(np.median(def_xs))
            direction = "left" if def_median > frame_w / 2.0 else "right"
        else:
            direction = "right"
        return {
            "attacking_team": defending_team,
            "defending_team": attacking_team,
            "direction": direction,
        }

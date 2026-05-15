"""
Offside Detection and Judgement for Offside Detection System
"""

import numpy as np
from typing import Tuple, List, Optional

from .utils import GeometryUtils, KeypointUtils


class OffsideLineComputer:
    """Computes the offside line based on defender positions."""
    
    @staticmethod
    def compute_projection_y(
        kx: float, ky: float,
        vp: Tuple[float, float],
        x_axis: int
    ) -> float:
        """
        Project point onto axis using vanishing point.
        
        Args:
            kx, ky: Keypoint position
            vp: Vanishing point
            x_axis: Target x-coordinate (0 or W)
            
        Returns:
            Projected y-coordinate
        """
        x1, y1 = vp
        if abs(kx - x1) < 1e-6:
            return y1
        slope = (ky - y1) / (kx - x1)
        return y1 + slope * (x_axis - x1)
    
    @staticmethod
    def collect_defender_keypoints(
        detections: List[dict],
        team_labels: List[int],
        defending_team: int,
        vp: Tuple[float, float],
        x_axis: int
    ) -> List[Tuple]:
        """
        Collect all defender keypoints with projection.
        
        Returns:
            List of (kx, ky, proj_y, det_idx)
        """
        defender_candidates = []
        for i, (det, lbl) in enumerate(zip(detections, team_labels)):
            if lbl != defending_team:
                continue
            kps = list(det["keypoints"].values())
            if not kps:
                x1, _, x2, _ = det["bbox"]
                kps = [((x1 + x2) / 2.0, (det["bbox"][1] + det["bbox"][3]) / 2.0)]
            projected_candidates = []
            for kp in kps:

                kx, ky = kp

                # project kp to ground
                projected_point = (kx, det["bbox"][3])

                proj_y = OffsideLineComputer.compute_projection_y(
                    projected_point[0],
                    projected_point[1],
                    vp,
                    x_axis
                )

                projected_candidates.append(
                    (
                        kp,                 # original keypoint
                        projected_point,    # projected ground point
                        proj_y
                    )
                )

            # select best projected point
            best_proj = min(
                projected_candidates,
                key=lambda p: p[2]
            )

            selected_kp, attacker_ground_point, adv_proj = best_proj

            defender_candidates.append(
                (
                    selected_kp[0],
                    selected_kp[1],
                    proj_y,
                    i
                )
            )
        return defender_candidates
    
    @staticmethod
    def compute_offside_line(
        vanishing_point,
        detections,
        team_labels,
        attack_info,
        frame_shape,
        manual_last_defender_kp=None
    ) -> Tuple:
        """
        Compute the offside line.

        Returns:
            (
                offside_line,
                ground_line,
                all_def_lines,
                last_kp,
                projected_point,
                projection_points,
                x_axis
            )
        """

        defending_team = attack_info["defending_team"]

        H, W = frame_shape[:2]

        vp = vanishing_point

        x_axis = W if vp[0] > W / 2 else 0

        defender_candidates = (
            OffsideLineComputer.collect_defender_keypoints(
                detections,
                team_labels,
                defending_team,
                vp,
                x_axis
            )
        )

        if not defender_candidates:
            return None, None, [], None, None, None, x_axis

        # sort by projected depth
        defender_candidates.sort(key=lambda c: c[2])

        # selected defender
        if manual_last_defender_kp is not None:

            last_kp = (
                manual_last_defender_kp[0],
                manual_last_defender_kp[1],
                0,
                -1
            )

        else:

            last_kp = defender_candidates[0]

        # -------------------------------------------------
        # PROJECT KEYPOINT TO GROUND (BOTTOM OF BBOX)
        # -------------------------------------------------

        kx, ky = last_kp[0], last_kp[1]

        _, _, _, last_det_idx = last_kp

        # default bbox
        proj_bbox = detections[last_det_idx]["bbox"]

        # if manually dragged kp -> find matching bbox
        if manual_last_defender_kp is not None:

            for other_det in detections:

                bx1, by1, bx2, by2 = other_det["bbox"]

                if bx1 <= kx <= bx2 and by1 <= ky <= by2:

                    proj_bbox = other_det["bbox"]

                    break

        # bottom of bbox
        _, _, _, proj_y2 = proj_bbox

        # projected ground point
        projected_point = (kx, proj_y2)

        # -------------------------------------------------
        # OFFSIDE LINE NOW USES PROJECTED POINT
        # -------------------------------------------------

        offside_line = GeometryUtils.extend_line_to_frame(
            vp,
            projected_point,
            W,
            H
        )

        # optional separate ground line
        ground_line = offside_line

        # -------------------------------------------------
        # ALL DEFENDER LINES
        # -------------------------------------------------

        all_def_lines = []

        seen = set()

        for det, lbl in zip(detections, team_labels):

            if lbl != defending_team:
                continue

            if id(det) in seen:
                continue

            seen.add(id(det))

            x1, _, x2, y2 = det["bbox"]

            cx = (x1 + x2) / 2

            defender_ground_point = (cx, y2)

            all_def_lines.append(
                GeometryUtils.extend_line_to_frame(
                    vp,
                    defender_ground_point,
                    W,
                    H
                )
            )

        # -------------------------------------------------
        # DEBUG / VISUALIZATION POINTS
        # -------------------------------------------------

        projection_points = [
            (int(x_axis), int(proj_y))
            for _, _, proj_y, _ in defender_candidates
        ]

        return (
            offside_line,
            ground_line,
            all_def_lines,
            last_kp,
            projected_point,
            projection_points,
            x_axis
        )


class OffsideJudge:
    """Judges whether players are in offside position."""
    
    @staticmethod
    def judge_attackers(
        detections: List[dict],
        team_labels: List[int],
        attack_info: dict,
        offside_line: Optional[Tuple],
        vanishing_point: Optional[Tuple[float, float]],
        frame_shape: Tuple[int, int]
    ) -> List[str]:
        """
        Judge offside status for all players.
        """
        if offside_line is None or vanishing_point is None:
            return [""] * len(detections)

        attacking_team = attack_info["attacking_team"]
        vp = vanishing_point
        H, W = frame_shape[:2]
        x_axis = W if vp[0] > W / 2 else 0

        def line_projection_y():
            (x1, y1), (x2, y2) = offside_line
            midx = (x1 + x2) / 2
            midy = (y1 + y2) / 2
            return OffsideLineComputer.compute_projection_y(midx, midy, vp, x_axis)

        offside_proj = line_projection_y()
        judgements = [""] * len(detections)

        manual_attacker_idx = None

        # Find if any player has manual keypoint
        for i, det in enumerate(detections):
            if det.get("manual_offside_kp") is not None:
                manual_attacker_idx = i
                break

        for i, (det, lbl) in enumerate(zip(detections, team_labels)):
            if lbl != attacking_team:
                continue

            manual_kp = det.get("manual_offside_kp")

            # Use manual keypoint if available
            if manual_kp is not None:
                kx, ky = manual_kp

                # Find correct bbox for projection (important when switching players)
                proj_bbox = det["bbox"]
                for other_det in detections:
                    bx1, by1, bx2, by2 = other_det["bbox"]
                    if bx1 <= kx <= bx2 and by1 <= ky <= by2:
                        proj_bbox = other_det["bbox"]
                        break

                _, _, _, proj_y2 = proj_bbox
                projected_point = (kx, proj_y2)

                proj_y = OffsideLineComputer.compute_projection_y(
                    projected_point[0], projected_point[1], vp, x_axis
                )

                det["offside_keypoint"] = manual_kp
                det["offside_proj_point"] = projected_point

                is_offside = proj_y < offside_proj
                judgements[i] = "OFFSIDE" if is_offside else "ONSIDE"

            else:
                # Normal keypoint logic (unchanged)
                kps = list(det["keypoints"].values())
                if not kps:
                    x1, _, x2, _ = det["bbox"]
                    kps = [((x1 + x2) / 2.0, (det["bbox"][1] + det["bbox"][3]) / 2.0)]

                projected_candidates = []
                for kp in kps:
                    kx, ky = kp
                    projected_point = (kx, det["bbox"][3])
                    proj_y = OffsideLineComputer.compute_projection_y(
                        projected_point[0], projected_point[1], vp, x_axis
                    )
                    projected_candidates.append((kp, projected_point, proj_y))

                if projected_candidates:
                    best = min(projected_candidates, key=lambda p: p[2])
                    selected_kp, attacker_ground_point, adv_proj = best

                    det["offside_keypoint"] = selected_kp
                    det["offside_proj_point"] = attacker_ground_point

                    is_offside = adv_proj < offside_proj
                    judgements[i] = "OFFSIDE" if is_offside else "ONSIDE"

        # Force manual attacker as OFFSIDE if present
        if manual_attacker_idx is not None:
            for j in range(len(judgements)):
                judgements[j] = "ONSIDE"
            judgements[manual_attacker_idx] = "OFFSIDE"

        return judgements

class OffsideDetector:
    """Main offside detection orchestrator."""
    
    def __init__(self):
        """Initialize offside detector."""
        self.line_computer = OffsideLineComputer()
        self.judge = OffsideJudge()
    
    def compute_offside_status(
        self,
        detections,
        team_labels,
        attack_info,
        vanishing_point,
        frame_shape,
        manual_last_defender_kp=None
    ):
        """
        Full offside detection pipeline.
        
        Returns:
            (offside_line, ground_line, all_def_lines, last_kp, projected_point,
             projection_points, x_axis, judgements)
        """
        if vanishing_point is None:
            return None, None, [], None, None, None, None, [""] * len(detections)
        offside_line, ground_line, all_def_lines, last_kp, projected_point, \
        projection_points, x_axis = self.line_computer.compute_offside_line(
            vanishing_point, detections, team_labels, attack_info, frame_shape, manual_last_defender_kp
        )
        judgements = self.judge.judge_attackers(
            detections, team_labels, attack_info, offside_line, vanishing_point, frame_shape
        )
        return (
            offside_line, ground_line, all_def_lines, last_kp,
            projected_point, projection_points, x_axis, judgements
        )

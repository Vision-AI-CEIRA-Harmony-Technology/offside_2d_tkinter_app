"""
Offside Detection and Judgement for Offside Detection System
"""

import numpy as np
from typing import Tuple, List, Optional

from .utils import GeometryUtils, KeypointUtils


class OffsideLineComputer:
    """Computes the offside line based on defender positions."""
    
    @staticmethod
    def compute_depth_metric(
        px: float,
        py: float,
        vp: Tuple[float, float],
        frame_w: int
    ) -> float:

        vx, vy = vp

        if 0 <= vx <= frame_w:

            denom = (py - vy)

            if abs(denom) < 1e-6:
                return px

            t = -vy / denom

            x_intersect = vx + t * (px - vx)

            return x_intersect

        else:

            x_axis = frame_w if vx > frame_w / 2 else 0

            if abs(px - vx) < 1e-6:
                return vy

            slope = (py - vy) / (px - vx)

            return vy + slope * (x_axis - vx)
    
    @staticmethod
    def collect_defender_keypoints(
        detections: List[dict],
        team_labels: List[int],
        defending_team: int,
        vp: Tuple[float, float],
        frame_w: int,
        attack_direction: str
    ) -> List[Tuple]:

        """
        Collect all defender keypoints with projection.

        Returns:
            List of (kx, ky, depth_metric, det_idx)
        """

        defender_candidates = []

        for i, (det, lbl) in enumerate(zip(detections, team_labels)):

            if lbl != defending_team:
                continue

            kps = list(det["keypoints"].values())

            if not kps:

                x1, _, x2, _ = det["bbox"]

                kps = [
                    (
                        (x1 + x2) / 2.0,
                        (det["bbox"][1] + det["bbox"][3]) / 2.0
                    )
                ]

            projected_candidates = []

            for kp in kps:

                kx, ky = kp


                projected_point = (
                    kx,
                    det["bbox"][3]
                )

                depth_metric = (
                    OffsideLineComputer.compute_depth_metric(
                        projected_point[0],
                        projected_point[1],
                        vp,
                        frame_w
                    )
                )

                projected_candidates.append(
                    (
                        kp,
                        projected_point,
                        depth_metric
                    )
                )

            # --------------------------------------------------
            # choose most advanced body part
            # --------------------------------------------------

            # --------------------------------------------------
            # Choose comparison direction according to:
            # 1. attack direction
            # 2. VP side
            # --------------------------------------------------

            vx, _ = vp

            # --------------------------------------------------
            # VP INSIDE IMAGE
            # --------------------------------------------------

            if 0 <= vx <= frame_w:

                if attack_direction == "right":

                    best_proj = max(
                        projected_candidates,
                        key=lambda p: p[2]
                    )

                else:

                    best_proj = min(
                        projected_candidates,
                        key=lambda p: p[2]
                    )

            # --------------------------------------------------
            # VP OUTSIDE IMAGE
            # --------------------------------------------------

            else:

                vp_is_right = vx > frame_w / 2

                if attack_direction == "right":

                    if vp_is_right:
                        best_proj = max(
                            projected_candidates,
                            key=lambda p: p[2]
                        )
                    else:
                        best_proj = min(
                            projected_candidates,
                            key=lambda p: p[2]
                        )

                else:

                    if vp_is_right:
                        best_proj = min(
                            projected_candidates,
                            key=lambda p: p[2]
                        )
                    else:
                        best_proj = max(
                            projected_candidates,
                            key=lambda p: p[2]
                        )

            selected_kp, defender_ground_point, depth_metric = best_proj

            defender_candidates.append(
                (
                    selected_kp[0],
                    selected_kp[1],
                    depth_metric,
                    i
                )
            )

        return defender_candidates

    @staticmethod
    def compute_offside_line(
        vanishing_point,
        vanishig_point_horiz,
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

        defender_candidates = (
            OffsideLineComputer.collect_defender_keypoints(
                detections,
                team_labels,
                defending_team,
                vp,
                W,
                attack_info["direction"]
            )
        )

        if not defender_candidates:
            return None, None, [], None, None, None, None

        # sort by projected depth
        attack_direction = attack_info["direction"]


        vx, _ = vp

        # --------------------------------------------------
        # VP INSIDE IMAGE
        # --------------------------------------------------

        if 0 <= vx <= W:

            reverse_sort = (
                attack_info["direction"] == "right"
            )

        # --------------------------------------------------
        # VP OUTSIDE IMAGE
        # --------------------------------------------------

        else:

            vp_is_right = vx > W / 2

            if attack_info["direction"] == "right":

                reverse_sort = vp_is_right

            else:

                reverse_sort = not vp_is_right

        defender_candidates.sort(
            key=lambda c: c[2],
            reverse=reverse_sort
        )

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

        projection_points = []

        vx, vy = vp

        if 0 <= vx <= W:

            for _, _, metric, _ in defender_candidates:

                projection_points.append(
                    (int(metric), 0)
                )

        else:

            x_axis = W if vx > W / 2 else 0

            for _, _, metric, _ in defender_candidates:

                projection_points.append(
                    (int(x_axis), int(metric))
                )

        return (
            offside_line,
            ground_line,
            all_def_lines,
            last_kp,
            projected_point,
            projection_points,
            None
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
        vanishing_point_horiz: Optional[Tuple[float, float]],
        frame_shape: Tuple[int, int]
    ) -> List[str]:
        """
        Judge offside status for all players.
        
        Args:
            detections: List of detection dicts
            team_labels: List of team labels
            attack_info: Attack direction info
            offside_line: Offside line coordinates
            vanishing_point: Vanishing point
            vanishing_point_horiz: Horizontal vanishing point
            frame_shape: Frame dimensions
            
        Returns:
            List of judgements ("OFFSIDE", "ONSIDE", or "")
        """
        if offside_line is None or vanishing_point is None:
            return [""] * len(detections)

        attacking_team = attack_info["attacking_team"]
        vp = vanishing_point

        H, W = frame_shape[:2]

        vx, _ = vp
        vp_is_right = vx > W / 2

        line_mid_x = (
            offside_line[0][0] + offside_line[1][0]
        ) / 2

        line_mid_y = (
            offside_line[0][1] + offside_line[1][1]
        ) / 2

        offside_metric = (
            OffsideLineComputer.compute_depth_metric(
                line_mid_x,
                line_mid_y,
                vp,
                W
            )
        )

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

                depth_metric = OffsideLineComputer.compute_depth_metric(
                    projected_point[0],
                    projected_point[1],
                    vp,
                    W
                )

                # projected_candidates.append(
                #     (
                #         manual_kp,                 # original keypoint
                #         projected_point,    # projected ground point
                #         proj_y
                #     )
                # )
                det["offside_keypoint"] = manual_kp
                det["offside_proj_point"] = projected_point

                attack_direction = attack_info["direction"]

                # --------------------------------------------------
                # VP INSIDE IMAGE
                # --------------------------------------------------

                if 0 <= vx <= W:

                    if attack_direction == "right":
                        is_offside = depth_metric > offside_metric
                    else:
                        is_offside = depth_metric < offside_metric

                # --------------------------------------------------
                # VP OUTSIDE IMAGE
                # --------------------------------------------------

                else:

                    vp_is_right = vx > W / 2

                    if attack_direction == "right":

                        if vp_is_right:
                            is_offside = depth_metric > offside_metric
                        else:
                            is_offside = depth_metric < offside_metric

                    else:

                        if vp_is_right:
                            is_offside = depth_metric < offside_metric
                        else:
                            is_offside = depth_metric > offside_metric
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
                    depth_metric = OffsideLineComputer.compute_depth_metric(
                        projected_point[0],
                        projected_point[1],
                        vp,
                        W
                    )
                    projected_candidates.append((kp, projected_point, depth_metric))

                if projected_candidates:
                    attack_direction = attack_info["direction"]
                    if attack_direction == "right":

                        if vp_is_right:
                            best = max(
                                projected_candidates,
                                key=lambda p: p[2]
                            )
                        else:
                            best = min(
                                projected_candidates,
                                key=lambda p: p[2]
                            )

                    else:

                        if vp_is_right:
                            best = min(
                                projected_candidates,
                                key=lambda p: p[2]
                            )
                        else:
                            best = max(
                                projected_candidates,
                                key=lambda p: p[2]
                            )

                    selected_kp, attacker_ground_point, adv_proj = best

                    det["offside_keypoint"] = selected_kp
                    det["offside_proj_point"] = attacker_ground_point

                    attack_direction = attack_info["direction"]

                    # --------------------------------------------------
                    # VP INSIDE IMAGE
                    # --------------------------------------------------

                    if 0 <= vx <= W:

                        if attack_direction == "right":
                            is_offside = adv_proj > offside_metric
                        else:
                            is_offside = adv_proj < offside_metric

                    # --------------------------------------------------
                    # VP OUTSIDE IMAGE
                    # --------------------------------------------------

                    else:

                        vp_is_right = vx > W / 2

                        if attack_direction == "right":

                            if vp_is_right:
                                is_offside = adv_proj > offside_metric
                            else:
                                is_offside = adv_proj < offside_metric

                        else:

                            if vp_is_right:
                                is_offside = adv_proj < offside_metric
                            else:
                                is_offside = adv_proj > offside_metric
                    judgements[i] = "OFFSIDE" if is_offside else "ONSIDE"

        # Force manual attacker as OFFSIDE if present
        if manual_attacker_idx is not None:
            for j in range(len(judgements)):
                judgements[j] = "ONSIDE"
            judgements[manual_attacker_idx] = "OFFSIDE"
            return judgements

        # If no attacker is offside, mark the closest onside attacker to the offside line
        if not any(j == "OFFSIDE" for j in judgements):
            attacking_team = attack_info["attacking_team"]
            best_delta = float("inf")
            best_idx = None
            for i, (det, lbl, judgement) in enumerate(zip(detections, team_labels, judgements)):
                if lbl != attacking_team or judgement != "ONSIDE":
                    continue
                proj_point = det.get("offside_proj_point")
                if proj_point is None:
                    continue
                metric = OffsideLineComputer.compute_depth_metric(
                    proj_point[0], proj_point[1], vp, W
                )
                delta = abs(metric - offside_metric)
                if delta < best_delta:
                    best_delta = delta
                    best_idx = i
            if best_idx is not None:
                judgements[best_idx] = "CLOSEST"

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
        vanishig_point_horiz,
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
            vanishing_point, vanishig_point_horiz, detections, team_labels, attack_info, frame_shape, manual_last_defender_kp
        )
        judgements = self.judge.judge_attackers(
            detections, team_labels, attack_info, offside_line, vanishing_point, vanishig_point_horiz, frame_shape
        )
        return (
            offside_line, ground_line, all_def_lines, last_kp,
            projected_point, projection_points, x_axis, judgements
        )

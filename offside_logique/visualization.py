"""
Visualization for Offside Detection System
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from .config import (
    TEAM_COLORS, OFFSIDE_COLOR, ONSIDE_COLOR, VP_COLOR, KEYPOINT_COLOR, LAST_DEF_KEYPOINT_COLOR
)
from .offside_detector import OffsideLineComputer
from .utils import ColorUtils, GeometryUtils


class KeypointVisualizer:
    """Visualizes keypoints for debugging."""
    
    @staticmethod
    def draw_keypoints_image(
        frame: np.ndarray,
        detections: List[dict],
        all_keypoints: bool = False,
        offside_line: Optional[Tuple] = None,
        vanishing_point: Optional[Tuple] = None,
        vanishing_point_horiz: Optional[Tuple] = None,
        last_kp: Optional[Tuple] = None,
        projected_point: Optional[Tuple] = None,
        projection_points: Optional[List[Tuple]] = None,
        judgements: Optional[List[str]] = None,
        only_offside_attackers: bool = False,
        pitch_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        out = frame.copy()
        out_save = frame.copy()

        H, W = frame.shape[:2]

        ref_diag = (1280**2 + 720**2) ** 0.5
        img_diag = (W**2 + H**2) ** 0.5

        ui_scale = np.clip(
            img_diag / ref_diag,
            0.18,
            3.0
        )

        line_thickness = max(1, int(1.5 * ui_scale))
        point_radius = max(1, int(2 * ui_scale))
        big_point_radius = min(5, int(3 * ui_scale))

        dash_length = max(6, int(12 * ui_scale))
        gap_length = max(4, int(8 * ui_scale))

        text_scale = max(0.35, 0.55 * ui_scale)
        text_thickness = max(1, int(1.2 * ui_scale))

        # =========================================================
        # ATTACKERS
        # =========================================================

        for det_idx, det in enumerate(detections):

            judgement = None

            if judgements is not None and det_idx < len(judgements):
                judgement = judgements[det_idx]

            if judgement not in ["OFFSIDE", "CLOSEST"]:
                continue

            if det.get("offside_keypoint") is None:
                continue

            okx, oky = det["offside_keypoint"]
            color = OFFSIDE_COLOR if judgement == "OFFSIDE" else ONSIDE_COLOR
            label = "Potential Offside" if judgement == "OFFSIDE" else "Closest Attacker"

            # keypoint
            cv2.circle(
                out,
                (int(okx), int(oky)),
                big_point_radius,
                color,
                -1
            )

            cv2.circle(
                out_save,
                (int(okx), int(oky)),
                big_point_radius,
                color,
                -1
            )

            # label
            cv2.putText(
                out,
                label,
                (int(okx) + 3, int(oky) - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                color,
                text_thickness
            )

            proj = det.get("offside_proj_point")

            if proj is None:
                continue

            px, py = int(proj[0]), int(proj[1])

            # local connection
            cv2.line(
                out,
                (int(okx), int(oky)),
                (px, py),
                color,
                line_thickness,
                cv2.LINE_AA
            )

            cv2.line(
                out_save,
                (int(okx), int(oky)),
                (px, py),
                color,
                line_thickness,
                cv2.LINE_AA
            )

            cv2.circle(
                out,
                (px, py),
                point_radius,
                color,
                -1
            )

            cv2.circle(
                out_save,
                (px, py),
                point_radius,
                color,
                -1
            )

            vp_x, vp_y = map(int, vanishing_point)
            vph_x, vph_y = map(int, vanishing_point_horiz)

            full_line = GeometryUtils.extend_line_to_frame(
                (vp_x, vp_y),
                (px, py),
                W,
                H
            )

            full_lineh = GeometryUtils.extend_line_to_frame(
                (vph_x, vph_y),
                (px, py),
                W,
                H
            )

            if pitch_mask is not None:

                clipped_line = GeometryUtils.clip_line_to_pitch_only_on_vp_side(
                    full_line,
                    pitch_mask,
                    vanishing_point
                )

                clipped_lineh = GeometryUtils.clip_line_to_pitch_only_on_vp_side(
                    full_lineh,
                    pitch_mask,
                    vanishing_point_horiz
                )

                # horizontal VP line
                if clipped_lineh is not None:

                    # preview dashed
                    GeometryUtils.draw_dashed_line(
                        out,
                        clipped_lineh[0],
                        clipped_lineh[1],
                        color,
                        line_thickness,
                        dash_length,
                        gap_length
                    )

                    # save solid
                    cv2.line(
                        out_save,
                        clipped_line[0],
                        clipped_line[1],
                        color,
                        line_thickness,
                        cv2.LINE_AA
                    )

                # vertical VP line => preview only
                if clipped_line is not None:

                    GeometryUtils.draw_dashed_line(
                        out,
                        clipped_line[0],
                        clipped_line[1],
                        color,
                        line_thickness,
                        dash_length,
                        gap_length
                    )

        # =========================================================
        # LAST DEFENDER
        # =========================================================

        if last_kp is not None:

            lx, ly = int(last_kp[0]), int(last_kp[1])

            cv2.circle(
                out,
                (lx, ly),
                big_point_radius,
                LAST_DEF_KEYPOINT_COLOR,
                -1
            )

            cv2.circle(
                out_save,
                (lx, ly),
                big_point_radius,
                LAST_DEF_KEYPOINT_COLOR,
                -1
            )

            cv2.putText(
                out,
                "Last Def",
                (lx + 5, ly - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                LAST_DEF_KEYPOINT_COLOR,
                text_thickness
            )

        if last_kp is not None and projected_point is not None:

            px, py = int(projected_point[0]), int(projected_point[1])

            # local connection
            cv2.line(
                out,
                (int(last_kp[0]), int(last_kp[1])),
                (px, py),
                LAST_DEF_KEYPOINT_COLOR,
                line_thickness,
                cv2.LINE_AA
            )

            cv2.line(
                out_save,
                (int(last_kp[0]), int(last_kp[1])),
                (px, py),
                LAST_DEF_KEYPOINT_COLOR,
                line_thickness,
                cv2.LINE_AA
            )

            cv2.circle(
                out,
                (px, py),
                point_radius,
                LAST_DEF_KEYPOINT_COLOR,
                -1
            )

            cv2.circle(
                out_save,
                (px, py),
                point_radius,
                LAST_DEF_KEYPOINT_COLOR,
                -1
            )

            vp_x, vp_y = map(int, vanishing_point)
            vph_x, vph_y = map(int, vanishing_point_horiz)

            full_line = GeometryUtils.extend_line_to_frame(
                (vp_x, vp_y),
                (px, py),
                W,
                H
            )

            full_lineh = GeometryUtils.extend_line_to_frame(
                (vph_x, vph_y),
                (px, py),
                W,
                H
            )

            if pitch_mask is not None:

                clipped_line = GeometryUtils.clip_line_to_pitch_only_on_vp_side(
                    full_line,
                    pitch_mask,
                    vanishing_point
                )

                clipped_lineh = GeometryUtils.clip_line_to_pitch_only_on_vp_side(
                    full_lineh,
                    pitch_mask,
                    vanishing_point_horiz
                )

                # horizontal VP line
                if clipped_lineh is not None:

                    # preview dashed
                    GeometryUtils.draw_dashed_line(
                        out,
                        clipped_lineh[0],
                        clipped_lineh[1],
                        LAST_DEF_KEYPOINT_COLOR,
                        line_thickness,
                        dash_length,
                        gap_length
                    )

                    # save solid
                    cv2.line(
                        out_save,
                        clipped_line[0],
                        clipped_line[1],
                        LAST_DEF_KEYPOINT_COLOR,
                        line_thickness,
                        cv2.LINE_AA
                    )

                # vertical VP line => preview only
                if clipped_line is not None:

                    GeometryUtils.draw_dashed_line(
                        out,
                        clipped_line[0],
                        clipped_line[1],
                        LAST_DEF_KEYPOINT_COLOR,
                        line_thickness,
                        dash_length,
                        gap_length
                    )

        return out, out_save

class OverlayRenderer:
    """Renders final overlay with all visualization elements."""
    
    @staticmethod
    def draw_overlay(
        frame: np.ndarray,
        detections: List[dict],
        team_labels: List[int],
        team_colors: Tuple,
        pitch_lines: List[Tuple],
        vanishing_point: Optional[Tuple],
        offside_line: Optional[Tuple],
        ground_line: Optional[Tuple],
        all_defender_lines: List[Tuple],
        judgements: List[str],
        attack_info: dict,
        projection_points: Optional[List[Tuple]] = None,
        x_axis: Optional[int] = None,
        last_kp: Optional[Tuple] = None,
        projected_point: Optional[Tuple] = None
    ) -> np.ndarray:
        """
        Draw complete overlay on frame.
        
        Returns:
            Frame with overlay
        """
        out = frame.copy()
        H, W = out.shape[:2]
        c0 = ColorUtils.bgr_to_display(team_colors[0])
        c1 = ColorUtils.bgr_to_display(team_colors[1])
        team_display_colors = [c0, c1]

        closest_onside_attacker = None
        if vanishing_point is not None and offside_line is not None:
            attacking_team = attack_info.get("attacking_team") if attack_info else None
            has_offside = any(j == "OFFSIDE" for j in judgements)
            if not has_offside and attacking_team is not None:
                offside_mid_x = (offside_line[0][0] + offside_line[1][0]) / 2
                offside_mid_y = (offside_line[0][1] + offside_line[1][1]) / 2
                offside_metric = OffsideLineComputer.compute_depth_metric(
                    offside_mid_x,
                    offside_mid_y,
                    vanishing_point,
                    W
                )
                best_delta = float("inf")
                for idx, (det, lbl, judgement) in enumerate(zip(detections, team_labels, judgements)):
                    if lbl != attacking_team or judgement != "ONSIDE":
                        continue
                    proj = det.get("offside_proj_point")
                    if proj is None:
                        continue
                    metric = OffsideLineComputer.compute_depth_metric(
                        proj[0], proj[1], vanishing_point, W
                    )
                    delta = abs(metric - offside_metric)
                    if delta < best_delta:
                        best_delta = delta
                        closest_onside_attacker = idx

        if vanishing_point:
            vx, vy = int(vanishing_point[0]), int(vanishing_point[1])
            if 0 <= vx < W and 0 <= vy < H:
                cv2.circle(out, (vx, vy), 10, VP_COLOR, -1)
                cv2.circle(out, (vx, vy), 14, (0, 0, 0), 2)
        for idx, (det, lbl, judgement) in enumerate(zip(detections, team_labels, judgements)):
            x1, y1, x2, y2 = det["bbox"]
            if judgement == "OFFSIDE":
                color = OFFSIDE_COLOR
            elif idx == closest_onside_attacker:
                color = ONSIDE_COLOR
            else:
                color = team_display_colors[lbl] if lbl >= 0 else (180, 180, 180)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            for kp_name, (kx, ky) in det["keypoints"].items():
                cv2.circle(out, (int(kx), int(ky)), 1, KEYPOINT_COLOR, -1)
            if judgement == "OFFSIDE" and det.get("offside_keypoint") is not None:
                okx, oky = det["offside_keypoint"]
                cv2.circle(out, (int(okx), int(oky)), 1, OFFSIDE_COLOR, -1)
        return out


class VisualizationRenderer:
    """Main visualization orchestrator."""
    
    def __init__(self):
        """Initialize visualization renderer."""
        self.keypoint_viz = KeypointVisualizer()
        self.overlay_renderer = OverlayRenderer()
        self.dragging_offside_kp = None
        self.dragging_projection_kp = None
    
    def render_main_output(
        self,
        frame: np.ndarray,
        detections: List[dict],
        team_labels: List[int],
        team_colors: Tuple,
        pitch_lines: List[Tuple],
        vanishing_point: Optional[Tuple],
        offside_line: Optional[Tuple],
        ground_line: Optional[Tuple],
        all_defender_lines: List[Tuple],
        judgements: List[str],
        attack_info: dict,
        projection_points: Optional[List[Tuple]] = None,
        x_axis: Optional[int] = None,
        last_kp: Optional[Tuple] = None,
        projected_point: Optional[Tuple] = None
    ) -> np.ndarray:
        """Render main output image."""
        return self.overlay_renderer.draw_overlay(
            frame, detections, team_labels, team_colors,
            pitch_lines, vanishing_point,
            offside_line, ground_line, all_defender_lines,
            judgements, attack_info,
            projection_points, x_axis, last_kp, projected_point
        )
    
    def render_keypoints_image(
        self,
        frame: np.ndarray,
        detections: List[dict],
        all_keypoints: bool = False,
        offside_line: Optional[Tuple] = None,
        vanishing_point: Optional[Tuple] = None,
        vanishing_point_horiz: Optional[Tuple] = None,
        last_kp: Optional[Tuple] = None,
        projected_point: Optional[Tuple] = None,
        projection_points: Optional[List[Tuple]] = None,
        judgements: Optional[List[str]] = None,
        only_offside_attackers: bool = False,
        pitch_mask=None,
    ) -> np.ndarray:
        """Render keypoints debug image."""
        return self.keypoint_viz.draw_keypoints_image(
            frame,
            detections,
            all_keypoints,
            offside_line,
            vanishing_point,
            vanishing_point_horiz,
            last_kp,
            projected_point,
            projection_points,
            judgements=judgements,
            only_offside_attackers=only_offside_attackers,
            pitch_mask=pitch_mask  
        )
"""
Visualization for Offside Detection System
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from .config import (
    TEAM_COLORS, OFFSIDE_COLOR, ONSIDE_COLOR, VP_COLOR, KEYPOINT_COLOR, LAST_DEF_KEYPOINT_COLOR
)
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
        """
        Draw keypoints on frame.
        
        Args:
            frame: Input frame
            detections: List of detection dicts
            all_keypoints: Show all 17 keypoints or only offside keypoints
            offside_line: Offside line to draw
            vanishing_point: Vanishing point to draw and connect to keypoints
            last_kp: Last defender keypoint
            projected_point: Projected point of last defender
            projection_points: All projection points
            judgements: Optional list of offside judgements
            only_offside_attackers: If True, draw only attackers judged offside
            
        Returns:
            Frame with keypoints drawn
        """
        out = frame.copy()
        out_save = frame.copy()
        # if offside_line is not None:
        #     cv2.line(out, offside_line[0], offside_line[1], (0, 255, 0), 1, cv2.LINE_AA)
        if all_keypoints:
            for det in detections:
                all_kps = det.get("all_kps")
                if all_kps is None:
                    continue
                for x, y, c in all_kps:
                    if float(c) > 0.1:
                        cv2.circle(out, (int(x), int(y)), 2, KEYPOINT_COLOR, 3)
                        if vanishing_point is not None:
                            vp_x, vp_y = map(int, vanishing_point)
                            cv2.line(out, (int(x), int(y)), (vp_x, vp_y), (255, 255, 255), 1, cv2.LINE_AA)
        else:
            if judgements is None:
                det_items = [(det, None) for det in detections]
            else:
                det_items = list(zip(detections, judgements))
            if only_offside_attackers:
                det_items = [(det, j) for det, j in det_items if j == "OFFSIDE"]
            # for det, _ in det_items:
            #     for kp_name, (kx, ky) in det["keypoints"].items():
            #         cv2.circle(out, (int(kx), int(ky)), 4, KEYPOINT_COLOR, -1)
            for det, judgement in det_items:
                # draw ONLY offside players
                if judgement not in ["OFFSIDE", "CLOSEST"]:
                    continue

                if det.get("offside_keypoint") is not None:
                    kx, ky = det["offside_keypoint"]

                    # smaller keypoint
                    cv2.circle(
                        out,
                        (int(kx), int(ky)),
                        2,   # smaller size
                        OFFSIDE_COLOR,
                        3
                    )
                    # smaller keypoint
                    cv2.circle(
                        out_save,
                        (int(kx), int(ky)),
                        2,   # smaller size
                        OFFSIDE_COLOR,
                        3
                    )
        for det_idx, det in enumerate(detections):
            judgement = None
            if judgements is not None and det_idx < len(judgements):
                judgement = judgements[det_idx]
            if judgement not in ["OFFSIDE", "CLOSEST"]:
                continue
            if det.get("offside_keypoint") is not None:
                okx, oky = det["offside_keypoint"]

                # smaller point
                cv2.circle(
                    out,
                    (int(okx), int(oky)),
                    2,
                    OFFSIDE_COLOR,
                    3
                )
                cv2.putText(out, "Potential Offside", (int(okx) + 3, int(oky) - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, OFFSIDE_COLOR, 1)
                
                # smaller point
                cv2.circle(
                    out_save,
                    (int(okx), int(oky)),
                    2,
                    OFFSIDE_COLOR,
                    3
                )
                # cv2.putText(out_save, "Potential Offside", (int(okx) + 3, int(oky) - 35),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, OFFSIDE_COLOR, 1)
                # proj = det.get("offside_proj_point")
                # if proj is not None:
                #     px, py = int(proj[0]), int(proj[1])
                #     cv2.line(out, (int(okx), int(oky)), (px, py), (0, 255, 0), 2, cv2.LINE_AA)
                #     cv2.circle(out, (px, py), 6, (0, 165, 255), -1)
                #     cv2.putText(out, "proj", (px + 5, py - 5),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                proj = det.get("offside_proj_point")

                if proj is not None:
                    px, py = int(proj[0]), int(proj[1])

                    cv2.line(
                        out,
                        (int(okx), int(oky)),
                        (px, py),
                        OFFSIDE_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        out_save,
                        (int(okx), int(oky)),
                        (px, py),
                        OFFSIDE_COLOR,
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.circle(out, (px, py), 1, (0, 165, 255), 3)
                    H, W = out.shape[:2]
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

                        if clipped_line is not None:

                            cv2.line(
                                out,
                                clipped_line[0],
                                clipped_line[1],
                                OFFSIDE_COLOR,
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.line(
                                out_save,
                                clipped_line[0],
                                clipped_line[1],
                                OFFSIDE_COLOR,
                                1,
                                cv2.LINE_AA,
                            )

                        if clipped_lineh is not None:

                            cv2.line(
                                out,
                                clipped_lineh[0],
                                clipped_lineh[1],
                                OFFSIDE_COLOR,
                                1,
                                cv2.LINE_AA,
                            )
        if vanishing_point is not None and vanishing_point_horiz is not None and all_keypoints:
            vx, vy = map(int, vanishing_point)
            vhx, vhy = map(int, vanishing_point_horiz)
            cv2.circle(out, (vx, vy), 10, VP_COLOR, 3)
            cv2.circle(out_save, (vx, vy), 10, VP_COLOR, 3)
            cv2.circle(out, (vhx, vhy), 10, VP_COLOR, 3)
        if projection_points and not only_offside_attackers:
            for px, py in projection_points:
                cv2.circle(out, (px, py), 1, (255, 255, 255), 3)
                cv2.circle(out_save, (px, py), 1, (255, 255, 255), 3)
        if last_kp is not None:

            lx, ly = int(last_kp[0]), int(last_kp[1])

            cv2.circle(
                out,
                (lx, ly),
                2,
                LAST_DEF_KEYPOINT_COLOR,
                3
            )
            cv2.circle(
                out_save,
                (lx, ly),
                2,
                LAST_DEF_KEYPOINT_COLOR,
                3
            )
            cv2.putText(out, "Last Def", (lx + 5, ly - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, LAST_DEF_KEYPOINT_COLOR, 1)
            
        # if last_kp is not None and projected_point is not None:
        #     px, py = int(projected_point[0]), int(projected_point[1])
        #     cv2.line(out, (int(last_kp[0]), int(last_kp[1])), (px, py), (0, 255, 0), 2, cv2.LINE_AA)
        #     cv2.circle(out, (px, py), 3, OFFSIDE_COLOR, -1)
        #     cv2.putText(out, "proj", (px + 5, py - 5),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, OFFSIDE_COLOR, 1)
        if last_kp is not None and projected_point is not None:
            px, py = int(projected_point[0]), int(projected_point[1])
            vp_x, vp_y = map(int, vanishing_point)
            vph_x, vph_y = map(int, vanishing_point_horiz)
            H, W = out.shape[:2]
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

                if clipped_line is not None:

                    cv2.line(
                        out,
                        clipped_line[0],
                        clipped_line[1],
                        LAST_DEF_KEYPOINT_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        out_save,
                        clipped_line[0],
                        clipped_line[1],
                        LAST_DEF_KEYPOINT_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
                if clipped_lineh is not None:

                    cv2.line(
                        out,
                        clipped_lineh[0],
                        clipped_lineh[1],
                        LAST_DEF_KEYPOINT_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
                    
            cv2.line(out, (int(last_kp[0]), int(last_kp[1])), (px, py), LAST_DEF_KEYPOINT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(out, (int(last_kp[0]), int(last_kp[1])), 1, LAST_DEF_KEYPOINT_COLOR, -1)
            cv2.circle(out, (px, py), 1, LAST_DEF_KEYPOINT_COLOR, -1)
            cv2.line(out_save, (int(last_kp[0]), int(last_kp[1])), (px, py), LAST_DEF_KEYPOINT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(out_save, (int(last_kp[0]), int(last_kp[1])), 1, LAST_DEF_KEYPOINT_COLOR, -1)
            cv2.circle(out_save, (px, py), 1, LAST_DEF_KEYPOINT_COLOR, -1)

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
        if vanishing_point:
            vx, vy = int(vanishing_point[0]), int(vanishing_point[1])
            if 0 <= vx < W and 0 <= vy < H:
                cv2.circle(out, (vx, vy), 10, VP_COLOR, -1)
                cv2.circle(out, (vx, vy), 14, (0, 0, 0), 2)
        for det, lbl, judgement in zip(detections, team_labels, judgements):
            x1, y1, x2, y2 = det["bbox"]
            color = team_display_colors[lbl] if lbl >= 0 else (180, 180, 180)
            if judgement == "OFFSIDE":
                cv2.rectangle(out, (x1, y1), (x2, y2), OFFSIDE_COLOR, 2)
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
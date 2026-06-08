"""
Main Offside Detection Pipeline Orchestrator
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple


from .models import ModelManager, PoseEstimator, PlayerMasker, preprocess_rtdetr, preprocess_onnx
from .pitch_analyzer import PitchAnalyzer
from .team_classifier import TeamClassifier, AttackDirectionDetector
from .offside_detector import OffsideDetector
from .visualization import VisualizationRenderer
from .utils import ColorUtils


class OffsideDetectionPipeline:
    """Main orchestrator for offside detection."""
    
    def __init__(
        self,
        model_type: str = "vitpose",
        device: str = "cpu"
    ):
        """
        Initialize pipeline.
        
        Args:
            model_type: Type of pose model ('yolo', 'resnet', 'vitpose')
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        model_manager = ModelManager(model_type, device)
        self.pose_estimator = None
        self.model_type = model_type
        self.pitch_analyzer = PitchAnalyzer()
        self.pitch_segmentation_model = None
        self.model_manager = model_manager
        self.offside_detector = OffsideDetector()
        self.visualization = VisualizationRenderer()
    
    def get_pose_estimator(self):
        if self.pose_estimator is None:

            model = self.model_manager.load_models()

            self.pose_estimator = PoseEstimator(
                model,
                self.model_type
            )

        return self.pose_estimator

    def process_frame(
        self,
        frame: np.ndarray,
        teamA_color: Optional[Tuple] = None,
        teamB_color: Optional[Tuple] = None,
        forced_attacking_team: Optional[int] = None,
    ) -> dict:
        """
        Process single frame.
        
        Args:
            frame: Input frame
            teamA_color: Override color for team A
            teamB_color: Override color for team B
            forced_attacking_team: Force attacking team (0 or 1)
            
        Returns:
            Dict with processing results
        """
        H, W = frame.shape[:2]
        pitch_mask = self.generate_pitch_polygon_mask(frame)

        masked = cv2.bitwise_and(
            frame,
            frame,
            mask=pitch_mask
        )        
        pose_estimator = self.get_pose_estimator()

        detections = pose_estimator.estimate_pose(masked)
        player_mask = PlayerMasker.create_player_mask(detections, frame.shape)
        clean_frame = PlayerMasker.hide_players(masked, player_mask)
        pitch_vertical_lines = self.pitch_analyzer.line_detector.detect_vertical_pitch_lines(clean_frame)
        pitch_horizontal_lines = self.pitch_analyzer.line_detector.detect_horizontal_pitch_lines(clean_frame)
        vpv = self.pitch_analyzer.vp_estimator.compute_vanishing_point(
            pitch_vertical_lines, frame
        ) if pitch_vertical_lines else None
        vph = self.pitch_analyzer.vp_estimator.compute_vanishing_point(
            pitch_horizontal_lines, frame
        ) if pitch_horizontal_lines else None
        team_labels, c0, c1 = TeamClassifier.classify_teams(masked, detections)
        if teamA_color is not None:
            c0 = np.array(teamA_color, dtype=np.float32)
        if teamB_color is not None:
            c1 = np.array(teamB_color, dtype=np.float32)
        attack_info = AttackDirectionDetector.detect_attack_direction(
            detections, team_labels, W
        )
        if forced_attacking_team is not None:
            attack_info["attacking_team"] = forced_attacking_team
            attack_info["defending_team"] = 1 - forced_attacking_team
            def_team = attack_info["defending_team"]
            def_xs = [
                (det["bbox"][0] + det["bbox"][2]) / 2.0
                for det, lbl in zip(detections, team_labels)
                if lbl == def_team
            ]
            if def_xs:
                def_median = np.median(def_xs)
                attack_info["direction"] = "left" if def_median < W / 2 else "right"
        #! this the entire pipeline in step 0, do we want this behaviour ?
        result = self.offside_detector.compute_offside_status(
            detections, team_labels, attack_info, vpv, vph, (H, W)
        )
        offside_line, ground_line, all_def_lines, last_kp, projected_point, \
        projection_points, x_axis, judgements = result
        

        return {
            "frame": frame,
            "detections": detections,
            "team_labels": team_labels,
            "team_colors": (c0, c1),
            "pitch_vertical_lines": pitch_vertical_lines,
            "pitch_horizontal_lines": pitch_horizontal_lines,
            "vanishing_point": vpv,
            "vanishing_point_horiz": vph,
            "offside_line": offside_line,
            "ground_line": ground_line,
            "all_defender_lines": all_def_lines,
            "judgements": judgements,
            "attack_info": attack_info,
            "projection_points": projection_points,
            "x_axis": x_axis,
            "last_kp": last_kp,
            "projected_point": projected_point,
            "pitch_mask": pitch_mask,
        }
    
    def render_output(self, result: dict) -> np.ndarray:
        """Render main output image from processing result."""
        return self.visualization.render_main_output(
            result["frame"],
            result["detections"],
            result["team_labels"],
            result["team_colors"],
            result["pitch_lines"],
            result["vanishing_point"],
            result["vanishing_point_horiz"],
            result["offside_line"],
            result["ground_line"],
            result["all_defender_lines"],
            result["judgements"],
            result["attack_info"],
            result["projection_points"],
            result["x_axis"],
            result["last_kp"],
            result["projected_point"],
        )
    
    def generate_pitch_polygon_mask(self, frame):

        if self.pitch_segmentation_model is None:

            self.pitch_segmentation_model = (
                self.model_manager.load_pitch_segmentation_model()
            )

        input_tensor, orig_w, orig_h = preprocess_rtdetr(frame)

        outputs = self.pitch_segmentation_model.run(input_tensor)
        """
        #khadija comments
        if len(outputs) ==2:
                raw_logits = np.squeeze(outputs[0])
                pred_masks = np.squeeze(outputs[1])
                print("pred_masks shape:", pred_masks.shape)
                print("raw_logits shape:", raw_logits.shape)
        else:
                raw_logits = np.squeeze(outputs[1])
                pred_masks = np.squeeze(outputs[2])
        """

        raw_logits = np.squeeze(outputs[1])

        pred_masks = np.squeeze(outputs[2])

        pred_labels = np.argmax(raw_logits, axis=-1)

        TARGET_CLASS_ID = 3

        final_mask = np.zeros(
            (orig_h, orig_w),
            dtype=np.uint8
        )

        mask_threshold = 0.3

        for i, class_id in enumerate(pred_labels):

            if class_id != TARGET_CLASS_ID:
                continue


            m = pred_masks[i]
            """
            #khadija comments
            # Prevent crash when fewer masks than detections
            if i >= pred_masks.shape[0]:
                continue
            """
            m_binary = (
                m > mask_threshold
            ).astype(np.uint8)

            m_resized = cv2.resize(
                m_binary,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            final_mask = np.maximum(
                final_mask,
                m_resized
            )

        final_mask = (final_mask * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        return final_mask

    def render_keypoints_debug(
        self,
        frame: np.ndarray,
        detections: list,
        all_keypoints: bool = False,
        offside_line: Optional[Tuple] = None,
        vanishing_point: Optional[Tuple] = None,
        vanishing_point_horiz: Optional[Tuple] = None,
        last_kp: Optional[Tuple] = None,
        projected_point: Optional[Tuple] = None,
        projection_points: Optional[list] = None,
        judgements: Optional[list] = None,
        only_offside_attackers: bool = False,
        pitch_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render keypoints debug image."""
        return self.visualization.render_keypoints_image(
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
            pitch_mask=pitch_mask,
        )


class ImageProcessor:
    """Handles image input/output and file management."""
    
    def __init__(self, pipeline: OffsideDetectionPipeline):
        """
        Initialize image processor.
        
        Args:
            pipeline: Offside detection pipeline
        """
        self.pipeline = pipeline
    
    def process_image(
        self,
        image_path: str,
        output_dir: str = "output",
        save_debug: bool = True,
        teamA_color: Optional[Tuple] = None,
        teamB_color: Optional[Tuple] = None,
        forced_attacking_team: Optional[int] = None,
    ) -> str:
        """
        Process single image and save outputs.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            save_debug: Save debug visualizations
            teamA_color: Override color for team A
            teamB_color: Override color for team B
            forced_attacking_team: Force attacking team
            
        Returns:
            Path to output image
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        result = self.pipeline.process_frame(
            frame, teamA_color, teamB_color, forced_attacking_team
        )
        output_img = self.pipeline.render_output(result)
        fname = os.path.basename(image_path)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"offside_{fname}")
        cv2.imwrite(out_path, output_img)
        print(f"Saved → {out_path}")
        if save_debug:
            all_kp_vp_img = self.pipeline.render_keypoints_debug(
                frame,
                result["detections"],
                all_keypoints=True,
                offside_line=result["offside_line"],
                vanishing_point=result["vanishing_point"],
                last_kp=result["last_kp"],
                projected_point=result["projected_point"],
                projection_points=result["projection_points"],
            )
            all_kp_vp_path = out_path.replace("offside_", "keypoints_all_to_vp_", 1)
            cv2.imwrite(all_kp_vp_path, all_kp_vp_img)
            print(f"Saved → {all_kp_vp_path}")
            considered_proj_img = self.pipeline.render_keypoints_debug(
                frame,
                result["detections"],
                all_keypoints=False,
                offside_line=result["offside_line"],
                last_kp=result["last_kp"],
                projected_point=result["projected_point"],
                projection_points=result["projection_points"],
            )
            considered_proj_path = out_path.replace("offside_", "keypoints_offside_projection_", 1)
            cv2.imwrite(considered_proj_path, considered_proj_img)
            print(f"Saved → {considered_proj_path}")

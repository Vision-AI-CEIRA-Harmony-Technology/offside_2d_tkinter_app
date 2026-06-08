import sys
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageOps
import numpy as np

# Ensure root workspace is on sys.path when running this file directly.
# ROOT_DIR = Path(__file__).resolve().parents[1]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

from offside_logique.pipeline import OffsideDetectionPipeline
from offside_logique.team_classifier import TeamClassifier, AttackDirectionDetector
from ui import build_app_ui
from offside_logique.utils import GeometryUtils
from canvas_utils import (
    canvas_to_image,
    get_all_boxes,
    draw_handles,
    on_mouse_down,
    on_mouse_drag,
    on_mouse_up,
    on_right_click,
    on_mouse_scroll,
)


class OffsideApp:
    def __init__(self, root):
        self.root = root
        
        self.pipeline = None
        self.original_img = None
        self.tk_img = None
        self.logo_img = None
        self.final_render = None

        logo_path = Path(__file__).resolve().parent / "assets" / "logo2.png"

        if logo_path.exists():

            logo = Image.open(logo_path).convert("RGBA")

            self.logo_img = logo
        self.state = {
            "detections": [],
            "team_labels": [],
            "team_color_0": None,
            "team_color_1": None,
            "vp": None,
            "offside": None,
            "attack_info": None,
        }
        self.current_step = 0
        self.sidebar_visible = False
        self.selected_box = None
        self.drag_mode = None
        self.resize_corner = None
        self.drawing = False
        self.temp_box = None
        self.start_x = 0
        self.start_y = 0
        self.display_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.line_thickness = 1
        self.text_scale = 0.55
        self.text_thickness = 2
        self.handle_radius = 2
        self.point_radius = 2

        # Bind the canvas event handlers from helper utilities
        self.on_mouse_down = lambda e: on_mouse_down(self, e)
        self.on_mouse_drag = lambda e: on_mouse_drag(self, e)
        self.on_mouse_up = lambda e: on_mouse_up(self, e)
        self.on_right_click = lambda e: on_right_click(self, e)
        self.on_mouse_scroll = lambda e: on_mouse_scroll(self, e)

        self.canvas = build_app_ui(root, self)

        # Clustering 
        self.current_step = 0

        # ROI selection
        self.team_rois = {
            0: None,
            1: None,
        }
        self.video_team_rois = {
            0: None,
            1: None,
        }

        self.current_team_selection = 0
        self.roi_drawing = False
        self.roi_start = None
        self.temp_roi = None

        # Manual clustering priors
        self.team_centers = {
            0: None,
            1: None,
        }

        # Persist video-level team colors across frame selection
        self.video_team_centers = {
            0: None,
            1: None,
        }
        self.video_defending_team = 0
        self.video_left_team_id = None  # Team ID on left side of frame
        self.video_right_team_id = None  # Team ID on right side of frame
        self.left_team_id = None
        self.right_team_id = None
        self.video_left_team_id = None  # Team ID on left side of frame
        self.video_right_team_id = None  # Team ID on right side of frame
        self.left_team_id = None
        self.right_team_id = None

        self.selected_roi = None
        self.roi_drag_mode = None
        self.roi_resize_corner = None
        self.show_video_rois = True

        # First selected team is defending
        self.defending_team = 0

        # Manual VPvetical lines
        self.manual_vp_lines = []
        self.temp_vp_line = None
        self.vp_line_start = None

        # Manual VPhorizontal lines
        self.manual_vph_lines = []
        self.temp_vph_line = None
        self.vph_line_start = None

        # Manual lines for vpv drawing
        self.selected_vp_line = None
        self.dragging_vp_endpoint = None
        self.dragging_vp_line = False

        # Manual lines for vph drawing
        self.selected_vph_line = None
        self.dragging_vph_endpoint = None
        self.dragging_vph_line = False

        # Zoom state for step 6
        self.zoom_level = 1.0
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_crop_x1 = 0
        self.zoom_crop_y1 = 0

        self.final_render = None

        self.selected_bboxes = set()
        self.placing_offside_kp = False
        self.selected_kp = None
        self.dragging_kp = False
        self.manual_last_defender_kp = None
        self.manual_last_defender_proj = None
        # -----------------------------
        # VIDEO STATE
        # -----------------------------
        self.video_path = None
        self.video_capture = None
        self.video_total_frames = 0
        self.current_video_frame = 0
        self.current_frame_img = None
        self.video_playing = False
        self.video_fps = 30
        self.video_mode = False
        self.video_rois_locked_at_frame = None
        self.video_rois_locked_at_frame = None

    def toggle_sidebar(self):

        if self.sidebar_visible:

            self.sidebar.pack_forget()

            self.sidebar_handle.config(text=">>")

            self.sidebar_visible = False

        else:

            self.sidebar.pack(
                side="left",
                fill="y",
                padx=(15, 5),
                pady=10,
                before=self.workspace
            )

            self.sidebar_handle.config(text="<<")

            self.sidebar_visible = True
    def update_sidebar(self):
        if not hasattr(self, "left_team_label"):
            return
        self.half_label.config(
            text=f"Half : {self.mitemp_var.get()}"
        )
        # Left team
        if self.left_team_id is not None:
            self.left_team_label.config(
                text=f"Left Team : Team {self.left_team_id}"
            )

        # Right team
        if self.right_team_id is not None:
            self.right_team_label.config(
                text=f"Right Team : Team {self.right_team_id}"
            )

        # Team 0 color
        c0 = self.state.get("team_color_0")

        if c0 is not None:

            color = "#{:02x}{:02x}{:02x}".format(
                int(c0[2]),
                int(c0[1]),
                int(c0[0])
            )

            self.left_team_color.delete("all")

            self.left_team_color.create_rectangle(
                0,
                0,
                40,
                20,
                fill=color,
                outline=color
            )

        # Team 1 color
        c1 = self.state.get("team_color_1")

        if c1 is not None:

            color = "#{:02x}{:02x}{:02x}".format(
                int(c1[2]),
                int(c1[1]),
                int(c1[0])
            )

            self.right_team_color.delete("all")

            self.right_team_color.create_rectangle(
                0,
                0,
                40,
                20,
                fill=color,
                outline=color
            )

        attack_info = self.state.get("attack_info")

        if attack_info:

            self.attacking_team_label.config(
                text=f"Attacking Team : {attack_info['attacking_team']}"
            )

            self.defending_team_label.config(
                text=f"Defending Team : {attack_info['defending_team']}"
            )

    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv")
            ]
        )
        if not path:
            return
        self.video_path = path
        self.video_capture = cv2.VideoCapture(path)
        if not self.video_capture.isOpened():
            return
        self.video_total_frames = int(
            self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.current_video_frame = 0
        self.video_mode = True
        self.current_step = 0
        self.video_team_centers = {0: None, 1: None}
        self.video_defending_team = 0
        self.video_team_rois = {0: None, 1: None}
        self.video_left_team_id = None
        self.video_right_team_id = None
        self.state = {
            "detections": [],
            "team_labels": [],
            "team_color_0": None,
            "team_color_1": None,
            "vp": None,
            "offside": None,
            "pitch_mask": None,
            "attack_info": None,
        }
        self.selected_bboxes = set()
        self.team_rois = {0: None, 1: None}
        self.current_team_selection = 0
        self.roi_drawing = False
        self.roi_start = None
        self.temp_roi = None
        self.selected_box = None
        self.drag_mode = None
        self.resize_corner = None
        self.drawing = False
        self.temp_box = None
        self.selected_roi = None
        self.roi_drag_mode = None
        self.roi_resize_corner = None
        self.manual_vp_lines = []
        self.temp_vp_line = None
        self.vp_line_start = None
        self.manual_vph_lines = []
        self.temp_vph_line = None
        self.vph_line_start = None
        self.selected_vp_line = None
        self.dragging_vp_endpoint = None
        self.dragging_vp_line = False
        self.selected_vph_line = None
        self.dragging_vph_endpoint = None
        self.dragging_vph_line = False
        self.zoom_level = 1.0
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_crop_x1 = 0
        self.zoom_crop_y1 = 0
        self.current_frame_img = None
        self.video_playing = False
        self.team_centers = self.video_team_centers.copy()
        self.defending_team = self.video_defending_team
        self.read_current_frame()
        self.show_step()

    def read_current_frame(self):
        if self.video_capture is None:
            return
        self.video_capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            self.current_video_frame
        )
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame_img = frame

    def next_frame(self):
        if self.current_video_frame < self.video_total_frames - 1:
            self.current_video_frame += 1
            self.read_current_frame()
            self.show_step()


    def prev_frame(self):
        if self.current_video_frame > 0:
            self.current_video_frame -= 1
            self.read_current_frame()
            self.show_step()

    def toggle_play_video(self):
        self.video_playing = not self.video_playing
        if self.video_playing:
            self.play_video_loop()

    def play_video_loop(self):
        if not self.video_playing:
            return
        if self.current_video_frame >= self.video_total_frames - 1:
            self.video_playing = False
            return
        self.current_video_frame += 1
        self.read_current_frame()
        self.show_step()
        delay = int(1000 / max(1, self.video_fps))
        self.root.after(delay, self.play_video_loop)
    
    def choose_current_frame(self):
        if self.current_frame_img is None:
            return
        self.original_img = self.current_frame_img.copy()
        self.video_mode = False
        self.current_step = 1
        self.video_rois_locked_at_frame = None
        self.state = {
            "detections": [],
            "team_labels": [],
            "team_color_0": self.team_centers[0],
            "team_color_1": self.team_centers[1],
            "vp": None,
            "offside": None,
            "pitch_mask": None,
            "attack_info": None,
        }
        self.selected_bboxes = set()
        self.team_rois = self.video_team_rois.copy()
        self.current_team_selection = 2 if self.team_rois[0] is not None and self.team_rois[1] is not None else 0
        self.roi_drawing = False
        self.roi_start = None
        self.temp_roi = None
        self.selected_box = None
        self.drag_mode = None
        self.resize_corner = None
        self.drawing = False
        self.temp_box = None
        self.selected_roi = None
        self.roi_drag_mode = None
        self.roi_resize_corner = None
        self.manual_vp_lines = []
        self.temp_vp_line = None
        self.vp_line_start = None
        self.manual_vph_lines = []
        self.temp_vph_line = None
        self.vph_line_start = None
        self.selected_vp_line = None
        self.dragging_vp_endpoint = None
        self.dragging_vp_line = False
        self.selected_vph_line = None
        self.dragging_vph_endpoint = None
        self.dragging_vph_line = False
        self.zoom_level = 1.0
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_crop_x1 = 0
        self.zoom_crop_y1 = 0
        self.team_centers = self.video_team_centers.copy()
        self.defending_team = self.video_defending_team
        self.video_rois_locked_at_frame = None
        self.left_team_id = self.video_left_team_id
        self.right_team_id = self.video_right_team_id
        self._determine_left_right_teams()
        self.run_detection()
        self.show_step()
        self.manual_last_defender_kp = None
        self.manual_last_defender_proj = None
        self.selected_kp = None

    def delete_selected_roi(self):

        if self.selected_roi is None:
            return

        self.team_rois[self.selected_roi] = None
        self.team_centers[self.selected_roi] = None

        # allow re-selection
        self.current_team_selection = min(
            self.current_team_selection,
            self.selected_roi
        )

        self.selected_roi = None
        self.roi_drag_mode = None
        self.roi_resize_corner = None

        self.show_step()

    def get_pipeline(self):
        if self.pipeline is None:
            self.pipeline = OffsideDetectionPipeline(
                model_type="vitpose",
                device="cpu"
            )
        return self.pipeline

    def save_final_image(self):
        if self.final_render is None:
            return

        from pathlib import Path

        original_name = "image"

        if hasattr(self, "image_path") and self.image_path:

            original_name = Path(self.image_path).stem

        default_name = f"offside_{original_name}.png"

        path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg")
            ]
        )

        if not path:
            return

        cv2.imwrite(path, self.final_render)

    def load_image(self):
        path = filedialog.askopenfilename()
        self.image_path = path
        if not path:
            return
        self.original_img = cv2.imread(path)
        self.state = {
            "detections": [],
            "team_labels": [],
            "vp": None,
            "offside": None,
            "pitch_mask": None,
        }

        # RESET ROI SELECTION
        self.team_rois = {
            0: None,
            1: None,
        }

        self.team_centers = {
            0: None,
            1: None,
        }

        self.current_team_selection = 0

        self.roi_drawing = False
        self.roi_start = None
        self.temp_roi = None

        # RESET INTERACTION STATE
        self.selected_box = None
        self.drag_mode = None
        self.resize_corner = None
        self.drawing = False
        self.temp_box = None

        # RESET STEP
        self.current_step = 0

        self.selected_roi = None
        self.roi_drag_mode = None
        self.roi_resize_corner = None

        # Step 6 keypoint editing
        self.selected_kp = None
        self.dragging_kp = False
        self.manual_last_defender_kp = None
        self.manual_last_defender_proj = None

        # Manual VPvertical line editing
        self.manual_vp_lines = []
        self.temp_vp_line = None
        self.vp_line_start = None

        # Manual VPhorizontal line editing
        self.manual_vph_lines = []
        self.temp_vph_line = None
        self.vph_line_start = None

        # Manual vpv line editing
        self.selected_vp_line = None
        self.dragging_vp_endpoint = None
        self.dragging_vp_line = False

        # Manual vph line editing
        self.selected_vph_line = None
        self.dragging_vph_endpoint = None
        self.dragging_vph_line = False

        # Reset zoom
        self.zoom_level = 1.0
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_crop_x1 = 0
        self.zoom_crop_y1 = 0

        # Reset attack direction input
        if hasattr(self, "attack_direction_var"):
            self.attack_direction_var.set("")

        self.show_step()
        self.selected_bboxes = set()

    def run_detection(self):
        threading.Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        if self.original_img is None:
            return
        pipeline = self.get_pipeline()
        result = pipeline.process_frame(self.original_img)
        self.state["detections"] = result["detections"]
        self.state["vp"] = result["vanishing_point"]
        self.state["vph"] = result["vanishing_point_horiz"]
        self.state["pitch_mask"] = result["pitch_mask"]
        self.current_step = 1
        self.root.after(0, self.show_step)

    def run_pose_from_boxes(self):
        boxes = get_all_boxes(self)
        detections = []
        for (x1, y1, x2, y2) in boxes:
            crop = self.original_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pipeline = self.get_pipeline()
            pose_results = pipeline.pose_estimator._run_vitpose_inference(
                crop,
                pipeline.pose_estimator.model['pose_model']
            )
            if not pose_results:
                continue
            best = max(
                pose_results,
                key=lambda p: np.mean(p["scores"])
            )
            kps = best["keypoints"]
            scores = best["scores"]
            global_kps = []
            for (kx, ky), s in zip(kps, scores):
                global_kps.append((kx + x1, ky + y1, float(s)))
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "all_kps": global_kps,
                "keypoints": {
                    i: (kp[0], kp[1]) for i, kp in enumerate(global_kps) if kp[2] > 0.3
                }
            })
        self.state["detections"] = detections

    def run_offside(self):

        detections = self.state["detections"]

        # -----------------------------
        # TEAM CLASSIFICATION
        # -----------------------------
        labels, c0, c1 = TeamClassifier.classify_teams_with_priors(
            self.original_img,
            detections,
            self.team_centers[0],
            self.team_centers[1],
        )

        # -----------------------------
        # ATTACK DIRECTION
        # -----------------------------
        attack_info = AttackDirectionDetector.detect_attack_direction(
            detections,
            labels,
            self.original_img.shape[1]
        )

        manual_direction = None
        if hasattr(self, "attack_direction_var"):
            raw_direction = self.attack_direction_var.get().strip().lower()
            if raw_direction in ("left", "right"):
                manual_direction = raw_direction

        if manual_direction is not None:
            attack_info["direction"] = manual_direction
            attack_info = self._apply_mitemp_to_attack_info(
                attack_info
            )

        # first selected team = defending or swapped by mitemp
        self._apply_mitemp_to_attack_info(attack_info)

        # -----------------------------
        # OFFSIDE
        # -----------------------------
        pipeline = self.get_pipeline()
        result = pipeline.offside_detector.compute_offside_status(
            detections,
            labels,
            attack_info,
            self.state["vp"],
            self.state["vph"],
            self.original_img.shape[:2],
            manual_last_defender_kp=self.manual_last_defender_kp,
            manual_last_defender_proj=self.manual_last_defender_proj,
            
        )

        # -----------------------------
        # SAVE STATE
        # -----------------------------
        self.state["team_labels"] = labels
        self.state["team_color_0"] = c0
        self.state["team_color_1"] = c1
        self.state["attack_info"] = attack_info
        self.state["offside"] = result
        self.video_team_centers[0] = c0
        self.video_team_centers[1] = c1

    def recompute_offside(self):
        detections = self.state["detections"]

        pipeline = self.get_pipeline()
        attack_info = self._apply_mitemp_to_attack_info(self.state["attack_info"])
        attack_info = self._apply_mitemp_to_attack_info(
            self.state["attack_info"]
        )
        result = pipeline.offside_detector.compute_offside_status(
            detections,
            self.state["team_labels"],
            attack_info,
            self.state["vp"],
            self.state["vph"],
            self.original_img.shape[:2],
            manual_last_defender_kp=self.manual_last_defender_kp,
            manual_last_defender_proj=self.manual_last_defender_proj
        )

        self.state["offside"] = result

    def refresh_team_classification(self):
        frame = (
            self.original_img
            if self.original_img is not None
            else self.current_frame_img
        )

        for tid, roi in self.team_rois.items():
            if roi is None:
                continue

            color = TeamClassifier.extract_jersey_color(
                frame,
                roi
            )

            self.team_centers[tid] = color
            self.video_team_centers[tid] = color

        # Only recompute offside after a frame has been chosen
        if self.original_img is not None:
            self.run_offside()


    def _determine_left_right_teams(self):
        """Determine which team ID is on left/right based on ROI center X positions."""
        if self.team_rois[0] is None or self.team_rois[1] is None:
            return
        
        x1_0, y1_0, x2_0, y2_0 = self.team_rois[0]
        x1_1, y1_1, x2_1, y2_1 = self.team_rois[1]
        
        center_x_0 = (x1_0 + x2_0) / 2
        center_x_1 = (x1_1 + x2_1) / 2
        
        if center_x_0 < center_x_1:
            self.left_team_id = 0
            self.right_team_id = 1
        else:
            self.left_team_id = 1
            self.right_team_id = 0

    def _apply_mitemp_to_attack_info(self, attack_info):
        """Determine attacking/defending teams based on attack direction and mitemp.
        
        Logic:
        - direction=right, mitemp=1 (keep same) → attacking=left
        - direction=right, mitemp=2 (swap) → attacking=right
        - direction=left, mitemp=1 (keep same) → attacking=right
        - direction=left, mitemp=2 (swap) → attacking=left
        """
        if not hasattr(self, "mitemp_var") or not hasattr(self, "attack_direction_var"):
            return attack_info

        # Ensure left/right teams are determined
        if self.left_team_id is None or self.right_team_id is None:
            self._determine_left_right_teams()
            if self.left_team_id is None or self.right_team_id is None:
                return attack_info
        
        direction = self.attack_direction_var.get().strip().lower()
        mitemp = self.mitemp_var.get().strip()
        use_same_roles = mitemp == "1"
        
        # Determine attacking team based on direction and mitemp
        if direction == "right":
            attacking_team = self.left_team_id if use_same_roles else self.right_team_id
        elif direction == "left":
            attacking_team = self.right_team_id if use_same_roles else self.left_team_id
        else:
            # Fallback to old logic if direction is invalid
            use_same_roles = mitemp == "1"
            attacking_team = self.defending_team if use_same_roles else (1 - self.defending_team)
        
        defending_team = 1 - attacking_team
        
        attack_info["defending_team"] = defending_team
        attack_info["attacking_team"] = attacking_team

        return attack_info

    #! dikra: manual direction required??
    def _is_attack_direction_valid(self):
        if hasattr(self, "attack_direction_var"):
            raw_direction = self.attack_direction_var.get().strip().lower()
            return raw_direction in ("left", "right")
        return False

    def _line_angle(self, line):

        (x1, y1), (x2, y2) = line

        return abs(
            np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )
        )


    def _is_valid_vertical_line(self, line):

        angle = self._line_angle(line)

        return angle > 60


    def _is_valid_horizontal_line(self, line):

        angle = self._line_angle(line)

        return angle < 30

    def next_step(self):

        # STEP 0 -> RUN DETECTION -> STEP 1
        if self.current_step == 0:

            self.run_detection()
            return

        # STEP 1 -> STEP 2
        elif self.current_step == 1:
            # keep only selected detections
            if len(self.selected_bboxes) > 0:

                self.state["detections"] = [
                    det
                    for i, det in enumerate(self.state["detections"])
                    if i in self.selected_bboxes
                ]

            if self.team_rois[0] is not None and self.team_rois[1] is not None:
                self._determine_left_right_teams()
                self.current_step = 3
            else:
                self.current_step = 2

        # STEP 2 -> PROCESS -> STEP 3
        elif self.current_step == 2:

            # require both ROIs
            if self.team_rois[0] is None or self.team_rois[1] is None:
                return

            # recompute colors from final ROI positions
            for tid, roi in self.team_rois.items():

                color = TeamClassifier.extract_jersey_color(
                    self.original_img,
                    roi
                )

                self.team_centers[tid] = color

            # run processing once
            # self.run_pose_from_boxes()
            # self.run_offside()

            self.current_step = 3


        # STEP 3 -> VP SELECTION -> STEP 4
        elif self.current_step == 3:

            # use manual VP if user drew 2 lines
            if len(self.manual_vp_lines) == 2:

                l1 = self.manual_vp_lines[0]
                l2 = self.manual_vp_lines[1]

                vp = GeometryUtils.line_intersection(l1, l2)

                if vp is not None:
                    self.state["vp"] = vp

            # run processing now
            self.run_pose_from_boxes()
            self.run_offside()

            self.current_step = 4

        # STEP 4 > horizontal VP > STEP 5
        elif self.current_step == 4:

            # use manual VP if user drew 2 lines
            if len(self.manual_vph_lines) == 2:

                l1 = self.manual_vph_lines[0]
                l2 = self.manual_vph_lines[1]

                vph = GeometryUtils.line_intersection(l1, l2)

                if vph is not None:
                    self.state["vph"] = vph

            # run processing now
            self.run_pose_from_boxes()
            self.run_offside()

            self.current_step = 5

        # step 5 >> step 6
        elif self.current_step == 5:

            raw_direction = self.attack_direction_var.get().strip().lower()

            if raw_direction in ("left", "right"):
                self.state["attack_info"]["direction"] = raw_direction

            self._apply_mitemp_to_attack_info(self.state["attack_info"])
            #reset manual last def proj
            self.manual_last_defender_kp = None
            self.manual_last_defender_proj = None
            self.recompute_offside()
            # self.run_offside()

            self.current_step = 6

        # step 6 >> step 7 (preview saved image)
        elif self.current_step == 6:

            self.current_step = 7

        # final save
        else:

            self.save_final_image()
            return

        self.show_step()

    def prev_step(self):
        # Step 0 -> back to video selector
        if self.current_step == 0 and self.video_capture is not None:
            self.video_mode = True
            self.video_playing = False
            self.original_img = None
            self.show_step()
            return

        self.current_step = max(self.current_step - 1, 0)
        self.show_step()


    def back_to_video(self):
        if self.video_capture is None:
            return

        self.video_mode = True
        self.video_playing = False
        self.original_img = None
        self.show_video_rois = False
        self.zoom_level = 1.0
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_crop_x1 = 0
        self.zoom_crop_y1 = 0
        self.show_step()

    def display(self, img):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
        h, w = img.shape[:2]
        
        # Handle zoom for step 6
        zoom_level = getattr(self, "zoom_level", 1.0)
        if self.current_step == 6 and zoom_level > 1.0:
            zoom_cx = getattr(self, "zoom_center_x", None)
            zoom_cy = getattr(self, "zoom_center_y", None)
            if zoom_cx is None or zoom_cy is None:
                zoom_cx = w // 2
                zoom_cy = h // 2
                self.zoom_center_x = zoom_cx
                self.zoom_center_y = zoom_cy
            
            zoomed_w = int(canvas_w / zoom_level)
            zoomed_h = int(canvas_h / zoom_level)
            
            x1 = max(0, zoom_cx - zoomed_w // 2)
            y1 = max(0, zoom_cy - zoomed_h // 2)
            x2 = min(w, x1 + zoomed_w)
            y2 = min(h, y1 + zoomed_h)
            
            # Save crop coordinates for canvas_to_image conversion
            self.zoom_crop_x1 = x1
            self.zoom_crop_y1 = y1
            
            img_crop = img[y1:y2, x1:x2]
            scale = min(canvas_w / (x2 - x1), canvas_h / (y2 - y1))
        else:
            self.zoom_crop_x1 = 0
            self.zoom_crop_y1 = 0
            scale = min(canvas_w / w, canvas_h / h)
            img_crop = img
        
        self.display_scale = scale
        new_w = int(img_crop.shape[1] * scale)
        new_h = int(img_crop.shape[0] * scale)
        resized = cv2.resize(img_crop, (new_w, new_h))
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)

    def show_step(self):
        next_labels = {
            0: "Players ▶",
            1: "Lines ▶",
            2: "Lines ▶",
            3: "Lines ▶",
            4: "Direction ▶",
            5: "Projection ▶",
            6: "Preview ▶",
            7: "Save ▶"
        }

        prev_labels = {
            0: "◀ Back",
            1: "◀ Previous",
            2: "◀ Players",
            3: "◀ Teams",
            4: "◀ Lines",
            5: "◀ Lines",
            6: "◀ Direction",
            7: "◀ Projection"
        }

        if self.video_mode and self.current_frame_img is not None and self.original_img is None:
            if hasattr(self, "upload_btn"):
                self.upload_btn.pack_forget()
            if hasattr(self, "upload_video_btn"):
                self.upload_video_btn.pack_forget()
            if hasattr(self, "prev_btn"):
                self.prev_btn.grid_remove()
            if hasattr(self, "next_btn"):
                self.next_btn.grid_remove()
            if hasattr(self, "back_to_video_btn"):
                self.back_to_video_btn.pack_forget()
            if hasattr(self, "play_btn"):
                self.play_btn.pack(side="left", padx=5)
            if hasattr(self, "prev_frame_btn"):
                self.prev_frame_btn.pack(side="left", padx=5)
            if hasattr(self, "next_frame_btn"):
                self.next_frame_btn.pack(side="left", padx=5)
            if hasattr(self, "choose_frame_btn"):
                self.choose_frame_btn.pack(side="left", padx=5)
            if hasattr(self, "attack_direction_entry"):
                self.attack_direction_entry.pack_forget()
                self.mitemp_label.pack_forget()
                self.mitemp_entry.pack_forget()
            if hasattr(self, "attack_dir_label"):
                self.attack_dir_label.pack_forget()
            if hasattr(self, "mitemp_entry"):
                self.mitemp_entry.pack_forget()
            if hasattr(self, "mitemp_label"):
                self.mitemp_label.pack_forget()
            if hasattr(self, "add_keypoint_btn"):
                self.add_keypoint_btn.pack_forget()
            if hasattr(self, "delete_keypoint_btn"):
                self.delete_keypoint_btn.pack_forget()
            if hasattr(self, "step_label"):
                self.step_label.config(text="Step 0 • Video team color selection — start with right team")

            frame = self.current_frame_img.copy()
            frame_matches_locked = (self.video_rois_locked_at_frame is None or 
                                   self.current_video_frame == self.video_rois_locked_at_frame)
            if self.show_video_rois:
                for tid, roi in self.team_rois.items():
                    if roi is None:
                        continue
                    if not frame_matches_locked:
                        continue
                    x1, y1, x2, y2 = roi
                    color = tuple(int(c) for c in self.team_centers[tid]) if self.team_centers[tid] is not None else ((0, 0, 255) if tid == 0 else (0, 255, 0))
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                    draw_handles(frame, (x1, y1, x2, y2), self.handle_radius)
                    cv2.putText(frame, "Right Team" if tid == 0 else "Left Team",
                                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                self.text_scale, color, self.text_thickness, cv2.LINE_AA)
            if self.temp_roi is not None and frame_matches_locked:
                x1, y1, x2, y2 = self.temp_roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), self.line_thickness)

            if self.team_rois[0] is None or self.team_rois[1] is None:
                instruction = (
                    "Draw right team ROI first" if self.current_team_selection == 0 else
                    "Draw left team ROI next"
                )
            else:
                instruction = "Team colors selected. Choose frame and continue."

            cv2.putText(
                frame,
                instruction,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            self.update_sidebar()
            self.display(frame)
            return

        # hide normal workflow buttons
        if hasattr(self, "upload_btn"):
            self.upload_btn.pack_forget()

        if hasattr(self, "prev_btn"):
            self.prev_btn.grid_remove()

        if hasattr(self, "next_btn"):
            self.next_btn.grid_remove()

        if hasattr(self, "back_to_video_btn"):
            if self.video_capture is not None and self.original_img is not None:
                self.back_to_video_btn.pack(side="left", padx=5)
            else:
                self.back_to_video_btn.pack_forget()
        # ------------------------------------------------
        # SHOW LOGO SCREEN BEFORE IMAGE UPLOAD
        # ------------------------------------------------
        if hasattr(self, "play_btn"):
            self.play_btn.pack_forget()

        if hasattr(self, "prev_frame_btn"):
            self.prev_frame_btn.pack_forget()

        if hasattr(self, "next_frame_btn"):
            self.next_frame_btn.pack_forget()

        if hasattr(self, "choose_frame_btn"):
            self.choose_frame_btn.pack_forget()

        if self.original_img is None:

            # STEP 0 → only show upload video
            if hasattr(self, "upload_btn"):
                self.upload_btn.pack_forget()

            if hasattr(self, "upload_video_btn"):
                self.upload_video_btn.pack(side="left", padx=5, pady=10)

            if hasattr(self, "prev_btn"):
                self.prev_btn.grid_remove()

            if hasattr(self, "next_btn"):
                self.next_btn.grid_remove()

            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            if canvas_w < 10 or canvas_h < 10:
                return

            # dark background
            img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            img[:] = (11, 18, 32)

            # draw logo
            if self.logo_img is not None:

                logo = self.logo_img.copy()

                target_size = int(min(canvas_w, canvas_h) * 0.35)

                logo.thumbnail((target_size, target_size))

                logo_np = np.array(logo)

                lh, lw = logo_np.shape[:2]

                x = (canvas_w - lw) // 2
                y = (canvas_h - lh) // 2 - 40

                if logo_np.shape[2] == 4:

                    alpha = logo_np[:, :, 3] / 255.0

                    for c in range(3):

                        img[y:y+lh, x:x+lw, c] = (
                            alpha * logo_np[:, :, c] +
                            (1 - alpha) * img[y:y+lh, x:x+lw, c]
                        )

                else:

                    img[y:y+lh, x:x+lw] = logo_np

            # title
            cv2.putText(
                img,
                "AI OFFSIDE DETECTION",
                (canvas_w // 2 - 240, canvas_h // 2 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                img,
                "Upload an image to begin",
                (canvas_w // 2 - 170, canvas_h // 2 + 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
                cv2.LINE_AA
            )
            self.update_sidebar()

            self.display(img)

            return

        # normal workflow starts after frame selection
        if hasattr(self, "upload_video_btn"):
            self.upload_video_btn.pack_forget()
        if hasattr(self, "upload_btn"):
            self.upload_btn.pack_forget()
        if hasattr(self, "prev_btn"):
            self.prev_btn.grid()
        if hasattr(self, "next_btn"):
            self.next_btn.grid()
        img = self.original_img.copy()
        h, w = img.shape[:2]

        # Reference resolution = 1280x720
        ref_diag = (1280**2 + 720**2) ** 0.5
        img_diag = (w**2 + h**2) ** 0.5

        self.ui_scale = np.clip(
            img_diag / ref_diag,
            0.18,
            3.0
        )

        # reusable drawing sizes
        self.line_thickness = max(1, int(0.3 * self.ui_scale))
        self.small_thickness = max(1, int(1 * self.ui_scale))
        self.handle_radius = max(2, int(1.5 * self.ui_scale))
        self.point_radius = max(1, int(1.5 * self.ui_scale))
        self.text_scale = max(0.35, 0.55 * self.ui_scale)
        self.text_thickness = max(1, int(2 * self.ui_scale))
        if self.current_step == 1:
            for i, det in enumerate(self.state["detections"]):

                x1, y1, x2, y2 = det["bbox"]

                # RIGHT CLICK SELECTION = BLUE (priority)
                if i in self.selected_bboxes:
                    color = (255, 0, 0)

                # CURRENTLY EDITING = YELLOW
                elif self.selected_box == i:
                    color = (0, 255, 255)

                # NORMAL
                else:
                    color = (0, 255, 0)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, self.line_thickness)

                draw_handles(
                    img,
                    (x1, y1, x2, y2),
                    self.handle_radius
                )
            
            if self.temp_box:
                x1, y1, x2, y2 = self.temp_box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), self.line_thickness)

        elif self.current_step == 2:
            for tid, roi in self.team_rois.items():

                    if roi is None:
                        continue

                    x1, y1, x2, y2 = roi

                    color = tuple(int(c) for c in self.team_centers[tid]) #! dikra: color roi with team color

                    # transparent overlay
                    overlay = img.copy()

                    cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        color,
                        -1
                    )

                    img = cv2.addWeighted(
                        overlay,
                        0.2,
                        img,
                        0.8,
                        0
                    )

                    # border
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.line_thickness
                    )
                    draw_handles(
                        img,
                        (x1, y1, x2, y2),
                        self.handle_radius
                    )

                    label = (
                        "Right Team"
                        if tid == 0
                        else "Left Team"
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.text_scale,
                        color,
                        self.text_thickness,
                    )

            if self.temp_roi:
                x1, y1, x2, y2 = self.temp_roi

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    self.line_thickness,
                )


        elif self.current_step == 3:
            
            # draw existing lines
            for line in self.manual_vp_lines:

                (x1, y1), (x2, y2) = line

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    self.small_thickness,
                    cv2.LINE_AA
                )
                cv2.circle(img, (x1, y1), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (x2, y2), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)

            # draw temp line
            if self.temp_vp_line is not None:

                (x1, y1), (x2, y2) = self.temp_vp_line

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    self.small_thickness,
                    cv2.LINE_AA
                )
                cv2.circle(img, (x1, y1), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (x2, y2), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)

            # show computed VP preview
            if len(self.manual_vp_lines) == 2:

                vp = GeometryUtils.line_intersection(
                    self.manual_vp_lines[0],
                    self.manual_vp_lines[1]
                )

                if vp is not None:

                    vx, vy = map(int, vp)

                    cv2.circle(
                        img,
                        (vx, vy),
                        self.point_radius,
                        (0, 0, 255),
                        -1
                    )
        
        elif self.current_step == 4:
             # draw existing lines
            for line in self.manual_vph_lines:

                (x1, y1), (x2, y2) = line

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    self.small_thickness,
                    cv2.LINE_AA
                )
                cv2.circle(img, (x1, y1), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (x2, y2), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)

            # draw temp line
            if self.temp_vph_line is not None:

                (x1, y1), (x2, y2) = self.temp_vph_line

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    self.small_thickness,
                    cv2.LINE_AA
                )
                cv2.circle(img, (x1, y1), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (x2, y2), self.point_radius, (0, 255, 255), -1, cv2.LINE_AA)

            # show computed VP preview
            if len(self.manual_vph_lines) == 2:

                vph = GeometryUtils.line_intersection(
                    self.manual_vph_lines[0],
                    self.manual_vph_lines[1]
                )

                if vph is not None:

                    vx, vy = map(int, vph)

                    cv2.circle(
                        img,
                        (vx, vy),
                        self.point_radius,
                        (0, 0, 255),
                        -1
                    )

        elif self.current_step == 5:
            #! dikra: show detections colored with assigned team color
            for det, label in zip(self.state["detections"], self.state["team_labels"]):
                x1, y1, x2, y2 = det["bbox"]
                if label == 0: color = tuple(int(c) for c in self.state["team_color_0"])
                else: color = tuple(int(c) for c in self.state["team_color_1"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, self.line_thickness)

        elif self.current_step == 6:
            off = self.state["offside"]
            if off:
                pipeline = self.get_pipeline()
                img, img_save = pipeline.render_keypoints_debug(
                    img,
                    self.state["detections"],
                    all_keypoints=False,
                    offside_line=off[0],
                    vanishing_point=self.state["vp"],
                    vanishing_point_horiz=self.state["vph"],
                    last_kp=off[3],
                    projected_point=off[4],
                    projection_points=off[5],
                    judgements=off[7],
                    only_offside_attackers=True,
                    pitch_mask=self.state["pitch_mask"],
                )

                (
                    offside_line,
                    ground_line,
                    all_def_lines,
                    last_kp,
                    projected_point,
                    projection_points,
                    x_axis,
                    judgements
                ) = off

                # draw draggable attacking keypoints
                # for det in self.state["detections"]:

                #     kp = det.get("offside_keypoint")

                #     if kp is None:
                #         continue

                #     kx, ky = int(kp[0]), int(kp[1])

                    
            self.final_render = img_save.copy()
        elif self.current_step == 7:

            off = self.state["offside"]

            if off:

                pipeline = self.get_pipeline()

                img, img_save = pipeline.render_keypoints_debug(
                    img,
                    self.state["detections"],
                    all_keypoints=False,
                    offside_line=off[0],
                    vanishing_point=self.state["vp"],
                    vanishing_point_horiz=self.state["vph"],
                    last_kp=off[3],
                    projected_point=off[4],
                    projection_points=off[5],
                    judgements=off[7],
                    only_offside_attackers=True,
                    pitch_mask=self.state["pitch_mask"],
                )
                img = img_save.copy()
                self.final_render = img_save.copy()

        if self.current_step == 2:

            current_team_text = (
                " • Pick DEFENDING team player"
                if self.current_team_selection == 0
                else " • Pick ATTACKING team player"
            )

        else:

            current_team_text = ""

        step_text = {
            0: "Step 0 • choose video",
            1: "Step 1 • Edit player bounding boxes",
            2: f"Step 2 • Select one player from each team{current_team_text}",
            3: "Step 3 - Draw 2 horizontal parallel pitch lines (optional)",
            4: "Step 4 - Draw 2 vertical parallel pitch lines ",
            5: "Step 5 - Review team assignment",
            6: "Step 6 - Edit offside keypoints",
            7: "Step 7 - Final Visualisation - Save Result"
        }

        if hasattr(self, "step_label"):
            self.step_label.config(
                text=step_text.get(self.current_step, "")
            )
        if hasattr(self, "next_btn"):
            self.next_btn.config(
                text=next_labels.get(self.current_step, "Next ▶")
            )

        if hasattr(self, "prev_btn"):
            self.prev_btn.config(
                text=prev_labels.get(self.current_step, "◀ Previous")
            )
        # hide first
        if hasattr(self, "undo_btn"):
            self.undo_btn.pack_forget()

        if hasattr(self, "clear_btn"):
            self.clear_btn.pack_forget()

        if hasattr(self, "attack_direction_entry"):
            self.attack_direction_entry.pack_forget()
            self.mitemp_label.pack_forget()
            self.mitemp_entry.pack_forget()

        if hasattr(self, "attack_dir_label"):
            self.attack_dir_label.pack_forget()

        # if hasattr(self, "add_keypoint_btn"):
        #     self.add_keypoint_btn.pack_forget()

        # if hasattr(self, "delete_keypoint_btn"):
        #     self.delete_keypoint_btn.pack_forget()

        # show only during bbox editing
        if self.current_step == 1:

            self.undo_btn.pack(side="left", padx=5)
            self.clear_btn.pack(side="left", padx=5)

        #! dikra: show attack direction input during team assign review 
        if self.current_step == 5:
            self.attack_dir_label.pack(side="left", padx=(15, 5), pady=12)
            self.attack_direction_entry.pack(side="left", padx=5)
            self.mitemp_label.pack(
                side="left",
                padx=(15, 5),
                pady=12
            )

            self.mitemp_entry.pack(
                side="left",
                padx=5
            )
            if hasattr(self, "mitemp_label"):
                self.mitemp_label.pack(side="left", padx=(15, 5), pady=12)
            if hasattr(self, "mitemp_entry"):
                self.mitemp_entry.pack(side="left", padx=5)
        self.update_sidebar()

        self.display(img)

    def delete_selected_box(self):

        if self.selected_box is None:
            return

        del self.state["detections"][self.selected_box]

        self.selected_box = None

        self.show_step()

    #TODO: no backend logic for deleting kpts, to be removed
    def delete_selected_keypoint(self):

        if self.selected_kp is None:
            return

        kp_type, idx = self.selected_kp

        if kp_type == "attacker" and idx is not None:
            det = self.state["detections"][idx]
            det.pop("manual_offside_kp", None)
            det.pop("offside_keypoint", None)
            self.selected_kp = None
            self.dragging_kp = False
            self.recompute_offside()
            self.show_step()
            return

        if kp_type == "defender":
            self.manual_last_defender_kp = None
            self.selected_kp = None
            self.dragging_kp = False
            self.recompute_offside()
            self.show_step()
            return


    def undo_box(self):

        if self.state["detections"]:

            self.state["detections"].pop()

            self.show_step()

    def clear_boxes(self):
        self.state["detections"] = []

        self.show_step()


def main():
    import ctypes

    myappid = "offside.ai.app.1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    root = tk.Tk()
    # Hide window during initialization
    root.withdraw()
    icon_path = Path(__file__).resolve().parent / "assets" / "logo2.ico"
    root.iconbitmap(icon_path)

    app = OffsideApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
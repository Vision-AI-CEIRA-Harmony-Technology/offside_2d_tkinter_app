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

        self.current_team_selection = 0
        self.roi_drawing = False
        self.roi_start = None
        self.temp_roi = None

        # Manual clustering priors
        self.team_centers = {
            0: None,
            1: None,
        }

        self.selected_roi = None
        self.roi_drag_mode = None
        self.roi_resize_corner = None

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
        self.manual_last_defender_kp = None

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

        # first selected team = defending
        attack_info["defending_team"] = self.defending_team
        attack_info["attacking_team"] = 1 - self.defending_team

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
            manual_last_defender_kp=self.manual_last_defender_kp
        )

        # -----------------------------
        # SAVE STATE
        # -----------------------------
        self.state["team_labels"] = labels
        self.state["team_color_0"] = c0
        self.state["team_color_1"] = c1
        self.state["attack_info"] = attack_info
        self.state["offside"] = result

    def recompute_offside(self):

        detections = self.state["detections"]

        pipeline = self.get_pipeline()
        result = pipeline.offside_detector.compute_offside_status(
            detections,
            self.state["team_labels"],
            self.state["attack_info"],
            self.state["vp"],
            self.state["vph"],
            self.original_img.shape[:2],
            manual_last_defender_kp=self.manual_last_defender_kp
        )

        self.state["offside"] = result

    def refresh_team_classification(self):

        # recompute colors from updated ROIs
        for tid, roi in self.team_rois.items():

            if roi is None:
                continue

            color = TeamClassifier.extract_jersey_color(
                self.original_img,
                roi
            )

            self.team_centers[tid] = color

        # rerun offside logic
        self.run_offside()

    #! dikra: manual direction required??
    def _is_attack_direction_valid(self):
        if hasattr(self, "attack_direction_var"):
            raw_direction = self.attack_direction_var.get().strip().lower()
            return raw_direction in ("left", "right")
        return False

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

            self.recompute_offside()

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
        self.current_step = max(self.current_step - 1, 0)
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
            1: "Teams ▶",
            2: "Lines ▶",
            3: "Lines ▶",
            4: "Direction ▶",
            5: "Projection ▶",
            6: "Preview ▶",
            7: "Save ▶"
        }

        prev_labels = {
            0: "◀ Back",
            1: "◀ Upload",
            2: "◀ Players",
            3: "◀ Teams",
            4: "◀ Lines",
            5: "◀ Lines",
            6: "◀ Direction",
            7: "◀ Projection"
        }
        # ------------------------------------------------
        # SHOW LOGO SCREEN BEFORE IMAGE UPLOAD
        # ------------------------------------------------
        if self.original_img is None:

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

            self.display(img)

            return

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
        self.line_thickness = max(1, int(2 * self.ui_scale))
        self.small_thickness = max(1, int(1 * self.ui_scale))
        self.handle_radius = max(2, int(5 * self.ui_scale))
        self.point_radius = max(1, int(4 * self.ui_scale))
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
                    "Defending Team"
                    if tid == 0
                    else "Attacking Team"
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
            0: "Step 0 • Upload image",
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

        self.display(img)


    def delete_selected_box(self):

        if self.selected_box is None:
            return

        del self.state["detections"][self.selected_box]

        self.selected_box = None

        self.show_step()

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

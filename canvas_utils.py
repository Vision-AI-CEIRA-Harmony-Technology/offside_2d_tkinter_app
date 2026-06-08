from tkinter import messagebox

import cv2
from offside_logique.team_classifier import TeamClassifier
from offside_logique.utils import GeometryUtils


def canvas_to_image(app, x, y):
    # Adjust for zoom if enabled (step 6 only)
    zoom_level = getattr(app, "zoom_level", 1.0)
    if app.current_step == 6 and zoom_level > 1.0:
        crop_x1 = getattr(app, "zoom_crop_x1", 0)
        crop_y1 = getattr(app, "zoom_crop_y1", 0)
        # Convert canvas coords to cropped image coords, then add crop offset
        x_in_crop = (x - app.offset_x) / app.display_scale
        y_in_crop = (y - app.offset_y) / app.display_scale
        ix = int(x_in_crop + crop_x1)
        iy = int(y_in_crop + crop_y1)
        return ix, iy
    
    ix = (x - app.offset_x) / app.display_scale
    iy = (y - app.offset_y) / app.display_scale
    return int(ix), int(iy)

def on_right_click(app, e):

    # only bbox selection step
    if app.current_step != 1:
        return

    x, y = canvas_to_image(app, e.x, e.y)

    for i, det in enumerate(app.state["detections"]):

        if point_in_box(det["bbox"], x, y):

            # toggle selection
            if i in app.selected_bboxes:
                app.selected_bboxes.remove(i)
            else:
                app.selected_bboxes.add(i)

            app.show_step()
            return

def get_all_boxes(app):
    return [det["bbox"] for det in app.state["detections"]]


def point_in_box(box, x, y):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def detect_corner(box, x, y, thresh=10):
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for i, (cx, cy) in enumerate(corners):
        if abs(cx - x) < thresh and abs(cy - y) < thresh:
            return i
    return None

def point_near_point(p1, p2, thresh=10):

    return abs(p1[0] - p2[0]) < thresh and abs(p1[1] - p2[1]) < thresh

def point_near_line(pt, line, thresh=10):

    import numpy as np

    (x1, y1), (x2, y2) = line
    px, py = pt

    line_vec = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    pt_vec = np.array([px - x1, py - y1], dtype=np.float32)

    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return False

    proj = np.dot(pt_vec, line_vec) / line_len

    if proj < 0 or proj > line_len:
        return False

    closest = np.array([x1, y1]) + (proj / line_len) * line_vec

    dist = np.linalg.norm(np.array([px, py]) - closest)

    return dist < thresh


def get_box(app, ref):
    return app.state["detections"][ref]["bbox"]


def set_box(app, ref, box):
    app.state["detections"][ref]["bbox"] = box


def draw_handles(img, box, radius=5):

    x1, y1, x2, y2 = box

    for (x, y) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:

        cv2.circle(
            img,
            (x, y),
            radius,
            (255, 255, 255),
            -1
        )

def get_roi(app, idx):
    return app.team_rois[idx]


def set_roi(app, idx, roi):
    app.team_rois[idx] = roi
    if app.video_mode and app.original_img is None:
        app.video_team_rois[idx] = roi


def on_mouse_down(app, e):
    x, y = canvas_to_image(app, e.x, e.y)
    # -----------------------------
    # STEP 3 & 4: VP vert & horiz LINE DRAWING
    # -----------------------------
    if app.current_step == 3:

        # -----------------------------
        # EDIT EXISTING LINES
        # -----------------------------
        for idx, line in enumerate(app.manual_vp_lines):

            p1, p2 = line

            # endpoint 1
            if point_near_point((x, y), p1):

                app.selected_vp_line = idx
                app.dragging_vp_endpoint = 0
                return

            # endpoint 2
            if point_near_point((x, y), p2):

                app.selected_vp_line = idx
                app.dragging_vp_endpoint = 1
                return

            # move whole line
            if point_near_line((x, y), line):

                app.selected_vp_line = idx
                app.dragging_vp_line = True
                app.start_x = x
                app.start_y = y
                return

        # -----------------------------
        # CREATE NEW LINE
        # -----------------------------
        if len(app.manual_vp_lines) >= 2:
            return

        app.vp_line_start = (x, y)
        app.temp_vp_line = ((x, y), (x, y))

        return
    
    if app.current_step == 4:

    # -----------------------------
        # EDIT EXISTING LINES
        # -----------------------------
        for idx, line in enumerate(app.manual_vph_lines):

            p1, p2 = line

            # endpoint 1
            if point_near_point((x, y), p1):

                app.selected_vph_line = idx
                app.dragging_vph_endpoint = 0
                return

            # endpoint 2
            if point_near_point((x, y), p2):

                app.selected_vph_line = idx
                app.dragging_vph_endpoint = 1
                return

            # move whole line
            if point_near_line((x, y), line):

                app.selected_vph_line = idx
                app.dragging_vph_line = True
                app.start_x = x
                app.start_y = y
                return

        # -----------------------------
        # CREATE NEW LINE
        # -----------------------------
        if len(app.manual_vph_lines) >= 2:
            return

        app.vph_line_start = (x, y)
        app.temp_vph_line = ((x, y), (x, y))

        return

    # -----------------------------
    # STEP 6 : KEYPOINT EDITING
    # -----------------------------
    if app.current_step == 6:

        # if getattr(app, "placing_offside_kp", False):
        #     if app.place_offside_keypoint_at(x, y):
        #         return

        hit = detect_keypoint_hit(app, x, y)

        if hit is not None:
            app.selected_kp = hit
            app.dragging_kp = True

        return
    
    if app.current_step == 5:
        # Iterate through all detections to see if we clicked inside one
        for i, det in enumerate(app.state["detections"]):
            if point_in_box(det["bbox"], x, y):
                # Toggle the label: 0 becomes 1, 1 becomes 0
                current_label = app.state["team_labels"][i]
                app.state["team_labels"][i] = 1 - current_label

                #! bug here !            
                # Since the team changed, the offside line might move!
                app.recompute_offside()
                
                # Refresh the canvas to show the new color
                app.show_step()
                return # Exit after finding the first hit

    # ONLY STEP 1 & 2 BELOW, or video preview ROI selection before frame choice
    video_roi_mode = app.video_mode and app.current_step == 0 and app.original_img is None
    if app.current_step != 1 and app.current_step != 2 and not video_roi_mode:
        return
    if app.current_step == 2 or video_roi_mode:

        x, y = canvas_to_image(app, e.x, e.y)

        # EDIT EXISTING ROI
        for tid, roi in app.team_rois.items():

            if roi is None:
                continue

            corner = detect_corner(roi, x, y)

            if corner is not None:
                app.selected_roi = tid
                app.roi_drag_mode = "resize"
                app.roi_resize_corner = corner
                return

            if point_in_box(roi, x, y):
                app.selected_roi = tid
                app.roi_drag_mode = "move"
                app.start_x = x
                app.start_y = y
                return

        # CREATE NEW ROI
        if sum(roi is not None for roi in app.team_rois.values()) >= 2:
            return

        app.selected_roi = None

        app.roi_drawing = True
        app.roi_start = (x, y)
        app.temp_roi = (x, y, x, y)

        return
    
    ########################################
    x, y = canvas_to_image(app, e.x, e.y)
    for i, det in enumerate(app.state["detections"]):
        box = det["bbox"]
        corner = detect_corner(box, x, y)
        if corner is not None:
            app.selected_box = i
            app.drag_mode = "resize"
            app.resize_corner = corner
            return
        if point_in_box(box, x, y):
            app.selected_box = i
            app.drag_mode = "move"
            app.start_x, app.start_y = x, y
            return
    app.selected_box = None
    app.drawing = True
    app.start_x, app.start_y = x, y


def on_mouse_drag(app, e):
    x, y = canvas_to_image(app, e.x, e.y)
    # -----------------------------
    # STEP 3 : DRAW VP LINE
    # -----------------------------
    if app.current_step == 3:

        # -----------------------------
        # DRAG ENDPOINT
        # -----------------------------
        if app.selected_vp_line is not None and app.dragging_vp_endpoint is not None:

            p1, p2 = app.manual_vp_lines[app.selected_vp_line]

            if app.dragging_vp_endpoint == 0:
                p1 = (x, y)
            else:
                p2 = (x, y)

            app.manual_vp_lines[app.selected_vp_line] = (p1, p2)

            app.show_step()

            return

        # -----------------------------
        # MOVE WHOLE LINE
        # -----------------------------
        if app.dragging_vp_line:

            dx = x - app.start_x
            dy = y - app.start_y

            p1, p2 = app.manual_vp_lines[app.selected_vp_line]

            p1 = (p1[0] + dx, p1[1] + dy)
            p2 = (p2[0] + dx, p2[1] + dy)

            app.manual_vp_lines[app.selected_vp_line] = (p1, p2)

            app.start_x = x
            app.start_y = y

            app.show_step()

            return

        if app.vp_line_start is not None:
            app.temp_vp_line = (app.vp_line_start, (x, y))
            app.show_step()
            return

        # -----------------------------
        # DRAW NEW LINE
        # -----------------------------
        if app.vp_line_start is not None:

            app.temp_vp_line = (
                app.vp_line_start,
                (x, y)
            )

            app.show_step()

            return
        
    if app.current_step == 4:

        # -----------------------------
        # DRAG ENDPOINT
        # -----------------------------
        if app.selected_vph_line is not None and app.dragging_vph_endpoint is not None:
            p1, p2 = app.manual_vph_lines[app.selected_vph_line]
            if app.dragging_vph_endpoint == 0:
                p1 = (x, y)
            else:
                p2 = (x, y)
            app.manual_vph_lines[app.selected_vph_line] = (p1, p2)
            app.show_step()
            return

        # -----------------------------
        # MOVE WHOLE LINE
        # -----------------------------
        if app.dragging_vph_line:
            dx = x - app.start_x
            dy = y - app.start_y
            p1, p2 = app.manual_vph_lines[app.selected_vph_line]
            p1 = (p1[0] + dx, p1[1] + dy)
            p2 = (p2[0] + dx, p2[1] + dy)
            app.manual_vph_lines[app.selected_vph_line] = (p1, p2)
            app.start_x = x
            app.start_y = y
            app.show_step()
            return

        if app.vph_line_start is not None:
            app.temp_vph_line = (app.vph_line_start, (x, y))
            app.show_step()
            return

        # -----------------------------
        # DRAW NEW LINE
        # -----------------------------
        if app.vph_line_start is not None:

            app.temp_vph_line = (
                app.vph_line_start,
                (x, y)
            )

            app.show_step()

            return
        
    # -----------------------------
    # STEP 6 : DRAG KEYPOINT
    # -----------------------------
    if app.current_step == 6 and app.dragging_kp:

        kp_type, idx = app.selected_kp

        if kp_type == "attacker":
            attacking_team = app.state["attack_info"]["attacking_team"]
            in_attacker_bbox = False
            #! can we do it better than a loop ?
            for i, det in enumerate(app.state["detections"]):
                if app.state["team_labels"][i] != attacking_team:
                    continue  # skip defenders

                x1, y1, x2, y2 = det["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    in_attacker_bbox = True
                    break
            
            if not in_attacker_bbox:
                return
            
            target_det = app.state["detections"][idx]
            app.state["detections"][idx]["manual_offside_kp"] = (x, y)
            app.state["detections"][idx]["offside_keypoint"] = (x, y)
            # reset this attacker's manual proj point
            app.state["detections"][idx]["manual_offside_proj"] = None
            app.recompute_offside()

        elif kp_type == "attacker_projected":
            target_det = app.state["detections"][idx]
            orig_projection = target_det.get("offside_proj_point")
            fixed_x = int(orig_projection[0]) if orig_projection is not None else x
            app.state["detections"][idx]["offside_proj_point"] = (fixed_x, y)
            app.state["detections"][idx]["manual_offside_proj"] = (fixed_x, y) #! 
            app.recompute_offside()

            # result = app.state.get("offside")
            # if result is not None:
            #     offside_line, ground_line, all_def_lines, last_kp, projected_point, projection_points, x_axis, judgements = result
            #     if not any(j == "OFFSIDE" for j in judgements):
            #         vp = app.state.get("vp")
            #         if vp is not None:
            #             H, W = app.original_img.shape[:2]
            #             new_offside_line = GeometryUtils.extend_line_to_frame(vp, (x, y), W, H)
            #             app.state["offside"] = (
            #                 new_offside_line,
            #                 new_offside_line,
            #                 all_def_lines,
            #                 last_kp,
            #                 projected_point,
            #                 projection_points,
            #                 x_axis,
            #                 judgements,
            #             )
            

        elif kp_type == "defender":
            app.manual_last_defender_proj = None

            attacking_team = app.state["attack_info"]["attacking_team"]
            in_defender_bbox = False
            for i, det in enumerate(app.state["detections"]):
                if app.state["team_labels"][i] == attacking_team:
                    continue  # skip attackers
                x1, y1, x2, y2 = det["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    in_defender_bbox = True
                    break
            
            if not in_defender_bbox:
                return  # block drag outside defender bboxes
            
            app.manual_last_defender_kp = (x, y)
            app.recompute_offside()

        elif kp_type == "projected":
            
            # Move only the projected ground point and update offside line/ground line
            result = app.state.get("offside")
            if result is not None:
                offside_line, ground_line, all_def_lines, last_kp, projected_point, projection_points, x_axis, judgements = result
                fixed_x = int(projected_point[0]) if projected_point is not None else x
                new_projected = (fixed_x, y)
                app.manual_last_defender_proj = (fixed_x, y)
                vp = app.state.get("vp")
                if vp is not None:
                    H, W = app.original_img.shape[:2]
                    new_offside_line = GeometryUtils.extend_line_to_frame(vp, new_projected, W, H)
                    # new_ground_line = new_offside_line
                    app.state["offside"] = (
                        new_offside_line,
                        new_offside_line,
                        all_def_lines,
                        last_kp,
                        new_projected,
                        projection_points,
                        x_axis,
                        judgements,
                    )

            app.recompute_offside()

        app.show_step()
        return
    
    # ONLY STEP 1 & 2 BELOW
    video_roi_mode = (
        app.video_mode and
        app.current_step == 0 and
        app.original_img is None
    )

    # ONLY STEP 1, STEP 2, OR VIDEO ROI MODE
    if app.current_step != 1 and app.current_step != 2 and not video_roi_mode:
        return

    video_roi_mode = app.video_mode and app.current_step == 0 and app.original_img is None
    if app.current_step == 2 or video_roi_mode:

        x, y = canvas_to_image(app, e.x, e.y)

        # MOVE ROI
        if app.roi_drag_mode == "move" and app.selected_roi is not None:

            roi = get_roi(app, app.selected_roi)

            dx = x - app.start_x
            dy = y - app.start_y

            new_roi = (
                roi[0] + dx,
                roi[1] + dy,
                roi[2] + dx,
                roi[3] + dy,
            )

            set_roi(app, app.selected_roi, new_roi)
            # recompute clustering immediately
            app.refresh_team_classification()
            app.start_x = x
            app.start_y = y

            app.show_step()
            return

        # RESIZE ROI
        elif app.roi_drag_mode == "resize" and app.selected_roi is not None:

            x1, y1, x2, y2 = get_roi(app, app.selected_roi)

            if app.roi_resize_corner == 0:
                new_roi = (x, y, x2, y2)

            elif app.roi_resize_corner == 1:
                new_roi = (x1, y, x2, y2)

            elif app.roi_resize_corner == 2:
                new_roi = (x, y1, x2, y)

            else:
                new_roi = (x1, y1, x, y)

            set_roi(app, app.selected_roi, new_roi)
            # recompute clustering immediately
            app.refresh_team_classification()

            app.show_step()
            return

        # DRAW NEW ROI
        elif app.roi_drawing:

            x1, y1 = app.roi_start

            app.temp_roi = (
                x1,
                y1,
                x,
                y
            )

            app.show_step()
            return
    
    #################################
    x, y = canvas_to_image(app, e.x, e.y)
    if app.drag_mode == "move" and app.selected_box is not None:
        box = get_box(app, app.selected_box)
        dx = x - app.start_x
        dy = y - app.start_y
        new_box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        set_box(app, app.selected_box, new_box)
        app.start_x, app.start_y = x, y

    elif app.drag_mode == "resize" and app.selected_box is not None:
        x1, y1, x2, y2 = get_box(app, app.selected_box)
        # top-left
        if app.resize_corner == 0:

            new_box = (
                x,
                y,
                x2,
                y2
            )

        # top-right
        elif app.resize_corner == 1:

            new_box = (
                x1,
                y,
                x,
                y2
            )

        # bottom-left
        elif app.resize_corner == 2:

            new_box = (
                x,
                y1,
                x2,
                y
            )

        # bottom-right
        else:

            new_box = (
                x1,
                y1,
                x,
                y
            )
        set_box(app, app.selected_box, new_box)
        app.selected_bboxes.add(app.selected_box)
    elif app.drawing:
        app.temp_box = (
            app.start_x,
            app.start_y,
            x,
            y
        )
    app.show_step()


def detect_keypoint_hit(app, x, y, radius=10):
    """Detect if mouse clicked on an editable keypoint (attacker or defender)."""
    # ATTACKER OFFSIDE KEYPOINTS
    for i, det in enumerate(app.state["detections"]):
        kp = det.get("offside_keypoint")
        if kp is None:
            continue

        kx, ky = kp
        dist = ((kx - x) ** 2 + (ky - y) ** 2) ** 0.5

        if dist <= radius:
            return ("attacker", i)

    # ATTACKER PROJECTED KEYPOINTS
    for i, det in enumerate(app.state["detections"]):
        proj = det.get("offside_proj_point")
        if proj is None:
            continue

        px, py = proj
        dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if dist <= radius:
            return ("attacker_projected", i)


    # DEFENDER LAST PLAYER KEYPOINT
    result = app.state.get("offside")
    if result is not None:
        (
            offside_line,
            ground_line,
            all_def_lines,
            last_kp,
            projected_point,
            projection_points,
            x_axis,
            judgements
        ) = result

        if last_kp is not None:
            lx, ly = last_kp[0], last_kp[1]
            dist = ((lx - x) ** 2 + (ly - y) ** 2) ** 0.5
            if dist <= radius:
                return ("defender", None)
        # PROJECTED GROUND POINT (draggable independently — moves only projection & line)
        if projected_point is not None:
            px, py = projected_point[0], projected_point[1]
            distp = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if distp <= radius:
                return ("projected", None)

    return None

def on_mouse_scroll(app, e):
    """Handle mouse wheel zoom for step 6."""
    if app.current_step != 6:
        return
    
    # Determine zoom direction
    if e.num == 5 or e.delta < 0:  # Scroll down / zoom out
        zoom_delta = 0.9
    else:  # Scroll up / zoom in
        zoom_delta = 1.1
    
    x, y = canvas_to_image(app, e.x, e.y)
    
    old_zoom = app.zoom_level
    new_zoom = max(1.0, min(5.0, app.zoom_level * zoom_delta))
    
    if new_zoom != old_zoom:
        h, w = app.original_img.shape[:2]
        
        if app.zoom_center_x is None or app.zoom_center_y is None:
            app.zoom_center_x = w // 2
            app.zoom_center_y = h // 2
        
        # Adjust zoom center to keep mouse position stable
        app.zoom_center_x = int(app.zoom_center_x + (x - app.zoom_center_x) * (1 - old_zoom / new_zoom))
        app.zoom_center_y = int(app.zoom_center_y + (y - app.zoom_center_y) * (1 - old_zoom / new_zoom))
        
        # Clamp zoom center to image bounds
        app.zoom_center_x = max(0, min(w - 1, app.zoom_center_x))
        app.zoom_center_y = max(0, min(h - 1, app.zoom_center_y))
        
        app.zoom_level = new_zoom
        app.show_step()

def on_mouse_up(app, e):
    # -----------------------------
    # STEP 3 : FINALIZE VP LINE
    # -----------------------------
    #! dikra: same applies for vphorizontal and vpvertical (step 3&4)
    if app.current_step == 3 or app.current_step == 4:

        # finish new line
        if app.current_step == 3 and app.temp_vp_line is not None:
            if app.vp_line_start is not None and app.temp_vp_line is not None:
                    if not app._is_valid_vertical_line(app.temp_vp_line):
                        messagebox.showerror(
                            "Invalid line",
                            "Please draw a vertical pitch line."
                        )

                        return
                    app.manual_vp_lines.append(app.temp_vp_line)

        elif app.current_step == 4 and app.temp_vph_line is not None:
            if app.vph_line_start is not None and app.temp_vph_line is not None:    
                    if not app._is_valid_horizontal_line(app.temp_vph_line):
                        messagebox.showerror(
                            "Invalid line",
                            "Please draw a horizontal pitch line."
                        )

                        return    
                    app.manual_vph_lines.append(app.temp_vph_line)

        if app.current_step == 3 and app.temp_vp_line is not None:
        
            app.vp_line_start = None
            app.temp_vp_line = None
            # stop editing
            app.selected_vp_line = None
            app.dragging_vp_endpoint = None
            app.dragging_vp_line = False

        if app.current_step == 4 and app.temp_vph_line is not None:
        
            app.vph_line_start = None
            app.temp_vph_line = None
            # stop editing
            app.selected_vph_line = None
            app.dragging_vph_endpoint = None
            app.dragging_vph_line = False
            

        app.show_step()

        return

    if app.current_step == 3 or app.current_step == 4:
        #* why is these lines here?
        app.dragging_kp = False
        app.selected_kp = None
        return

    # FINISH KEYPOINT DRAG (STEP 6)
    if app.current_step == 6:
        app.dragging_kp = False
        # app.selected_kp = None
        app.show_step()
        return

    # ROI SELECTION STEP
    video_roi_mode = app.video_mode and app.current_step == 0 and app.original_img is None
    if (app.current_step == 2 or video_roi_mode) and app.roi_drawing:

        x, y = canvas_to_image(app, e.x, e.y)

        x1, y1 = app.roi_start

        roi = (
            min(x1, x),
            min(y1, y),
            max(x1, x),
            max(y1, y),
        )

        team_id = app.current_team_selection

        # save roi
        app.team_rois[team_id] = roi
        app.video_team_rois[team_id] = roi

        # extract dominant jersey color
        frame = app.original_img if app.original_img is not None else app.current_frame_img
        color = TeamClassifier.extract_jersey_color(
            frame,
            roi
        )

        app.team_centers[team_id] = color
        app.video_team_centers[team_id] = color

        # Lock ROI to current frame if in video preview mode
        if app.video_mode and app.original_img is None:
            app.video_rois_locked_at_frame = app.current_video_frame

        # first selected team = defending
        if team_id == 0:
            app.defending_team = 0

        app.current_team_selection += 1

        app.roi_drawing = False
        app.temp_roi = None

        # BOTH TEAMS SELECTED
        if app.current_team_selection > 1:
            app.current_team_selection = 2
            # Update video_left_team_id and video_right_team_id based on ROI positions
            if app.video_mode and app.original_img is None:
                app._determine_left_right_teams()
                app.video_left_team_id = app.left_team_id
                app.video_right_team_id = app.right_team_id

        # refresh ui
        app.show_step()
        return

    # finalize interactive ROI move/resize
    if (app.current_step == 2 or video_roi_mode) and app.roi_drag_mode is not None:
        if app.selected_roi is not None:
            if app.video_mode and app.original_img is None:
                app.video_team_rois[app.selected_roi] = app.team_rois[app.selected_roi]
        app.roi_drag_mode = None
        app.roi_resize_corner = None
        app.selected_roi = None
        app.show_step()
        return

    # NORMAL BBOX DRAWING STEP
    if app.current_step != 1:
        return

    if app.drawing:
        x, y = canvas_to_image(app, e.x, e.y)

        x1, y1 = app.start_x, app.start_y

        box = (
            min(x1, x),
            min(y1, y),
            max(x1, x),
            max(y1, y)
        )

        app.state["detections"].append({
            "bbox": box
        })
        app.selected_bboxes.add(len(app.state["detections"]) - 1)

        app.drawing = False
        app.temp_box = None

    app.show_step()

    app.drawing = False
    app.drag_mode = None
    app.temp_box = None
    app.show_step()

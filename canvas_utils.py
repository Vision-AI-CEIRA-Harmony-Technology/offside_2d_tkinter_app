import cv2
from offside_logique.team_classifier import TeamClassifier


def canvas_to_image(app, x, y):
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


def draw_handles(img, box):
    x1, y1, x2, y2 = box
    for (x, y) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(img, (x, y), 5, (255, 255, 255), -5)

def get_roi(app, idx):
    return app.team_rois[idx]


def set_roi(app, idx, roi):
    app.team_rois[idx] = roi


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

        if getattr(app, "placing_offside_kp", False):
            if app.place_offside_keypoint_at(x, y):
                return

        hit = detect_keypoint_hit(app, x, y)

        if hit is not None:
            app.selected_kp = hit
            app.dragging_kp = True

        return
    
    #!dikra: STEP5: assignment review, click bbox to change color
    if app.current_step == 5:
        # Iterate through all detections to see if we clicked inside one
        for i, det in enumerate(app.state["detections"]):
            if point_in_box(det["bbox"], x, y):
                # Toggle the label: 0 becomes 1, 1 becomes 0
                current_label = app.state["team_labels"][i]
                app.state["team_labels"][i] = 1 - current_label
                
                # Since the team changed, the offside line might move!
                app.recompute_offside()
                
                # Refresh the canvas to show the new color
                app.show_step()
                return # Exit after finding the first hit

    # ONLY STEP 1 & 2 BELOW
    if app.current_step != 1 and app.current_step != 2:
        return
    if app.current_step == 2:

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
        if app.selected_vp_line is not None and app.dragging_vp_endpoint is not None:
            p1, p2 = app.manual_vp_lines[app.selected_vp_line]
            if app.dragging_vp_endpoint == 0:
                p1 = (x, y)
            else:
                p2 = (x, y)
            app.manual_vp_lines[app.selected_vp_line] = (p1, p2)
            app.show_step()
            return

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

            if app.dragging_vp_endpoint == 0:
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
        if app.selected_vph_line is not None and app.dragging_vph_endpoint is not None:
            p1, p2 = app.manual_vph_lines[app.selected_vph_line]
            if app.dragging_vph_endpoint == 0:
                p1 = (x, y)
            else:
                p2 = (x, y)
            app.manual_vph_lines[app.selected_vph_line] = (p1, p2)
            app.show_step()
            return

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
            # Find which player the mouse is currently over
            target_idx = idx
            for i, det in enumerate(app.state["detections"]):
                x1, y1, x2, y2 = det["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    target_idx = i
                    break

            # Clear manual keypoint from ALL detections
            for det in app.state["detections"]:
                det.pop("manual_offside_kp", None)
                det.pop("offside_keypoint", None)

            # Assign new keypoint to target player
            target_det = app.state["detections"][target_idx]
            target_det["manual_offside_kp"] = (x, y)
            target_det["offside_keypoint"] = (x, y)

            # Update selection reference
            app.selected_kp = ("attacker", target_idx)

            # Force this player to be the offside candidate
            result = app.state.get("offside")
            if result is not None:
                judgements = result[-1]
                for j in range(len(judgements)):
                    judgements[j] = "ONSIDE"
                if target_idx < len(judgements):
                    judgements[target_idx] = "OFFSIDE"

            app.recompute_offside()

        elif kp_type == "defender":
            app.manual_last_defender_kp = (x, y)
            app.recompute_offside()

        app.show_step()
        return
    
    # ONLY STEP 1 & 2 BELOW
    if app.current_step != 1 and app.current_step != 2:
        return

    if app.current_step == 2:

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


def detect_keypoint_hit(app, x, y, radius=20):
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

    return None

def on_mouse_up(app, e):
    # -----------------------------
    # STEP 3 : FINALIZE VP LINE
    # -----------------------------
    #! dikra: same applies for vphorizontal and vpvertical (step 3&4)
    if app.current_step == 3 or app.current_step == 4:

        # finish new line
        if app.current_step == 3 and app.temp_vp_line is not None:
            if app.vp_line_start is not None and app.temp_vp_line is not None:
                    app.manual_vp_lines.append(app.temp_vp_line)
        elif app.current_step == 4 and app.temp_vph_line is not None:
            if app.vph_line_start is not None and app.temp_vph_line is not None:        
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

        app.dragging_kp = False
        app.selected_kp = None
        return

    # ROI SELECTION STEP
    if app.current_step == 2 and app.roi_drawing:

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

        # extract dominant jersey color
        color = TeamClassifier.extract_jersey_color(
            app.original_img,
            roi
        )

        app.team_centers[team_id] = color

        # first selected team = defending
        if team_id == 0:
            app.defending_team = 0

        app.current_team_selection += 1

        app.roi_drawing = False
        app.temp_roi = None

        # BOTH TEAMS SELECTED
        if app.current_team_selection > 1:
            app.current_team_selection = 2

        # refresh ui
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

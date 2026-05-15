"""
Configuration and Constants for Offside Detection System
"""

import numpy as np

# ─────────────────────────────────────────────
# DETECTION CONSTANTS
# ─────────────────────────────────────────────

# COCO keypoint indices relevant to offside rule
OFFSIDE_KP_IDX = {
    # head
    "nose": 0,
    "left_ear": 3,
    "right_ear": 4,

    # shoulders
    "left_shoulder": 5,
    "right_shoulder": 6,

    # hips
    "left_hip": 11,
    "right_hip": 12,

    # knees
    "left_knee": 13,
    "right_knee": 14,

    # ankles
    "left_ankle": 15,
    "right_ankle": 16,
}

# Minimum players to consider skipping goalkeeper
GK_SKIP_THRESHOLD = 3

# Confidence thresholds
POSE_CONF_THRESH = 0.3
KEYPOINT_CONF_THRESH = 0.5
IoU_THRESHOLD = 0.5

# Field segmentation parameters
GREEN_HSV_LOWER = np.array([30, 40, 40])
GREEN_HSV_UPPER = np.array([90, 255, 255])

# Color classification parameters
JERSEY_SAMPLE_TOP = 0.15  # Upper bound of jersey region (fraction of bbox height)
JERSEY_SAMPLE_BOTTOM = 0.55  # Lower bound of jersey region
JERSEY_SAMPLE_LEFT = 0.2  # Left margin
JERSEY_SAMPLE_RIGHT = 0.2  # Right margin

# Vanishing point parameters
VP_N_ITERATIONS = 700
VP_DISTANCE_THRESHOLD = 20.0
VP_EARLY_STOP_RATIO = 0.8
VP_MIN_DISTANCE_FROM_CENTER = 0.6  # As fraction of width

# Line detection parameters
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LENGTH = 60
HOUGH_MAX_GAP = 20
WHITE_LINE_THRESHOLD = 30

# Visualization colors
TEAM_COLORS = [
    (119, 57, 44),      # team 0 – blue-ish
    (30, 55, 43),    # team 1 – red-ish
]
OFFSIDE_COLOR = (0, 0, 255)  # Red for offside
ONSIDE_COLOR = (0, 200, 0)   # Green for onside
KEYPOINT_COLOR = (255, 255, 0)  # Cyan for keypoints
VP_COLOR = (0, 255, 255)  # Yellow for vanishing point
LAST_DEF_KEYPOINT_COLOR = (255, 0, 0)  # Blue for keypoints

# ViTPose parameters
VITPOSE_MIN_CROP_SIZE = 100
VITPOSE_CROP_SCALE_FACTOR = 2
VITPOSE_CROP_PADDING = 10

from __future__ import annotations

import math
from dataclasses import dataclass

APP_NAME = "CVML Lab 03 - Image Matching"
ORGANIZATION_NAME = "Sharif CVML Lab"

# ---------------------------------------------------------------------------
# Scene input
# ---------------------------------------------------------------------------
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
TARGET_FPS = 30
DEFAULT_CAMERA_INDEX = 0
DEFAULT_THEME = "dark"

# Images / video frames larger than this (longest side, in pixels) are
# downscaled before SIFT so that the pure-Python matching stays usable.
MAX_PROCESS_DIMENSION = 960

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm")

# ---------------------------------------------------------------------------
# Algorithm controls (the only two student-facing parameters)
# ---------------------------------------------------------------------------
DEFAULT_RATIO_THRESHOLD = 0.75
RATIO_THRESHOLD_MIN = 0.10
RATIO_THRESHOLD_MAX = 0.99
RATIO_SLIDER_SCALE = 100  # slider integer / 100 = ratio threshold

# The student's matching loop costs about N_reference * N_frame * 128 inner
# iterations, so the defaults are deliberately modest.
DEFAULT_MAX_KEYPOINTS = 100
MAX_KEYPOINTS_MIN = 20
MAX_KEYPOINTS_MAX = 500

DEFAULT_SHOW_KEYPOINTS = True

# ---------------------------------------------------------------------------
# Fixed geometric post-processing (NOT student-facing, NOT exposed in the GUI)
# ---------------------------------------------------------------------------
RANSAC_REPROJECTION_THRESHOLD = 5.0  # pixels
MIN_HOMOGRAPHY_MATCHES = 4  # cv2.findHomography needs at least 4 pairs
MIN_RANSAC_INLIERS = 12  # inliers required before localization is reported
QUAD_MIN_AREA_RATIO = 0.005  # quadrilateral area / scene area
QUAD_MAX_AREA_RATIO = 4.0
QUAD_MIN_SIDE_PIXELS = 8.0

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
PANEL_HEIGHT = 480  # height of each panel inside the combined canvas
PANEL_MAX_WIDTH = 800
PANEL_GAP = 12
MAX_DRAWN_MATCHES = 150  # display limit only; the match list is never truncated

COLOR_BACKGROUND = (17, 13, 10)  # BGR
COLOR_KEYPOINT = (170, 170, 170)
COLOR_MATCH_INLIER = (120, 220, 90)
COLOR_MATCH_OUTLIER = (0, 150, 255)
COLOR_MATCH_PLAIN = (255, 200, 60)
COLOR_QUAD = (90, 255, 140)
COLOR_BADGE_OK = (70, 170, 90)
COLOR_BADGE_OFF = (90, 90, 100)

# ---------------------------------------------------------------------------
# Keyboard shortcuts
# ---------------------------------------------------------------------------
SHORTCUTS = {
    "pause": "Space",
    "fullscreen": "F11",
    "reset": "R",
    "quit": "Q",
    "capture": "C",
}


@dataclass(frozen=True)
class ProcessingParameters:
    ratio_threshold: float
    max_keypoints: int
    show_keypoints: bool


@dataclass(frozen=True)
class FrameResult:
    """Metrics that accompany every rendered canvas."""

    reference_loaded: bool = False
    reference_keypoints: int = 0
    frame_keypoints: int = 0
    accepted_matches: int = 0
    inliers: int = -1  # -1 = geometry was not evaluated
    localization: str = "-"  # "DETECTED", "NOT LOCALIZED" or "-"
    match_ms: float = 0.0
    processing_ms: float = 0.0
    fps: float = 0.0  # 0.0 = not a running live source
    status: str = ""


def default_parameters() -> ProcessingParameters:
    return ProcessingParameters(
        ratio_threshold=DEFAULT_RATIO_THRESHOLD,
        max_keypoints=DEFAULT_MAX_KEYPOINTS,
        show_keypoints=DEFAULT_SHOW_KEYPOINTS,
    )


def validated_parameters(
    ratio_threshold, max_keypoints, show_keypoints=DEFAULT_SHOW_KEYPOINTS
) -> ProcessingParameters:
    """Clamp arbitrary inputs to safe values; fall back to defaults if invalid."""
    try:
        ratio = float(ratio_threshold)
        if not math.isfinite(ratio):
            raise ValueError
    except (TypeError, ValueError):
        ratio = DEFAULT_RATIO_THRESHOLD
    ratio = min(max(ratio, RATIO_THRESHOLD_MIN), RATIO_THRESHOLD_MAX)

    try:
        keypoints = int(max_keypoints)
    except (TypeError, ValueError):
        keypoints = DEFAULT_MAX_KEYPOINTS
    keypoints = min(max(keypoints, MAX_KEYPOINTS_MIN), MAX_KEYPOINTS_MAX)

    return ProcessingParameters(ratio, keypoints, bool(show_keypoints))

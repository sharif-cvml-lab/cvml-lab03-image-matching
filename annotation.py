from __future__ import annotations

import cv2
import numpy as np

from config import (
    COLOR_BACKGROUND,
    COLOR_BADGE_OFF,
    COLOR_BADGE_OK,
    COLOR_KEYPOINT,
    COLOR_MATCH_INLIER,
    COLOR_MATCH_OUTLIER,
    COLOR_MATCH_PLAIN,
    COLOR_QUAD,
    MAX_DRAWN_MATCHES,
    PANEL_GAP,
    PANEL_HEIGHT,
    PANEL_MAX_WIDTH,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_placeholder(width: int, height: int, lines: list[str]) -> np.ndarray:
    """Dark panel with centered text, used when no image is available."""
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 66, 76), 1)
    y = height // 2 - 12 * (len(lines) - 1)
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, FONT, 0.7, 1)
        cv2.putText(
            panel, line, (max(8, (width - text_w) // 2), y), FONT, 0.7,
            (190, 198, 208), 1, cv2.LINE_AA,
        )
        y += 28
    return panel


def _fit_panel(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale an image to PANEL_HEIGHT (or PANEL_MAX_WIDTH if it is very wide)."""
    height, width = image.shape[:2]
    scale = min(PANEL_HEIGHT / height, PANEL_MAX_WIDTH / width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation), scale


def _map_points(points: np.ndarray, scale: float, x0: int, y0: int) -> np.ndarray:
    """Image coordinates -> canvas coordinates (integer pixels)."""
    mapped = np.asarray(points, dtype=np.float32).reshape(-1, 2) * scale
    mapped[:, 0] += x0
    mapped[:, 1] += y0
    return np.rint(mapped).astype(np.int32)


def _put_label(canvas, text, origin, color=(235, 235, 235), bg=(25, 25, 30), scale=0.5):
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    cv2.rectangle(
        canvas, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 2), bg, -1
    )
    cv2.putText(canvas, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        return indices[:0]
    if len(indices) <= limit:
        return indices
    picks = np.linspace(0, len(indices) - 1, limit).astype(int)
    return indices[picks]


def render_match_view(
    reference_image,
    scene_image: np.ndarray,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    match_reference_points: np.ndarray,
    match_frame_points: np.ndarray,
    inlier_mask,
    quad,
    localized: bool,
    show_keypoints: bool,
) -> np.ndarray:
    """Build one canvas: reference (left) | current scene (right).

    All point arguments are in the pixel coordinates of the images that were
    given to SIFT; they are mapped to canvas coordinates here.
    inlier_mask is a bool array aligned with the match points, or None when the
    geometry stage was not evaluated. quad is a 4x2 array or None.
    """
    scene_panel, scene_scale = _fit_panel(scene_image)
    if reference_image is None:
        reference_panel = make_placeholder(
            scene_panel.shape[1], scene_panel.shape[0], ["No reference image selected"]
        )
        reference_scale = 1.0
    else:
        reference_panel, reference_scale = _fit_panel(reference_image)

    canvas_h = max(reference_panel.shape[0], scene_panel.shape[0])
    ref_x0 = 0
    ref_y0 = (canvas_h - reference_panel.shape[0]) // 2
    scene_x0 = reference_panel.shape[1] + PANEL_GAP
    scene_y0 = (canvas_h - scene_panel.shape[0]) // 2
    canvas_w = scene_x0 + scene_panel.shape[1]

    canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = COLOR_BACKGROUND
    canvas[
        ref_y0 : ref_y0 + reference_panel.shape[0],
        ref_x0 : ref_x0 + reference_panel.shape[1],
    ] = reference_panel
    canvas[
        scene_y0 : scene_y0 + scene_panel.shape[0],
        scene_x0 : scene_x0 + scene_panel.shape[1],
    ] = scene_panel

    # Translucent quadrilateral fill (only when localization was accepted).
    scene_quad = None
    if localized and quad is not None:
        scene_quad = _map_points(quad, scene_scale, scene_x0, scene_y0)
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, scene_quad, COLOR_QUAD)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, dst=canvas)

    # Keypoints (small dots).
    if show_keypoints:
        if reference_image is not None:
            for x, y in _map_points(reference_points, reference_scale, ref_x0, ref_y0):
                cv2.circle(canvas, (int(x), int(y)), 2, COLOR_KEYPOINT, -1, cv2.LINE_AA)
        for x, y in _map_points(frame_points, scene_scale, scene_x0, scene_y0):
            cv2.circle(canvas, (int(x), int(y)), 2, COLOR_KEYPOINT, -1, cv2.LINE_AA)

    # Match lines (display limit only; the real match list is untouched).
    total = len(match_reference_points)
    drawn = 0
    if total > 0 and reference_image is not None:
        ref_canvas = _map_points(match_reference_points, reference_scale, ref_x0, ref_y0)
        frame_canvas = _map_points(match_frame_points, scene_scale, scene_x0, scene_y0)
        everything = np.arange(total)
        if inlier_mask is None:
            groups = [(_subsample(everything, MAX_DRAWN_MATCHES), COLOR_MATCH_PLAIN)]
        else:
            inlier_mask = np.asarray(inlier_mask, dtype=bool)
            inliers = everything[inlier_mask]
            outliers = everything[~inlier_mask]
            chosen_inliers = _subsample(inliers, MAX_DRAWN_MATCHES)
            chosen_outliers = _subsample(outliers, MAX_DRAWN_MATCHES - len(chosen_inliers))
            groups = [
                (chosen_outliers, COLOR_MATCH_OUTLIER),
                (chosen_inliers, COLOR_MATCH_INLIER),
            ]
        for indices, color in groups:
            for index in indices:
                p1 = (int(ref_canvas[index, 0]), int(ref_canvas[index, 1]))
                p2 = (int(frame_canvas[index, 0]), int(frame_canvas[index, 1]))
                cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, 3, color, -1, cv2.LINE_AA)
                drawn += 1

    # Quadrilateral outline on top of the lines.
    if scene_quad is not None:
        cv2.polylines(canvas, [scene_quad.reshape(-1, 1, 2)], True, (20, 20, 20), 5, cv2.LINE_AA)
        cv2.polylines(canvas, [scene_quad.reshape(-1, 1, 2)], True, COLOR_QUAD, 3, cv2.LINE_AA)

    # Labels.
    _put_label(canvas, "REFERENCE", (ref_x0 + 10, ref_y0 + 22))
    _put_label(canvas, "CURRENT SCENE", (scene_x0 + 10, scene_y0 + 22))

    if reference_image is not None:
        badge_text = "DETECTED" if localized else "NOT LOCALIZED"
        badge_bg = COLOR_BADGE_OK if localized else COLOR_BADGE_OFF
        (badge_w, _), _ = cv2.getTextSize(badge_text, FONT, 0.55, 2)
        _put_label(
            canvas, badge_text,
            (scene_x0 + scene_panel.shape[1] - badge_w - 14, scene_y0 + 22),
            color=(255, 255, 255), bg=badge_bg, scale=0.55,
        )

    legend_y = scene_y0 + scene_panel.shape[0] - 10
    if total > 0 and reference_image is not None:
        if inlier_mask is None:
            legend = "cyan lines: accepted by Lowe's ratio test"
        else:
            legend = "green: RANSAC inlier   orange: accepted, geometrically inconsistent"
        if drawn < total:
            legend += f"   (showing {drawn} of {total})"
        _put_label(canvas, legend, (scene_x0 + 10, legend_y), scale=0.45)

    return canvas

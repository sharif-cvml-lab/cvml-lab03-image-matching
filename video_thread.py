from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from annotation import make_placeholder, render_match_view
from config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DEFAULT_CAMERA_INDEX,
    IMAGE_EXTENSIONS,
    MAX_PROCESS_DIMENSION,
    MIN_HOMOGRAPHY_MATCHES,
    MIN_RANSAC_INLIERS,
    QUAD_MAX_AREA_RATIO,
    QUAD_MIN_AREA_RATIO,
    QUAD_MIN_SIDE_PIXELS,
    RANSAC_REPROJECTION_THRESHOLD,
    TARGET_FPS,
    VIDEO_EXTENSIONS,
    FrameResult,
    ProcessingParameters,
    default_parameters,
    validated_parameters,
)
from feature_matching import match_features

IDLE_WAIT = 0.03


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def read_image_file(path: str):
    """Read an image (Unicode-safe). Returns a BGR array or None."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def prepare_image(image):
    """Return a contiguous 8-bit BGR image no larger than MAX_PROCESS_DIMENSION."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        return None
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > MAX_PROCESS_DIMENSION:
        scale = MAX_PROCESS_DIMENSION / longest
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(image)


class _Features:
    """SIFT keypoints, descriptors and an (N, 2) array of keypoint coordinates."""

    def __init__(self, keypoints=(), descriptors=None):
        self.keypoints = tuple(keypoints)
        self.descriptors = descriptors
        if self.keypoints:
            self.points = np.float32([kp.pt for kp in self.keypoints]).reshape(-1, 2)
        else:
            self.points = np.zeros((0, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.keypoints)


# ---------------------------------------------------------------------------
# Fixed post-processing (NOT part of the student's task)
# ---------------------------------------------------------------------------
def _sanitize_matches(raw, reference_count: int, frame_count: int):
    """Turn whatever match_features returned into a safe list of (i, j) pairs."""
    if not isinstance(raw, (list, tuple)):
        return [], "match_features must return a list of (reference_index, frame_index) pairs"
    matches = []
    for item in raw:
        try:
            i, j = item
            i, j = int(i), int(j)
        except (TypeError, ValueError):
            return [], "match_features returned an item that is not an (i, j) index pair"
        if 0 <= i < reference_count and 0 <= j < frame_count:
            matches.append((i, j))
        else:
            return [], f"match_features returned an out-of-range index pair ({i}, {j})"
    return matches, None


def _quad_is_valid(quad: np.ndarray, scene_shape) -> bool:
    """Simple sanity checks so a degenerate projection is never drawn."""
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return False
    if np.max(np.abs(quad)) > 1e5:
        return False
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if np.min(sides) < QUAD_MIN_SIDE_PIXELS:
        return False
    # Signed (shoelace) area: positive keeps the orientation of the reference
    # corners, negative means the projection is mirrored/flipped.
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    scene_area = float(scene_shape[0] * scene_shape[1])
    if area <= 0.0:
        return False
    if not (QUAD_MIN_AREA_RATIO * scene_area <= area <= QUAD_MAX_AREA_RATIO * scene_area):
        return False
    return bool(cv2.isContourConvex(np.rint(quad).astype(np.int32).reshape(-1, 1, 2)))


def _localize(reference_points, frame_points, reference_shape, scene_shape):
    """Homography + RANSAC + corner projection.

    Returns (inlier_mask or None, quad or None, message). The caller guarantees
    at least MIN_HOMOGRAPHY_MATCHES correspondences.
    """
    try:
        homography, mask = cv2.findHomography(
            reference_points.reshape(-1, 1, 2),
            frame_points.reshape(-1, 1, 2),
            cv2.RANSAC,
            RANSAC_REPROJECTION_THRESHOLD,
        )
    except cv2.error:
        return None, None, "Homography estimation failed"
    if homography is None or mask is None or not np.all(np.isfinite(homography)):
        return None, None, "Homography estimation failed"

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < MIN_RANSAC_INLIERS:
        return inlier_mask, None, f"Too few RANSAC inliers ({inliers} < {MIN_RANSAC_INLIERS})"

    height, width = reference_shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    try:
        quad = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
    except cv2.error:
        return inlier_mask, None, "Corner projection failed"
    if not _quad_is_valid(quad, scene_shape):
        return inlier_mask, None, "Projected quadrilateral rejected as implausible"
    return inlier_mask, quad, "Reference localized"


# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------
class VideoThread(QThread):
    """Acquire scene frames, run SIFT + the student's matcher, render the canvas."""

    frame_ready = pyqtSignal(object, object)  # canvas (BGR ndarray), FrameResult
    source_status = pyqtSignal(str)
    source_error = pyqtSignal(str)
    source_changed = pyqtSignal(str, str)  # kind ("webcam"/"image"/"video"), description
    reference_changed = pyqtSignal(bool, str)  # loaded, description
    paused_changed = pyqtSignal(bool)

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._wake = threading.Event()

        # Shared with the GUI thread (always accessed under the lock).
        self._params: ProcessingParameters = default_parameters()
        self._commands: list[tuple] = []
        self._paused = False
        self._dirty = True

        # Owned by the processing thread.
        self._camera_index = camera_index
        self._kind = "webcam"
        self._cap = None
        self._video_ended = False
        self._video_period = 1.0 / TARGET_FPS
        self._scene = None
        self._scene_token = 0
        self._scene_message = "Starting camera..."
        self._last_status = ""
        self._reference = None
        self._reference_label = ""
        self._sift = None
        self._sift_keypoints = None
        self._reference_features = None
        self._reference_features_for = None
        self._frame_features = None
        self._frame_features_for = None
        self._frame_times: deque[float] = deque(maxlen=10)
        self._reported_errors: set[str] = set()
        self.extraction_count = 0  # number of SIFT runs (handy for self-tests)

    # ----- GUI-facing API (thread-safe) ------------------------------------
    def set_parameters(self, parameters: ProcessingParameters) -> None:
        safe = validated_parameters(
            parameters.ratio_threshold, parameters.max_keypoints, parameters.show_keypoints
        )
        with self._lock:
            self._params = safe
            self._dirty = True
        self._wake.set()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)
        self._wake.set()

    def set_camera_index(self, camera_index: int) -> None:
        self._post(("camera_index", int(camera_index)))

    def use_webcam(self) -> None:
        self._post(("webcam",))

    def open_scene_file(self, path: str) -> None:
        self._post(("file", path))

    def restart_source(self) -> None:
        self._post(("restart",))

    def set_reference_image(self, image, label: str) -> None:
        self._post(("ref_image", image, label))

    def capture_reference(self) -> None:
        self._post(("ref_capture",))

    def clear_reference(self) -> None:
        self._post(("ref_clear",))

    def stop(self) -> None:
        self._running.clear()
        self._wake.set()
        self.wait()

    def _post(self, command: tuple) -> None:
        with self._lock:
            self._commands.append(command)
            self._dirty = True
        self._wake.set()

    # ----- helpers ----------------------------------------------------------
    def _snapshot(self):
        with self._lock:
            return self._params, self._paused

    def _emit_source_status(self, message: str) -> None:
        if message != self._last_status:
            self._last_status = message
            self.source_status.emit(message)

    def _set_scene_message(self, message: str) -> None:
        if message != self._scene_message:
            self._scene_message = message
            with self._lock:
                self._dirty = True

    def _set_paused_internal(self, paused: bool) -> None:
        with self._lock:
            changed = self._paused != paused
            self._paused = paused
        if changed:
            self.paused_changed.emit(paused)

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _set_scene(self, frame) -> bool:
        prepared = prepare_image(frame)
        if prepared is None:
            return False
        self._scene = prepared
        self._scene_token += 1
        return True

    def _report_once(self, message: str) -> None:
        if message not in self._reported_errors:
            self._reported_errors.add(message)
            print(message)

    # ----- commands -----------------------------------------------------------
    def _apply_commands(self) -> None:
        with self._lock:
            commands, self._commands = self._commands, []
        for command in commands:
            try:
                self._run_command(command)
            except Exception as exc:  # keep the app alive after any failure
                self._report_once(traceback.format_exc())
                self.source_error.emit(f"Unexpected error: {exc}")

    def _run_command(self, command: tuple) -> None:
        name = command[0]
        if name == "webcam":
            self._switch_to_webcam()
        elif name == "camera_index":
            self._camera_index = command[1]
            if self._kind == "webcam":
                self._release_capture()
        elif name == "file":
            self._open_file(command[1])
        elif name == "restart":
            self._restart_source()
        elif name == "ref_image":
            self._set_reference(command[1], command[2])
        elif name == "ref_capture":
            if self._scene is None:
                self.source_error.emit("No scene frame is available to capture yet.")
            else:
                self._set_reference(self._scene.copy(), "Captured from the current scene")
        elif name == "ref_clear":
            self._reference = None
            self._reference_label = ""
            self._reference_features = None
            self._reference_features_for = None
            self.reference_changed.emit(False, "")

    def _set_reference(self, image, label: str) -> None:
        prepared = prepare_image(image)
        if prepared is None:
            self.source_error.emit("The reference image is empty or has an unsupported format.")
            return
        self._reference = prepared
        self._reference_label = label
        self._reference_features = None
        self._reference_features_for = None
        self.reference_changed.emit(True, label)

    def _switch_to_webcam(self) -> None:
        self._release_capture()
        self._kind = "webcam"
        self._video_ended = False
        self._scene = None
        self._scene_message = "Starting camera..."
        self._frame_times.clear()
        self._set_paused_internal(False)
        self.source_changed.emit("webcam", f"Webcam {self._camera_index}")

    def _open_file(self, path: str) -> None:
        extension = Path(path).suffix.lower()
        if extension in VIDEO_EXTENSIONS:
            order = ("video", "image")
        elif extension in IMAGE_EXTENSIONS:
            order = ("image", "video")
        else:
            order = ("image", "video")
        name = Path(path).name
        for kind in order:
            if kind == "image":
                image = prepare_image(read_image_file(path))
                if image is None:
                    continue
                self._release_capture()
                self._kind = "image"
                self._video_ended = False
                self._scene = image
                self._scene_token += 1
                self._frame_times.clear()
                self._set_paused_internal(False)
                self.source_changed.emit("image", f"Image: {name}")
                self._emit_source_status(f"Scene image loaded: {name}")
                return
            cap = cv2.VideoCapture(path)
            ok, frame = (cap.read() if cap.isOpened() else (False, None))
            if not ok or prepare_image(frame) is None:
                cap.release()
                continue
            self._release_capture()
            self._cap = cap
            self._kind = "video"
            self._video_ended = False
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._video_period = 1.0 / fps if 1.0 < fps <= 60.0 else 1.0 / TARGET_FPS
            self._set_scene(frame)
            self._frame_times.clear()
            self._set_paused_internal(False)
            self.source_changed.emit("video", f"Video: {name}")
            self._emit_source_status(f"Playing video: {name}")
            return
        self.source_error.emit(
            f"Could not open '{name}'. The file is missing, corrupted or not a supported "
            "image/video format."
        )

    def _restart_source(self) -> None:
        if self._kind == "webcam":
            self._release_capture()
            self._set_scene_message("Restarting camera...")
        elif self._kind == "video" and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_ended = False
            self._set_paused_internal(False)
            self._emit_source_status("Video restarted")

    # ----- acquisition ----------------------------------------------------------
    def _acquire_frame(self, paused: bool):
        """Return (got_new_frame, seconds_to_wait_before_next_loop)."""
        if self._kind == "image":
            return False, IDLE_WAIT

        if self._kind == "webcam":
            if paused:
                self._emit_source_status("Camera paused")
                return False, IDLE_WAIT
            if self._cap is None and not self._open_webcam():
                return False, 1.0
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._release_capture()
                self._emit_source_status("Camera frame read failed. Reconnecting...")
                self._set_scene_message("Camera frame unavailable. Reconnecting...")
                return False, 0.25
            if not self._set_scene(frame):
                return False, IDLE_WAIT
            self._emit_source_status(f"Camera {self._camera_index} connected")
            self._frame_times.append(time.perf_counter())
            return True, 1.0 / TARGET_FPS

        # video file
        if self._cap is None:
            return False, IDLE_WAIT
        if self._video_ended:
            if paused:
                return False, IDLE_WAIT
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_ended = False
        elif paused:
            self._emit_source_status("Video paused")
            return False, IDLE_WAIT
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._video_ended = True
            self._frame_times.clear()
            self._set_paused_internal(True)
            self._emit_source_status("Video ended - showing the last frame. Press Resume to replay.")
            return False, IDLE_WAIT
        if not self._set_scene(frame):
            return False, IDLE_WAIT
        self._emit_source_status("Playing video")
        self._frame_times.append(time.perf_counter())
        return True, self._video_period

    def _open_webcam(self) -> bool:
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            cap.release()
            self._emit_source_status(
                f"Camera could not be opened (index {self._camera_index}). Retrying..."
            )
            self._set_scene_message("Camera could not be opened")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _current_fps(self, paused: bool) -> float:
        if self._kind == "image" or paused or len(self._frame_times) < 2:
            return 0.0
        if time.perf_counter() - self._frame_times[-1] > 1.0:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # ----- SIFT ------------------------------------------------------------------
    def _get_sift(self, max_keypoints: int):
        if self._sift is None or self._sift_keypoints != max_keypoints:
            self._sift = cv2.SIFT_create(nfeatures=max_keypoints)
            self._sift_keypoints = max_keypoints
        return self._sift

    def _extract(self, image, max_keypoints: int) -> _Features:
        self.extraction_count += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            keypoints, descriptors = self._get_sift(max_keypoints).detectAndCompute(gray, None)
        except cv2.error:
            return _Features()
        if descriptors is None or len(keypoints) == 0:
            return _Features()
        return _Features(keypoints, np.ascontiguousarray(descriptors, dtype=np.float32))

    # ----- processing ------------------------------------------------------------
    def _process(self, params: ProcessingParameters, paused: bool):
        started = time.perf_counter()
        fps = self._current_fps(paused)

        if self._scene is None:
            placeholder = make_placeholder(
                CAPTURE_WIDTH, CAPTURE_HEIGHT, [self._scene_message or "No scene available"]
            )
            canvas = render_match_view(
                self._reference, placeholder, np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros((0, 2)), np.zeros((0, 2)), None, None, False, False,
            )
            status = (
                self._scene_message
                if self._reference is not None
                else "No reference image selected"
            )
            return canvas, FrameResult(
                reference_loaded=self._reference is not None, status=status
            )

        if self._reference is None:
            canvas = render_match_view(
                None, self._scene, np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros((0, 2)), np.zeros((0, 2)), None, None, False, False,
            )
            return canvas, FrameResult(
                status="No reference image selected - capture or load one",
                processing_ms=(time.perf_counter() - started) * 1000.0,
                fps=fps,
            )

        try:
            self._get_sift(params.max_keypoints)
        except (AttributeError, cv2.error):
            canvas = render_match_view(
                self._reference, self._scene, np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros((0, 2)), np.zeros((0, 2)), None, None, False, False,
            )
            return canvas, FrameResult(
                reference_loaded=True,
                status="SIFT is unavailable - install opencv-python-headless>=4.5.5",
            )

        # Cached features: recomputed only when the image or Max Keypoints changed.
        if self._reference_features_for != (id(self._reference), params.max_keypoints):
            self._reference_features = self._extract(self._reference, params.max_keypoints)
            self._reference_features_for = (id(self._reference), params.max_keypoints)
        if self._frame_features_for != (self._scene_token, params.max_keypoints):
            self._frame_features = self._extract(self._scene, params.max_keypoints)
            self._frame_features_for = (self._scene_token, params.max_keypoints)
        reference_features = self._reference_features
        frame_features = self._frame_features

        # ---- the student's function ----
        match_started = time.perf_counter()
        problem = None
        try:
            raw_matches = match_features(
                reference_features.descriptors,
                frame_features.descriptors,
                params.ratio_threshold,
            )
        except Exception as exc:  # student bugs must not kill the application
            raw_matches = []
            problem = f"match_features raised {type(exc).__name__}: {exc}"
            self._report_once(traceback.format_exc())
        match_ms = (time.perf_counter() - match_started) * 1000.0
        matches, sanitize_problem = _sanitize_matches(
            raw_matches, len(reference_features), len(frame_features)
        )
        problem = problem or sanitize_problem

        # ---- fixed downstream geometry ----
        inlier_mask = None
        quad = None
        inliers = -1
        localization_text = "NOT LOCALIZED"
        match_reference_points = np.zeros((0, 2), dtype=np.float32)
        match_frame_points = np.zeros((0, 2), dtype=np.float32)
        geometry_message = ""
        if matches:
            reference_indices = np.fromiter((m[0] for m in matches), dtype=np.int64)
            frame_indices = np.fromiter((m[1] for m in matches), dtype=np.int64)
            match_reference_points = reference_features.points[reference_indices]
            match_frame_points = frame_features.points[frame_indices]
            if len(matches) >= MIN_HOMOGRAPHY_MATCHES:
                inlier_mask, quad, geometry_message = _localize(
                    match_reference_points,
                    match_frame_points,
                    self._reference.shape,
                    self._scene.shape,
                )
                if inlier_mask is not None:
                    inliers = int(inlier_mask.sum())
            else:
                geometry_message = (
                    f"Too few accepted matches for a Homography ({len(matches)} < "
                    f"{MIN_HOMOGRAPHY_MATCHES})"
                )
        localized = quad is not None
        if localized:
            localization_text = "DETECTED"
        processing_ms = (time.perf_counter() - started) * 1000.0

        if problem:
            status = problem
        elif not matches:
            status = "No accepted matches"
        elif localized:
            status = f"{len(matches)} accepted matches, {inliers} inliers - reference localized"
        else:
            status = f"{len(matches)} accepted matches - No reliable localization: {geometry_message}"

        canvas = render_match_view(
            self._reference, self._scene,
            reference_features.points, frame_features.points,
            match_reference_points, match_frame_points,
            inlier_mask, quad, localized, params.show_keypoints,
        )
        return canvas, FrameResult(
            reference_loaded=True,
            reference_keypoints=len(reference_features),
            frame_keypoints=len(frame_features),
            accepted_matches=len(matches),
            inliers=inliers,
            localization=localization_text,
            match_ms=match_ms,
            processing_ms=processing_ms,
            fps=fps,
            status=status,
        )

    # ----- main loop ---------------------------------------------------------------
    def run(self) -> None:
        try:
            while self._running.is_set():
                loop_started = time.perf_counter()
                wait = IDLE_WAIT
                try:
                    self._apply_commands()
                    params, paused = self._snapshot()
                    got_frame, wait = self._acquire_frame(paused)
                    with self._lock:
                        dirty, self._dirty = self._dirty, False
                    if got_frame or dirty:
                        canvas, result = self._process(params, paused)
                        self.frame_ready.emit(canvas, result)
                except Exception as exc:  # never let the thread die
                    self._report_once(traceback.format_exc())
                    self._emit_source_status(f"Unexpected processing error: {exc}")
                    wait = 0.25
                elapsed = time.perf_counter() - loop_started
                self._wake.wait(timeout=max(0.005, wait - elapsed))
                self._wake.clear()
        finally:
            self._release_capture()
            self.source_status.emit("Stopped")

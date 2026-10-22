from __future__ import annotations

import cv2
from PyQt6.QtCore import QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_NAME,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_MAX_KEYPOINTS,
    DEFAULT_RATIO_THRESHOLD,
    DEFAULT_SHOW_KEYPOINTS,
    DEFAULT_THEME,
    IMAGE_EXTENSIONS,
    MAX_KEYPOINTS_MAX,
    MAX_KEYPOINTS_MIN,
    RATIO_SLIDER_SCALE,
    RATIO_THRESHOLD_MAX,
    RATIO_THRESHOLD_MIN,
    SHORTCUTS,
    VIDEO_EXTENSIONS,
    FrameResult,
    ProcessingParameters,
    validated_parameters,
)
from video_thread import VideoThread, read_image_file

DARK_STYLE = """
QMainWindow, QWidget {
    background: #11151b;
    color: #e7ebf0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: #171c23;
    border: 1px solid #28303a;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: #0a0d11;
    border: 1px solid #303945;
    border-radius: 10px;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: #8e98a6; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: #9aa6b4; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#ValueChip {
    background: #202833;
    border: 1px solid #364250;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: #212833;
    color: #eef2f6;
    border: 1px solid #35404d;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #2a3441; border-color: #4e6175; }
QPushButton:pressed { background: #18202a; }
QPushButton#accent { background: #3a7d67; border-color: #4c9b80; font-weight: 700; }
QPushButton#accent:hover { background: #479276; }
QPushButton#danger { background: #322126; border-color: #5d333d; }
QCheckBox, QComboBox, QSpinBox {
    background: #1a2028;
    border: 1px solid #35404d;
    border-radius: 7px;
    padding: 6px;
}
QSpinBox:hover { border-color: #4e6175; }
QSlider::groove:horizontal { height: 5px; background: #2b333d; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4d9d80; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #80c6ab;
    border: 1px solid #9ad9c0;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QGroupBox {
    border: 1px solid #28303a;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #313b47; border-radius: 5px; min-height: 30px; }
"""

LIGHT_STYLE = """
QMainWindow, QWidget {
    background: #eef2f5;
    color: #1e2630;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: #ffffff;
    border: 1px solid #d4dbe2;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: #f7f9fb;
    border: 1px solid #cbd4dd;
    border-radius: 10px;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: #667485; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: #657180; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#ValueChip {
    background: #f1f4f7;
    border: 1px solid #d2d9e0;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: #ffffff;
    color: #1e2630;
    border: 1px solid #cbd4dd;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #f3f6f8; border-color: #acbac7; }
QPushButton#accent { background: #3f806b; border-color: #3f806b; color: white; font-weight: 700; }
QPushButton#danger { background: #fff3f4; border-color: #e2b7bc; }
QCheckBox, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #cbd4dd;
    border-radius: 7px;
    padding: 6px;
}
QSlider::groove:horizontal { height: 5px; background: #d9e0e6; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4a9c80; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #4a9c80;
    border: 1px solid #3f806b;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QGroupBox {
    border: 1px solid #d4dbe2;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #c0cad3; border-radius: 5px; min-height: 30px; }
"""

OK_COLOR = "color: #3fae7c;"


class VideoLabel(QLabel):
    """Scalable image view that preserves the source aspect ratio."""

    def __init__(self):
        super().__init__()
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ValueSlider(QWidget):
    """Integer slider with a title and a chip that always shows the current value."""

    valueChanged = pyqtSignal(int)

    def __init__(self, title, minimum, maximum, value, formatter=str, parent=None):
        super().__init__(parent)
        self._formatter = formatter
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._on_value_changed)

        self.title_label = QLabel(title)
        self.title_label.setMinimumWidth(110)
        self.value_label = QLabel()
        self.value_label.setObjectName("ValueChip")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self._set_value_label(value)

    def _set_value_label(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))

    def _on_value_changed(self, value: int) -> None:
        self._set_value_label(value)
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int, emit: bool = True) -> None:
        if emit:
            self.slider.setValue(value)
        else:
            blocked = self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(blocked)
        self._set_value_label(self.slider.value())


class ImageMatchingWindow(QMainWindow):
    """Main UI for interactive SIFT descriptor matching."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1500, 900)

        self.settings = QSettings()
        self.video_thread: VideoThread | None = None
        self._closing = False
        self._paused = False
        self.current_theme = self.settings.value("theme", DEFAULT_THEME)
        self._last_dir = self.settings.value("last_dir", "", type=str)

        self._build_ui()
        self._restore_settings()
        self._apply_theme()
        self._start_video_thread()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 12)
        root_layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        title_box = QVBoxLayout()
        title = QLabel("Image Matching Studio")
        title.setObjectName("Title")
        subtitle = QLabel("SIFT descriptors, L2 distance, and Lowe's Ratio Test in real time")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)
        self.theme_button = QPushButton("☼ Light Mode")
        self.theme_button.clicked.connect(self._toggle_theme)
        self.fullscreen_button = QPushButton("Full Screen")
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        header_layout.addWidget(self.theme_button)
        header_layout.addWidget(self.fullscreen_button)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        video_side = QWidget()
        video_layout = QVBoxLayout(video_side)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("VideoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(8)
        card_title = QLabel("Reference Image  |  Current Scene / Matching View")
        card_title.setObjectName("PanelTitle")
        card_layout.addWidget(card_title)
        canvas = QFrame()
        canvas.setObjectName("VideoCanvas")
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_label = VideoLabel()
        self.canvas_label.setText("Starting...")
        canvas_layout.addWidget(self.canvas_label)
        card_layout.addWidget(canvas, 1)
        video_layout.addWidget(card, 1)

        status = QFrame()
        status.setObjectName("StatusBar")
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(14, 8, 14, 8)
        self.status_label = QLabel("Starting...")
        self.status_label.setObjectName("MetricValue")
        self.source_label = QLabel("")
        self.source_label.setObjectName("Muted")
        self.matches_label = QLabel("Accepted Matches: 0")
        self.matches_label.setObjectName("MetricValue")
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("MetricValue")
        left = QVBoxLayout()
        left.setSpacing(2)
        left.addWidget(self.status_label)
        left.addWidget(self.source_label)
        status_layout.addLayout(left, 1)
        status_layout.addWidget(self.matches_label)
        status_layout.addSpacing(18)
        status_layout.addWidget(self.fps_label)
        video_layout.addWidget(status)

        splitter.addWidget(video_side)
        splitter.addWidget(self._build_sidebar())
        splitter.setSizes([1120, 380])
        root_layout.addWidget(splitter, 1)
        self.splitter = splitter

        self.setCentralWidget(central)
        self.statusBar().hide()
        self._build_shortcuts()

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        # --- Scene source -------------------------------------------------
        source_box = QGroupBox("Scene Source")
        source_layout = QVBoxLayout(source_box)
        self.scene_label = QLabel("Webcam")
        self.scene_label.setObjectName("Muted")
        self.scene_label.setWordWrap(True)
        source_layout.addWidget(self.scene_label)
        source_row = QHBoxLayout()
        webcam_button = QPushButton("Use Webcam")
        webcam_button.clicked.connect(self._use_webcam)
        open_button = QPushButton("Open Scene Image/Video…")
        open_button.clicked.connect(self._open_scene)
        source_row.addWidget(webcam_button)
        source_row.addWidget(open_button)
        source_layout.addLayout(source_row)
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera index"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.camera_spin.valueChanged.connect(self._camera_index_changed)
        camera_row.addWidget(self.camera_spin)
        source_layout.addLayout(camera_row)
        action_row = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.restart_button = QPushButton("Restart Source")
        self.restart_button.clicked.connect(self._restart_source)
        action_row.addWidget(self.pause_button)
        action_row.addWidget(self.restart_button)
        source_layout.addLayout(action_row)
        layout.addWidget(source_box)

        # --- Reference ------------------------------------------------------
        reference_box = QGroupBox("Reference Image")
        reference_layout = QVBoxLayout(reference_box)
        self.reference_label = QLabel("No reference image selected")
        self.reference_label.setObjectName("Muted")
        self.reference_label.setWordWrap(True)
        reference_layout.addWidget(self.reference_label)
        capture_button = QPushButton(f"Capture Reference  ({SHORTCUTS['capture']})")
        capture_button.setObjectName("accent")
        capture_button.clicked.connect(self._capture_reference)
        load_button = QPushButton("Load Reference Image…")
        load_button.clicked.connect(self._load_reference)
        clear_button = QPushButton("Clear Reference")
        clear_button.clicked.connect(self._clear_reference)
        reference_layout.addWidget(capture_button)
        row = QHBoxLayout()
        row.addWidget(load_button)
        row.addWidget(clear_button)
        reference_layout.addLayout(row)
        layout.addWidget(reference_box)

        # --- Matching controls ------------------------------------------------
        matching_box = QGroupBox("Feature Matching")
        matching_layout = QVBoxLayout(matching_box)
        self.ratio_slider = ValueSlider(
            "Ratio Threshold",
            int(round(RATIO_THRESHOLD_MIN * RATIO_SLIDER_SCALE)),
            int(round(RATIO_THRESHOLD_MAX * RATIO_SLIDER_SCALE)),
            int(round(DEFAULT_RATIO_THRESHOLD * RATIO_SLIDER_SCALE)),
            formatter=lambda v: f"{v / RATIO_SLIDER_SCALE:.2f}",
        )
        matching_layout.addWidget(self.ratio_slider)
        ratio_note = QLabel(
            "A match is accepted when d1 < ratio × d2. Higher = looser (more, "
            "but more ambiguous matches); lower = stricter."
        )
        ratio_note.setObjectName("Muted")
        ratio_note.setWordWrap(True)
        matching_layout.addWidget(ratio_note)
        self.keypoint_slider = ValueSlider(
            "Max Keypoints", MAX_KEYPOINTS_MIN, MAX_KEYPOINTS_MAX, DEFAULT_MAX_KEYPOINTS
        )
        matching_layout.addWidget(self.keypoint_slider)
        keypoint_note = QLabel(
            "Upper limit for SIFT keypoints. Matching cost grows with "
            "N_reference × N_frame × 128, so more keypoints means a slower frame rate."
        )
        keypoint_note.setObjectName("Muted")
        keypoint_note.setWordWrap(True)
        matching_layout.addWidget(keypoint_note)
        self.show_keypoints = QCheckBox("Show keypoints")
        self.show_keypoints.setChecked(DEFAULT_SHOW_KEYPOINTS)
        matching_layout.addWidget(self.show_keypoints)
        layout.addWidget(matching_box)

        # --- Live metrics -------------------------------------------------------
        metrics_box = QGroupBox("Live Metrics")
        metrics_layout = QGridLayout(metrics_box)
        metrics_layout.setHorizontalSpacing(12)
        self.metric_values: dict[str, QLabel] = {}
        rows = [
            ("reference", "Reference"),
            ("reference_keypoints", "Reference Keypoints"),
            ("frame_keypoints", "Frame Keypoints"),
            ("accepted", "Accepted Matches"),
            ("inliers", "RANSAC Inliers"),
            ("localization", "Localization"),
            ("match_time", "Matching Time"),
            ("processing_time", "Processing Time"),
            ("fps", "FPS"),
        ]
        for row_index, (key, text) in enumerate(rows):
            name = QLabel(text)
            name.setObjectName("Metric")
            value = QLabel("-")
            value.setObjectName("MetricValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            metrics_layout.addWidget(name, row_index, 0)
            metrics_layout.addWidget(value, row_index, 1)
            self.metric_values[key] = value
        metrics_note = QLabel(
            "More accepted matches does not by itself mean they are correct, or "
            "that the reference was found. RANSAC and the quadrilateral are fixed "
            "application features, not part of your task."
        )
        metrics_note.setObjectName("Muted")
        metrics_note.setWordWrap(True)
        metrics_layout.addWidget(metrics_note, len(rows), 0, 1, 2)
        layout.addWidget(metrics_box)

        # --- Bottom buttons -----------------------------------------------------
        bottom_row = QHBoxLayout()
        reset = QPushButton("Reset Defaults")
        reset.setObjectName("accent")
        reset.clicked.connect(self._reset_defaults)
        quit_button = QPushButton("Quit")
        quit_button.setObjectName("danger")
        quit_button.clicked.connect(self.close)
        bottom_row.addWidget(reset)
        bottom_row.addWidget(quit_button)
        layout.addLayout(bottom_row)

        hint = QLabel(
            "Shortcuts: Space = pause/resume · C = capture reference · "
            "F11 = full screen · R = reset · Q = quit"
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

        self.ratio_slider.valueChanged.connect(self._parameters_changed)
        self.keypoint_slider.valueChanged.connect(self._parameters_changed)
        self.show_keypoints.toggled.connect(self._parameters_changed)

        scroll.setWidget(content)
        return scroll

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(SHORTCUTS["pause"]), self, activated=self._toggle_pause)
        QShortcut(QKeySequence(SHORTCUTS["fullscreen"]), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(SHORTCUTS["reset"]), self, activated=self._reset_defaults)
        QShortcut(QKeySequence(SHORTCUTS["quit"]), self, activated=self.close)
        QShortcut(QKeySequence(SHORTCUTS["capture"]), self, activated=self._capture_reference)

    # ------------------------------------------------------------ video thread
    def _start_video_thread(self) -> None:
        self._stop_video_thread()
        thread = VideoThread(self.camera_spin.value(), self)
        thread.set_parameters(self._current_parameters())
        thread.frame_ready.connect(self._update_frame)
        thread.source_status.connect(self.source_label.setText)
        thread.source_error.connect(self._show_error)
        thread.source_changed.connect(self._source_changed)
        thread.reference_changed.connect(self._reference_changed)
        thread.paused_changed.connect(self._set_pause_ui)
        self.video_thread = thread
        thread.start()

    def _stop_video_thread(self) -> None:
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

    # ----------------------------------------------------------------- actions
    def _current_parameters(self) -> ProcessingParameters:
        return validated_parameters(
            self.ratio_slider.value() / RATIO_SLIDER_SCALE,
            self.keypoint_slider.value(),
            self.show_keypoints.isChecked(),
        )

    def _parameters_changed(self, *_args) -> None:
        if self.video_thread is not None:
            self.video_thread.set_parameters(self._current_parameters())

    def _use_webcam(self) -> None:
        if self.video_thread is not None:
            self.video_thread.use_webcam()

    def _scene_dialog_filter(self) -> str:
        images = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
        videos = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
        return (
            f"Images and videos ({images} {videos});;Images ({images});;"
            f"Videos ({videos});;All files (*)"
        )

    def _open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scene image or video", self._last_dir, self._scene_dialog_filter()
        )
        if path and self.video_thread is not None:
            self._remember_directory(path)
            self.video_thread.open_scene_file(path)

    def _load_reference(self) -> None:
        images = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference image", self._last_dir, f"Images ({images});;All files (*)"
        )
        if not path:
            return
        self._remember_directory(path)
        image = read_image_file(path)
        if image is None:
            self._show_error(
                "The selected file could not be read as an image. Choose a valid "
                "PNG, JPEG, BMP, TIFF or WebP file."
            )
            return
        if self.video_thread is not None:
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            self.video_thread.set_reference_image(image, f"File: {name}")

    def _capture_reference(self) -> None:
        if self.video_thread is not None:
            self.video_thread.capture_reference()

    def _clear_reference(self) -> None:
        if self.video_thread is not None:
            self.video_thread.clear_reference()

    def _remember_directory(self, path: str) -> None:
        normalized = path.replace("\\", "/")
        if "/" in normalized:
            self._last_dir = normalized.rsplit("/", 1)[0]

    def _camera_index_changed(self, _: int) -> None:
        if self.video_thread is not None:
            self.video_thread.set_camera_index(self.camera_spin.value())

    def _restart_source(self) -> None:
        if self.video_thread is not None:
            self.video_thread.restart_source()

    def _toggle_pause(self) -> None:
        if self.video_thread is None:
            return
        paused = not self._paused
        self.video_thread.set_paused(paused)
        self._set_pause_ui(paused)

    def _set_pause_ui(self, paused: bool) -> None:
        self._paused = paused
        self.pause_button.setText("Resume" if paused else "Pause")

    def _show_error(self, message: str) -> None:
        if not self._closing:
            QMessageBox.warning(self, APP_NAME, message)

    def _source_changed(self, kind: str, description: str) -> None:
        self.scene_label.setText(description)

    def _reference_changed(self, loaded: bool, description: str) -> None:
        if loaded:
            self.reference_label.setText(f"Loaded - {description}")
        else:
            self.reference_label.setText("No reference image selected")

    def _reset_defaults(self) -> None:
        self.ratio_slider.setValue(
            int(round(DEFAULT_RATIO_THRESHOLD * RATIO_SLIDER_SCALE)), emit=False
        )
        self.keypoint_slider.setValue(DEFAULT_MAX_KEYPOINTS, emit=False)
        self.show_keypoints.blockSignals(True)
        self.show_keypoints.setChecked(DEFAULT_SHOW_KEYPOINTS)
        self.show_keypoints.blockSignals(False)
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.current_theme = DEFAULT_THEME
        self._apply_theme()
        self._parameters_changed()
        self.source_label.setText("Default settings restored (the reference image is kept)")

    # -------------------------------------------------------------------- theme
    def _toggle_theme(self) -> None:
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme()

    def _apply_theme(self) -> None:
        is_dark = self.current_theme == "dark"
        self.setStyleSheet(DARK_STYLE if is_dark else LIGHT_STYLE)
        self.theme_button.setText("☼ Light Mode" if is_dark else "◐ Dark Mode")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            self.fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")

    # ----------------------------------------------------------------- display
    @staticmethod
    def _to_qimage(frame) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(
            rgb.data, width, height, width * channels, QImage.Format.Format_RGB888
        ).copy()

    def _update_frame(self, canvas, result: FrameResult) -> None:
        if self._closing:
            return
        self.canvas_label.set_frame(self._to_qimage(canvas))
        self.status_label.setText(result.status or "Ready")
        self.matches_label.setText(f"Accepted Matches: {result.accepted_matches}")
        self.fps_label.setText(f"FPS: {result.fps:.1f}" if result.fps > 0 else "FPS: --")

        values = self.metric_values
        values["reference"].setText("Loaded" if result.reference_loaded else "Not Loaded")
        values["reference_keypoints"].setText(str(result.reference_keypoints))
        values["frame_keypoints"].setText(str(result.frame_keypoints))
        values["accepted"].setText(str(result.accepted_matches))
        values["inliers"].setText(str(result.inliers) if result.inliers >= 0 else "-")
        values["localization"].setText(result.localization)
        values["localization"].setStyleSheet(
            OK_COLOR if result.localization == "DETECTED" else ""
        )
        values["match_time"].setText(
            f"{result.match_ms:.1f} ms" if result.reference_loaded else "-"
        )
        values["processing_time"].setText(
            f"{result.processing_ms:.1f} ms" if result.reference_loaded else "-"
        )
        values["fps"].setText(f"{result.fps:.1f}" if result.fps > 0 else "--")

    # ----------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        self.camera_spin.setValue(
            self.settings.value("camera_index", DEFAULT_CAMERA_INDEX, type=int)
        )
        parameters = validated_parameters(
            self.settings.value("ratio_threshold", DEFAULT_RATIO_THRESHOLD, type=float),
            self.settings.value("max_keypoints", DEFAULT_MAX_KEYPOINTS, type=int),
            self.settings.value("show_keypoints", DEFAULT_SHOW_KEYPOINTS, type=bool),
        )
        self.ratio_slider.setValue(
            int(round(parameters.ratio_threshold * RATIO_SLIDER_SCALE)), emit=False
        )
        self.keypoint_slider.setValue(parameters.max_keypoints, emit=False)
        self.show_keypoints.setChecked(parameters.show_keypoints)

        if self.current_theme not in ("dark", "light"):
            self.current_theme = DEFAULT_THEME

        geometry = self.settings.value("window/geometry")
        splitter_state = self.settings.value("window/splitter")
        if geometry:
            self.restoreGeometry(geometry)
        if splitter_state:
            self.splitter.restoreState(splitter_state)

    def _save_settings(self) -> None:
        parameters = self._current_parameters()
        self.settings.setValue("theme", self.current_theme)
        self.settings.setValue("camera_index", self.camera_spin.value())
        self.settings.setValue("ratio_threshold", parameters.ratio_threshold)
        self.settings.setValue("max_keypoints", parameters.max_keypoints)
        self.settings.setValue("show_keypoints", parameters.show_keypoints)
        self.settings.setValue("last_dir", self._last_dir)
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.splitter.saveState())
        self.settings.sync()

    def closeEvent(self, event) -> None:
        self._closing = True
        self._save_settings()
        self._stop_video_thread()
        event.accept()

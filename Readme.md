# Lab03_ImageMatching

Welcome to the third assignment of the Computer Vision and Machine Learning lab course!

This project provides a real-time PyQt6 application for **feature matching**. You select a reference image (a book cover, a poster, a printed card...) and the application looks for it in a webcam stream, an image, or a video. SIFT keypoints and descriptors are provided for you. Your job is to write the part that decides **which descriptors match**.

## 🎯 Assignment Objectives

In this lab you will complete one specific function located in `feature_matching.py`:

```python
def match_features(
    reference_descriptors: np.ndarray,
    frame_descriptors: np.ndarray,
    ratio_threshold: float,
) -> list[tuple[int, int]]:
```

You will practice:

- local image features, keypoints, and descriptors
- SIFT descriptors (128-dimensional `float32` vectors) as a provided tool
- Euclidean / L2 distance between two descriptor vectors (`math.dist` is allowed for one pair)
- finding the nearest and second-nearest frame descriptor for each reference descriptor
- Lowe's Ratio Test and its threshold
- interpreting accepted correspondences and understanding ambiguity

Students should modify only `feature_matching.py`. Do not modify `main.py`, `gui.py`, `video_thread.py`, `annotation.py`, `config.py`, `requirements.txt`, or any other project file.

### What you implement

For every reference descriptor `i`:

1. compute the L2 distance from reference descriptor `i` to every frame descriptor `j`, storing each distance and index pair (e.g., `(d_ij, j)`) in a list;
2. sort the distance list in ascending order to easily get the smallest distance `d1` (with index `j1`) and the second-smallest distance `d2`;
3. accept the match only if `d1 < ratio_threshold * d2` (strict `<`), and then append `(i, j1)`;
4. return the list of accepted `(reference_index, frame_index)` pairs.

The function must return `[]` for empty or invalid inputs, incompatible descriptor sizes, non-finite values, fewer than two frame descriptors, or a threshold outside `0 < ratio_threshold < 1`.

Not allowed: `cv2.BFMatcher`, `cv2.DescriptorMatcher`, `match`, `knnMatch`, any ready-made matcher or nearest-neighbor search, `np.linalg.norm`, NumPy broadcasting / vectorized distance formulas, or SciPy. The standard-library `math.dist` is allowed only to get the distance between two descriptors; the nearest / second-nearest search and the ratio test must be written by you. Matching is one-way (reference → frame): no cross-check and no one-to-one rule.

The detailed step-by-step TODO is inside `feature_matching.py`.

### Matching pipeline

```
Reference image ──► SIFT (fixed) ──► reference keypoints + descriptors (cached)
Current scene   ──► SIFT (fixed) ──► frame keypoints + descriptors
                                           │
        ┌──────────────────────────────────┘
        ▼
  match_features(...)            ◄── YOUR CODE (feature_matching.py)
    loops over descriptor pairs
    L2 distance (math.dist allowed)  →  nearest two  →  Lowe's Ratio Test
        │
        ▼  list of (reference_index, frame_index)
  index → keypoint coordinates          ┐
  Homography + RANSAC (≥ 4 matches)     │ FIXED infrastructure,
  inlier check + corner projection      │ you do not implement it
  match lines + quadrilateral drawing   ┘
```

## ✨ Features

- Real-time webcam matching with PyQt6, plus offline **image** and **video** scene sources.
- One combined canvas: **Reference Image** on the left, **Current Scene / Matching View** on the right, with lines connecting accepted correspondences.
- **Ratio Threshold** and **Max Keypoints** sliders with visible values; changes take effect immediately.
- `Capture Reference`, `Load Reference Image`, and `Clear Reference`.
- Live metrics: reference/frame keypoints, accepted matches, RANSAC inliers, localization status, matching time, processing time, FPS.
- Dark mode and light mode, full-screen mode, responsive scaling, High-DPI support.
- Pause / resume, camera-index selection, source restart.
- Settings are restored automatically between sessions; reset-to-default button.
- Safe error handling: no camera, bad files, empty matches, student exceptions.

## 🛠️ Setup & Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (recommended):

```
python -m venv venv
```

Activate it on Linux/macOS:

```
source venv/bin/activate
```

On Windows PowerShell:

```
venv\Scripts\Activate.ps1
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

`opencv-python-headless` is used on purpose: it avoids Qt plugin conflicts between OpenCV and PyQt6. SIFT requires OpenCV 4.4 or newer.

4. Run the application:

```
python main.py
```

## 🎮 Using the Application

### Scene source

- **Webcam** is the default. If no camera can be opened the app keeps running and shows `Camera could not be opened`; change the camera index or use another source.
- **Open Scene Image/Video…** loads a still image or a video file. A still image is processed again only when something changes. When a video ends, the last frame stays on screen; press `Resume` (or `Restart Source`) to replay it.

### Reference image

- **Capture Reference** (`C`) uses the currently displayed scene frame. Pause the webcam first if you want a steady shot.
- **Load Reference Image…** opens a separate image file.
- **Clear Reference** removes the reference and its cached features.
- The reference stays fixed until you replace or clear it. `Reset Defaults` restores parameters and settings but keeps the reference.
- Good references are flat, well-textured objects. Plain or repetitive objects produce few unambiguous descriptors.

### Controls

- **Ratio Threshold** (0.10 – 0.99): a match is accepted when `d1 < ratio × d2`. Higher means looser (usually more, but more ambiguous, matches); lower means stricter. Changing it only re-runs your function on the cached descriptors; SIFT is not recomputed.
- **Max Keypoints** (20 – 500): upper limit for SIFT keypoints. Changing it recomputes reference and frame features. Your loops cost about `N_reference × N_frame × 128` steps, so a high value lowers the frame rate. The default is 100.
- **Show keypoints** only toggles the drawing of small keypoint dots.

### Reading the visualization

- Each line connects a reference keypoint to the frame keypoint of an accepted match. At most 150 lines are drawn; the real match list is never truncated.
- When the fixed geometry stage ran: **green** lines are RANSAC inliers, **orange** lines are accepted matches that are geometrically inconsistent. Otherwise lines are light blue.
- The quadrilateral drawn over the current scene is the projected outline of the reference. It appears only when the fixed checks pass; the badge shows `DETECTED` or `NOT LOCALIZED`.
- A large number of accepted matches does not by itself prove they are correct, or that the reference was found. Watch the inliers and the lines.

### The fixed Homography / RANSAC stage

After your function returns, the application converts the index pairs into keypoint coordinates and, with at least 4 matches, estimates a Homography using `cv2.findHomography(..., cv2.RANSAC, ...)`. A minimum number of RANSAC inliers and simple sanity checks on the projected corners are required before anything is drawn. The thresholds are fixed in `config.py`. **Students do not implement, tune, or edit this stage.**

## 📋 Expected behavior

- **Before the TODO is completed:** `match_features` returns `[]`. The app runs normally: it shows the webcam/scene, the reference, `No accepted matches`, and no lines or quadrilateral.
- **After the TODO is completed:** match lines appear, the accepted-match count responds to the Ratio Threshold, and (for a good reference) the quadrilateral locks onto the object.
- If your function raises an exception or returns something that is not a list of valid index pairs, the app shows a readable message in the status bar, prints the traceback in the terminal, and treats it as no matches.
- Pure-Python loops are slow (even with `math.dist`): a few frames per second is normal. Lower `Max Keypoints` while debugging.

## 📂 Repository Structure

```
cvml-lab03-image-matching/
├── .gitignore
├── Readme.md
├── requirements.txt
├── main.py
├── gui.py
├── video_thread.py
├── annotation.py
├── config.py
└── feature_matching.py      # The only file students should modify
```

### Module responsibilities

- `main.py` — application entry point only.
- `gui.py` — PyQt6 user interface, controls, themes, persistence, and shortcuts.
- `video_thread.py` — scene acquisition (webcam/image/video), SIFT extraction with cached reference features, the call to `match_features`, and the fixed Homography/RANSAC/projection stage.
- `annotation.py` — side-by-side canvas, keypoints, match lines, and quadrilateral drawing.
- `config.py` — constants, default and fixed parameters, shared data classes.
- `feature_matching.py` — the student implementation of descriptor matching.

## ⌨️ Keyboard Shortcuts

- `Space` — Pause / Resume
- `C` — Capture Reference
- `F11` — Full Screen / Windowed
- `R` — Reset Defaults
- `Q` — Quit

## About

Feature matching is the foundation of panorama stitching, visual localization, and augmented reality. In this lab you implement the core decision yourself: how similar are two descriptors, and when is a nearest neighbor trustworthy enough to accept? The rest of the application is there to let you see the answer.

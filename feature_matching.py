import math

import numpy as np


def match_features(
    reference_descriptors: np.ndarray,
    frame_descriptors: np.ndarray,
    ratio_threshold: float,
) -> list[tuple[int, int]]:
    """
    TODO: Implement manual descriptor matching with Lowe's Ratio Test.

    IMPORTANT:
    - This is the only file students should modify.
    - Keep this function signature unchanged.
    - The function receives SIFT DESCRIPTOR MATRICES only. It does not receive
      images or keypoints. SIFT extraction is done elsewhere.
    - Each ROW of a matrix is one descriptor. A standard SIFT descriptor is a
      128-dimensional float32 vector, so the matrices have shape (N, 128).
    - Return ONLY a plain list of (reference_index, frame_index) pairs.
      Do not return cv2.DMatch objects, keypoints, Homography matrices, or
      anything else. Do not draw anything. Do not touch the camera or the GUI.
    - Matching is ONE-WAY: reference descriptors -> frame descriptors.
      Do NOT use a cross-check / mutual-nearest-neighbor rule and do NOT force
      one-to-one matches. The same frame descriptor may be matched by several
      reference descriptors.

    FORBIDDEN (this is the point of the lab):
    - cv2.BFMatcher, cv2.DescriptorMatcher, .match(), .knnMatch(), or any other
      ready-made matcher or matching library.
    - np.linalg.norm, NumPy broadcasting, vectorized subtraction/sum/sqrt over
      descriptors, SciPy, or any other library/helper that computes distances
      (or finds nearest neighbors) for many descriptors at once.
    - ALLOWED: math.dist(a, b) from the standard library, to get the L2
      distance between ONE pair of descriptors. This is the only distance helper
      you may use. The loops over i and j, collecting distances in a list, sorting,
      and the ratio test must be written by you.

    Expected implementation sequence:

    1. Validate the inputs. Return [] (an empty list) when any of these holds:
        - reference_descriptors or frame_descriptors is None or empty;
        - an array is not two-dimensional (rows = descriptors, columns = values);
        - the two arrays have a different number of columns (descriptor size);
        - the values are not numeric or contain NaN / infinity;
        - frame_descriptors has fewer than 2 rows (the ratio test needs a
          nearest AND a second-nearest candidate);
        - ratio_threshold is not a number with 0 < ratio_threshold < 1.
        An empty result is a normal answer, never an exception.

    2. Calculate L2 (Euclidean) distances.
        For every reference descriptor i:
            - For every frame descriptor j, compute `d_ij = math.dist(reference_row, frame_row)`.
            - Store each distance along with its frame index as a pair (e.g., `(d_ij, j)`) in a list.

    3. Find the two nearest neighbors of reference descriptor i.
        - Sort your list of distances in ascending order (e.g., using `.sort()`).
        - The first element of the sorted list gives the smallest distance (`d1`) and its frame index (`j1`),
          and the second element gives the second-smallest distance (`d2`).

    4. Apply Lowe's Ratio Test with the STRICT inequality:
              d1 < ratio_threshold * d2
        If it holds, the closest candidate is unambiguous enough: accept it and
        append (i, j1) to your result list. Otherwise reject reference
        descriptor i. Do not replace "<" with "<=".

    5. Return the list of accepted (reference_index, frame_index) pairs.
        Indices are zero-based row numbers; they must stay aligned with the
        keypoint lists that the application keeps for you.

    Tips:
    - Speed matters a little: sorting the list per reference descriptor adds a small overhead ($N \log N$).
      Lower "Max Keypoints" in the GUI while you debug if needed.
    - Optional speed-up that is allowed: convert each matrix ONCE with
      .tolist() before the loops (plain Python lists index faster than NumPy
      arrays). This only changes the container, not the distance computation.
    - You are expected to be able to explain every line during the oral exam,
      so prefer simple, readable loops over clever tricks.
    """
    # TODO: Delete this placeholder and return your own list of accepted matches.
    # Until then the application runs normally and simply shows no matches.
    return []
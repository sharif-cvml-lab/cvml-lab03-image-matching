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
      you may use. The loops over i and j, the search for the nearest and
      second-nearest neighbors, and the ratio test must be written by you.

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
           for every frame descriptor j:
               d_ij = distance between descriptor i and descriptor j
       where   d_ij = sqrt( sum_k ( ref[i, k] - frame[j, k] ) ** 2 )
       You may compute each d_ij with math.dist(reference_row, frame_row), one
       pair of descriptors at a time. Keep track of which frame index j produced
       each distance, so that you know WHICH frame descriptor is the closest
       one, not just how far it is.

    3. Find the two nearest neighbors of reference descriptor i.
       While scanning j, remember:
           d1 = the smallest distance so far and j1 = the frame index that gave it
           d2 = the second-smallest distance so far
       Use simple if/elif logic: a new distance smaller than d1 pushes the old
       d1 down to become d2; otherwise a distance smaller than d2 only replaces
       d2. Start d1 and d2 with a very large value (for example float("inf")).

    4. Apply Lowe's Ratio Test with the STRICT inequality:
           d1 < ratio_threshold * d2
       If it holds, the closest candidate is unambiguous enough: accept it and
       append (i, j1) to your result list. Otherwise reject reference
       descriptor i. Do not replace "<" with "<=".

    5. Return the list of accepted (reference_index, frame_index) pairs.
       Indices are zero-based row numbers; they must stay aligned with the
       keypoint lists that the application keeps for you.

    Tips:
    - Speed matters a little: the work grows like N_reference * N_frame * 128.
      Lower "Max Keypoints" in the GUI while you debug.
    - Optional speed-up that is allowed: convert each matrix ONCE with
      .tolist() before the loops (plain Python lists index faster than NumPy
      arrays). This only changes the container, not the distance computation.
    - You are expected to be able to explain every line during the oral exam,
      so prefer simple, readable loops over clever tricks.
    """
    # TODO: Delete this placeholder and return your own list of accepted matches.
    # Until then the application runs normally and simply shows no matches.
    return []

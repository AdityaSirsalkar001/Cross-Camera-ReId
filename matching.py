import math

import cv2
import numpy as np

from config import (
    MAX_GROUND_ERROR_FRAC,
    MIN_APPEARANCE_SAMPLES,
    MIN_OVERLAP_SAMPLES,
    W_A,
    W_S,
    W_T,
)


def appearance_sim(t1, t2):
    """Compare accumulated OSNet, clothing-colour, and body-shape profiles."""
    if (
        t1.emb is None or t2.emb is None
        or t1.shape is None or t2.shape is None
        or t1.appearance_samples < MIN_APPEARANCE_SAMPLES
        or t2.appearance_samples < MIN_APPEARANCE_SAMPLES
    ):
        return None

    reid = float(np.clip(np.dot(t1.emb, t2.emb), 0.0, 1.0))
    color = float(np.mean([
        1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
        for a, b in zip(t1.hist, t2.hist)
    ]))
    hog = float(np.clip(np.dot(t1.shape, t2.shape), 0.0, 1.0))
    ratio = math.exp(
        -abs(math.log(max(t1.shape_ratio, 1e-6) / max(t2.shape_ratio, 1e-6))) / 0.45
    )
    body_shape = 0.75 * hog + 0.25 * ratio
    if t1.cls == 1:  # OSNet was trained for people, so rely more on robot colour/shape.
        return 0.20 * reid + 0.62 * color + 0.18 * body_shape
    return 0.75 * reid + 0.18 * color + 0.07 * body_shape


def tracklet_score(t1, t2, mapper, frame_diagonal):
    """Score two complete tracklets using ReID and synchronized floor positions."""
    if t1.cls != t2.cls:
        return None, "different classes"
    app = appearance_sim(t1, t2)
    if app is None:
        return None, "too few appearance samples"

    common_frames = sorted(set(t1.trajectory).intersection(t2.trajectory))
    if len(common_frames) < MIN_OVERLAP_SAMPLES:
        return None, "insufficient synchronized overlap"

    distances = []
    for frame in common_frames:
        projected = mapper.project(t2.trajectory[frame])
        point1 = t1.trajectory[frame]
        distances.append(math.hypot(projected[0] - point1[0], projected[1] - point1[1]))

    median_error = float(np.median(distances))
    error_limit = MAX_GROUND_ERROR_FRAC * frame_diagonal
    if median_error > error_limit:
        return None, f"floor mismatch ({median_error:.0f}px > {error_limit:.0f}px)"

    spatial = math.exp(-median_error / max(error_limit * 0.45, 1.0))
    overlap = len(common_frames) / max(min(len(t1.trajectory), len(t2.trajectory)), 1)
    score = W_A * app + W_S * spatial + W_T * min(overlap, 1.0)
    return (score, app, median_error, len(common_frames)), None

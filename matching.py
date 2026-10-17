import math
import numpy as np
import cv2
from config import DEEP_LOW, DEEP_HIGH, W_DEEP, W_COLOR, MOVE_THR, POS_SIGMA, SPATIAL_GATE, GATE_FRAC, ONSET_TAU_SEC, W_A, W_M, W_T
from homography import GroundMapper

def appearance_sim(t1, t2):
    if t1.emb is None or t2.emb is None:
        return 0.0
    cos = float(np.dot(t1.emb, t2.emb))
    deep = min(max((cos - DEEP_LOW) / (DEEP_HIGH - DEEP_LOW), 0.0), 1.0)
    col = np.mean([1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
                   for a, b in zip(t1.hist, t2.hist)])
    return W_DEEP * deep + W_COLOR * float(col)


def speed_series_corr(t1, t2):
    """Do both tracks speed up / slow down at the same times? (None if uninformative)"""
    d1, d2 = dict(t1.speeds), dict(t2.speeds)
    common = sorted(set(d1) & set(d2))
    if len(common) < 10:
        return None
    a = np.array([d1[f] for f in common])
    b = np.array([d2[f] for f in common])
    fa, fb = a.std() < 0.05, b.std() < 0.05
    if fa and fb:
        return None
    if fa != fb:
        return 0.3
    c = np.corrcoef(a, b)[0, 1]
    return 0.5 if np.isnan(c) else (c + 1.0) / 2.0


def spatial_dist(t1, t2, mapper, diag):
    """Distance (fraction of diagonal) between cam1 foot point and projected cam2 foot point."""
    if not mapper.ready:
        return None
    px, py = mapper.project(t2.foot())
    fx, fy = t1.foot()
    return math.hypot(px - fx, py - fy) / diag


def motion_sim(t1, t2, mapper, diag):
    s1, s2 = t1.cur_speed(), t2.cur_speed()
    parts = [(math.exp(-abs(s1 - s2) / (max(s1, s2) + 0.3)), 0.30)]   # speed agreement

    corr = speed_series_corr(t1, t2)                                  # temporal motion sync
    if corr is not None:
        parts.append((corr, 0.20))

    if mapper.ready:
        d = spatial_dist(t1, t2, mapper, diag)                        # trajectory position
        parts.append((math.exp(-(d / POS_SIGMA) ** 2), 0.30))

        dp1, dp2 = t1.displacement_pts(), t2.displacement_pts()       # trajectory direction
        direction = 0.5
        if dp1 and dp2 and s1 > MOVE_THR and s2 > MOVE_THR:
            v1 = np.array(dp1[1]) - np.array(dp1[0])
            a0, a1 = mapper.project(dp2[0]), mapper.project(dp2[1])
            v2 = np.array(a1) - np.array(a0)
            n = np.linalg.norm(v1) * np.linalg.norm(v2)
            if n > 1e-6:
                direction = (float(np.dot(v1, v2) / n) + 1.0) / 2.0
        parts.append((direction, 0.20))

    tw = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / tw


def temporal_sim(t1, t2, fps):
    overlap = max(0, min(t1.last, t2.last) - max(t1.first, t2.first) + 1)
    dur = min(t1.last - t1.first + 1, t2.last - t2.first + 1)
    overlap_ratio = overlap / max(dur, 1)
    onset = math.exp(-abs(t1.first - t2.first) / (ONSET_TAU_SEC * fps))
    return 0.5 * overlap_ratio + 0.5 * onset


def fused_score(t1, t2, mapper, diag, fps):
    """Returns fused score or None if a consistency check rejects the pair."""
    if t1.cls != t2.cls:
        return None
    if SPATIAL_GATE and mapper.ready:
        if spatial_dist(t1, t2, mapper, diag) > GATE_FRAC:
            return None
    sa = appearance_sim(t1, t2)
    sm = motion_sim(t1, t2, mapper, diag)
    st = temporal_sim(t1, t2, fps)
    return W_A * sa + W_M * sm + W_T * st

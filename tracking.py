import math
from collections import deque
import cv2
import numpy as np
from config import EMBED_EVERY, EMB_EMA, POS_HISTORY, SPEED_HISTORY, SPEED_WIN, MAX_LOST, MIN_TRACK_AGE, MIN_BOX, TRACKER_CFG, TARGET_CLASSES
from features import deep_embedding, color_hist

class Track:
    def __init__(self, cam, tid, cls, frame_idx):
        self.cam, self.tid, self.cls = cam, tid, cls
        self.first = self.last = frame_idx
        self.n_obs = 0
        self.box = None
        self.emb = None
        self.hist = None
        self.pos = deque(maxlen=POS_HISTORY)        # (frame, foot_x, foot_y, height)
        self.speeds = deque(maxlen=SPEED_HISTORY)   # (frame, speed in body-heights/sec)

    # ---------------- update ----------------
    def update(self, frame, frame_idx, box, fps):
        self.last = frame_idx
        self.box = box
        self.n_obs += 1
        x1, y1, x2, y2 = box
        self.pos.append((frame_idx, (x1 + x2) / 2.0, float(y2), float(y2 - y1)))
        self._update_speed(fps)
        if self.emb is None or self.n_obs % EMBED_EVERY == 0:
            self._update_appearance(frame, box)

    def _ref_point(self, win):
        if len(self.pos) < 2:
            return None
        target = self.pos[-1][0] - win
        for p in reversed(self.pos):
            if p[0] <= target:
                return p
        return self.pos[0] if self.pos[-1][0] - self.pos[0][0] >= 2 else None

    def _update_speed(self, fps):
        ref = self._ref_point(SPEED_WIN)
        if ref is None:
            return
        f0, x0, y0, h0 = ref
        f, x, y, h = self.pos[-1]
        dt = max((f - f0) / fps, 1e-6)
        mean_h = max((h + h0) / 2.0, 1.0)
        self.speeds.append((f, math.hypot(x - x0, y - y0) / mean_h / dt))

    def _update_appearance(self, frame, box):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        e = deep_embedding(crop)
        ch = color_hist(crop)
        if self.emb is None:
            self.emb, self.hist = e, ch
        else:
            m = (1 - EMB_EMA) * self.emb + EMB_EMA * e
            self.emb = m / (np.linalg.norm(m) + 1e-8)
            self.hist = [(1 - EMB_EMA) * a + EMB_EMA * b for a, b in zip(self.hist, ch)]

    # ---------------- accessors ----------------
    def cur_speed(self):
        if not self.speeds:
            return 0.0
        return float(np.mean([s for _, s in list(self.speeds)[-5:]]))

    def displacement_pts(self):
        """(ref_foot_point, current_foot_point) or None."""
        ref = self._ref_point(SPEED_WIN)
        if ref is None:
            return None
        return (ref[1], ref[2]), (self.pos[-1][1], self.pos[-1][2])

    def foot(self):
        return self.pos[-1][1], self.pos[-1][2]


class TrackManager:
    def __init__(self, cam):
        self.cam = cam
        self.tracks = {}

    def update(self, frame, frame_idx, dets, fps):
        for d in dets:
            tr = self.tracks.get(d["id"])
            if tr is None:
                tr = Track(self.cam, d["id"], d["cls"], frame_idx)
                self.tracks[d["id"]] = tr
            tr.update(frame, frame_idx, d["box"], fps)
        for tid in [t for t, tr in self.tracks.items() if frame_idx - tr.last > MAX_LOST]:
            del self.tracks[tid]

    def active(self, frame_idx):
        return [t for t in self.tracks.values() if t.last == frame_idx and t.n_obs >= MIN_TRACK_AGE]


def detect_and_track(model, frame):
    """YOLOv8 + ByteTrack. Returns list of {id, box, cls}."""
    res = model.track(frame, persist=True, tracker=TRACKER_CFG,
                      classes=TARGET_CLASSES, verbose=False)[0]
    out = []
    if res.boxes is None or res.boxes.id is None:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    ids = res.boxes.id.int().cpu().tolist()
    clss = res.boxes.cls.int().cpu().tolist()
    H, W = frame.shape[:2]
    for b, i, c in zip(xyxy, ids, clss):
        x1, y1, x2, y2 = map(int, b[:4])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        if (x2 - x1) > MIN_BOX and (y2 - y1) > MIN_BOX:
            out.append({"id": i, "box": (x1, y1, x2, y2), "cls": c})
    return out

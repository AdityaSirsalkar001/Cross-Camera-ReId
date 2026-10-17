"""
demo3.py - Cross-camera Re-ID using Multi-Feature Fusion
(appearance + motion/trajectory + temporal consistency)

Same I/O as demo1.py:
    input : cam1.mp4, cam2.mp4   (same scene, different angles, time-synchronised)
    output: reid_output.mp4      (side-by-side, same object -> same ID & colour)

Pipeline (per frame):
    1. YOLOv8 + ByteTrack  -> stable LOCAL track IDs inside each camera
    2. Per-track features:
         - Appearance : ResNet50 embedding (EMA) + HSV upper/lower-body colour histogram
         - Motion     : body-height-normalised speed, speed-series correlation,
                        direction + position consistency (via homography cam2 -> cam1)
         - Temporal   : lifetime overlap + onset closeness of the two tracks
    3. Fused score  S = Wa*Sa + Wm*Sm + Wt*St   (smoothed over time with EMA)
    4. Hungarian one-to-one assignment + threshold + consistency checks
    5. A pair must match for CONFIRM_FRAMES consecutive frames -> GLOBAL ID is
       assigned and then kept stable for the rest of the video.
"""

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO

# =====================================================================
# CONFIG  (tune these)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
CAM1_PATH = str(BASE_DIR / "cam1.mp4")
CAM2_PATH = str(BASE_DIR / "cam2.mp4")
OUT_PATH = str(BASE_DIR / "reid_output.mp4")

CAM2_OFFSET_FRAMES = 0      # >0: cam2 started earlier (skip its first N frames); <0: skip cam1's first N
TARGET_CLASSES = None       # None = all classes (like demo1). e.g. [0] for persons only
TRACKER_CFG = "bytetrack.yaml"
MIN_BOX = 20                # filter tiny boxes (same as demo1)

# fusion weights  (should sum to 1)
W_A, W_M, W_T = 0.50, 0.35, 0.15
THRESHOLD = 0.60            # min fused (smoothed) score to accept a match
LOCK_BONUS = 0.10           # hysteresis: already-linked pairs are preferred
SCORE_EMA = 0.20            # smoothing of pair score over time
CONFIRM_FRAMES = 3          # consecutive matched frames before a global ID is given
MIN_TRACK_AGE = 5           # a track needs this many observations before it can be matched
MAX_LOST = 90               # forget local tracks unseen for this many frames

# appearance
EMBED_EVERY = 3             # recompute embedding every N observations (speed)
EMB_EMA = 0.30
DEEP_LOW, DEEP_HIGH = 0.55, 0.95   # cosine range rescaled to [0,1] (ResNet cosines are all high)
W_DEEP, W_COLOR = 0.6, 0.4

# motion
SPEED_WIN = 5               # frames used for velocity estimate
POS_HISTORY = 60
SPEED_HISTORY = 45
MOVE_THR = 0.15             # body-heights/sec below this = "stationary"
POS_SIGMA = 0.15            # (fraction of frame diagonal) tolerance for projected position

# homography (cam2 -> cam1) used for position/direction consistency
USE_HOMOGRAPHY = True
H_INTERVAL = 30             # re-estimate every N frames
H_MIN_INLIERS = 20
SPATIAL_GATE = True         # reject pairs whose projected positions are far apart
GATE_FRAC = 0.40            # fraction of frame diagonal

# temporal
ONSET_TAU_SEC = 2.0

# =====================================================================
# 1. Vision models
# =====================================================================
print("Loading vision models...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# one YOLO instance per camera (tracker state is kept inside the model)
yolo_cam1 = YOLO("yolov8n.pt")
yolo_cam2 = YOLO("yolov8n.pt")

resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.eval()
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(DEVICE)

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Distinct colors for different matched pairs (BGR format)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
]


# =====================================================================
# 2. Feature helpers
# =====================================================================
def deep_embedding(crop_bgr):
    """L2-normalised ResNet50 embedding of a BGR crop."""
    tensor = preprocess(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        e = resnet(tensor).flatten().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)


def color_hist(crop_bgr):
    """HSV hist (H,S) of upper-body and lower-body regions -> list of 2 arrays."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = hsv.shape[0]
    parts = [hsv[int(0.15 * h):int(0.50 * h)], hsv[int(0.50 * h):int(0.90 * h)]]
    out = []
    for p in parts:
        hist = cv2.calcHist([p], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        out.append(hist)
    return out


# =====================================================================
# 3. Track state (per camera)
# =====================================================================
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


# =====================================================================
# 4. Homography cam2 -> cam1 (static background features, RANSAC)
# =====================================================================
class GroundMapper:
    def __init__(self):
        self.H = None
        self.inliers = 0
        try:
            self.det = cv2.SIFT_create(2000)
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        except AttributeError:
            self.det = cv2.ORB_create(3000)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    @property
    def ready(self):
        return USE_HOMOGRAPHY and self.H is not None

    def update(self, f1, f2, boxes1, boxes2, frame_idx):
        if not USE_HOMOGRAPHY or (frame_idx % H_INTERVAL != 0 and self.H is not None):
            return
        g1, g2 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        m1 = np.full(g1.shape, 255, np.uint8)
        m2 = np.full(g2.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes1:      # ignore moving objects -> use background only
            m1[y1:y2, x1:x2] = 0
        for x1, y1, x2, y2 in boxes2:
            m2[y1:y2, x1:x2] = 0
        kp1, d1 = self.det.detectAndCompute(g1, m1)
        kp2, d2 = self.det.detectAndCompute(g2, m2)
        if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
            return
        good = []
        for pair in self.bf.knnMatch(d2, d1, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < H_MIN_INLIERS:
            return
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return
        inl = int(mask.sum())
        if inl >= H_MIN_INLIERS:
            self.H, self.inliers = H, inl

    def project(self, pt):
        p = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), self.H)[0, 0]
        return float(p[0]), float(p[1])


# =====================================================================
# 5. Similarity scores
# =====================================================================
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


# =====================================================================
# 6. Global ID manager
# =====================================================================
class GlobalIDs:
    def __init__(self):
        self.of = {1: {}, 2: {}}    # cam -> {local_id: global_id}
        self.pair = {}              # global_id -> {1: local_id, 2: local_id}
        self.next_id = 1

    def get(self, cam, tid):
        return self.of[cam].get(tid)

    def allowed(self, t1, t2, act1, act2):
        g1, g2 = self.get(1, t1), self.get(2, t2)
        if g1 is not None and g2 is not None:
            return g1 == g2
        if g1 is not None:
            o = self.pair[g1].get(2)
            return o is None or o == t2 or o not in act2
        if g2 is not None:
            o = self.pair[g2].get(1)
            return o is None or o == t1 or o not in act1
        return True

    def link(self, t1, t2):
        g1, g2 = self.get(1, t1), self.get(2, t2)
        if g1 is not None and g1 == g2:
            return
        if g1 is not None:
            self.of[2][t2] = g1
            self.pair[g1][2] = t2
        elif g2 is not None:
            self.of[1][t1] = g2
            self.pair[g2][1] = t1
        else:
            g = self.next_id
            self.next_id += 1
            self.of[1][t1] = g
            self.of[2][t2] = g
            self.pair[g] = {1: t1, 2: t2}


def draw(frame, box, gid, text):
    color = COLORS[(gid - 1) % len(COLORS)]
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# =====================================================================
# 7. Open input video streams  (same as demo1)
# =====================================================================
cap1 = cv2.VideoCapture(CAM1_PATH)
cap2 = cv2.VideoCapture(CAM2_PATH)

width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS)) or 25
diag = math.hypot(width, height)

# temporal alignment of the two videos (if they don't start together)
for _ in range(max(CAM2_OFFSET_FRAMES, 0)):
    cap2.read()
for _ in range(max(-CAM2_OFFSET_FRAMES, 0)):
    cap1.read()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width * 2, height))

mgr1, mgr2 = TrackManager(1), TrackManager(2)
mapper = GroundMapper()
gids = GlobalIDs()
pair_ema = {}          # (tid1, tid2) -> smoothed fused score
pair_streak = {}       # (tid1, tid2) -> consecutive matched frames

print("Processing video streams...")
frame_idx = 0
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    frame2 = cv2.resize(frame2, (width, height))

    # --- local detection + tracking, feature update (uses clean frames) ---
    dets1 = detect_and_track(yolo_cam1, frame1)
    dets2 = detect_and_track(yolo_cam2, frame2)
    mgr1.update(frame1, frame_idx, dets1, fps)
    mgr2.update(frame2, frame_idx, dets2, fps)
    mapper.update(frame1, frame2, [d["box"] for d in dets1], [d["box"] for d in dets2], frame_idx)

    act1 = mgr1.active(frame_idx)
    act2 = mgr2.active(frame_idx)
    ids1, ids2 = {t.tid for t in act1}, {t.tid for t in act2}

    # --- pairwise fused scores ---
    score = np.full((len(act1), len(act2)), -1.0)
    for i, a in enumerate(act1):
        for j, b in enumerate(act2):
            s = fused_score(a, b, mapper, diag, fps)
            if s is None:
                continue
            key = (a.tid, b.tid)
            prev = pair_ema.get(key)
            pair_ema[key] = s if prev is None else (1 - SCORE_EMA) * prev + SCORE_EMA * s
            if not gids.allowed(a.tid, b.tid, ids1, ids2):
                continue
            m = pair_ema[key]
            if gids.get(1, a.tid) is not None and gids.get(1, a.tid) == gids.get(2, b.tid):
                m += LOCK_BONUS
            score[i, j] = m

    # --- one-to-one assignment (Hungarian) + threshold ---
    matched = {}   # cam1 tid -> (cam2 tid, score)
    new_streak = {}
    if score.size:
        rows, cols = linear_sum_assignment(-score)
        for r, c in zip(rows, cols):
            if score[r, c] >= THRESHOLD:
                key = (act1[r].tid, act2[c].tid)
                new_streak[key] = pair_streak.get(key, 0) + 1
                matched[key[0]] = (key[1], score[r, c])
                if new_streak[key] >= CONFIRM_FRAMES:
                    gids.link(*key)
    pair_streak = new_streak

    # --- draw (same output style as demo1) ---
    for t in act1:
        g = gids.get(1, t.tid)
        if g is None:
            continue
        txt = f"ID:{g}"
        if t.tid in matched:
            txt += f" ({min(matched[t.tid][1], 1.0):.2f})"
        draw(frame1, t.box, g, txt)
    rev = {c: s for (_, (c, s)) in matched.items()}
    for t in act2:
        g = gids.get(2, t.tid)
        if g is None:
            continue
        txt = f"ID:{g}"
        if t.tid in rev:
            txt += f" ({min(rev[t.tid], 1.0):.2f})"
        draw(frame2, t.box, g, txt)

    # housekeeping
    if frame_idx % 150 == 0:
        pair_ema = {k: v for k, v in pair_ema.items()
                    if k[0] in mgr1.tracks and k[1] in mgr2.tracks}

    combined_frame = cv2.hconcat([frame1, frame2])
    out.write(combined_frame)

    cv2.imshow("Cross-Camera Re-ID Stream", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! {gids.next_id - 1} global IDs assigned. Output video saved as '{OUT_PATH}'.")
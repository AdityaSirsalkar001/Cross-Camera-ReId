import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import CAM1_PATH, CAM2_PATH, OUT_PATH, CAM2_OFFSET_FRAMES, SCORE_EMA, THRESHOLD, CONFIRM_FRAMES, LOCK_BONUS
from models import yolo_cam1, yolo_cam2
from tracking import TrackManager, detect_and_track
from homography import GroundMapper
from matching import fused_score
from global_ids import GlobalIDs, draw

cap1 = cv2.VideoCapture(CAM1_PATH)
cap2 = cv2.VideoCapture(CAM2_PATH)

if not cap1.isOpened():
    raise FileNotFoundError(f"[ERROR] Could not open Camera 1 video stream at: '{CAM1_PATH}'. Please place 'cam1.mp4' in the project directory or update config.py.")
if not cap2.isOpened():
    raise FileNotFoundError(f"[ERROR] Could not open Camera 2 video stream at: '{CAM2_PATH}'. Please place 'cam2.mp4' in the project directory or update config.py.")

width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS)) or 25
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
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

    if frame_idx % 30 == 0:
        total = total_frames if total_frames > 0 else 1334
        print(f"Processing frame {frame_idx}/{total} ({(frame_idx/total*100):.1f}%) | Active: Cam1={len(act1)}, Cam2={len(act2)} | Global IDs: {gids.next_id - 1}")

    # housekeeping
    if frame_idx % 150 == 0:
        pair_ema = {k: v for k, v in pair_ema.items()
                    if k[0] in mgr1.tracks and k[1] in mgr2.tracks}

    combined_frame = cv2.hconcat([frame1, frame2])
    out.write(combined_frame)

    try:
        cv2.imshow("Cross-Camera Re-ID Stream", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested early termination via 'q' key.")
            break
    except Exception:
        pass
    frame_idx += 1

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! {gids.next_id - 1} global IDs assigned. Output video saved as '{OUT_PATH}'.")

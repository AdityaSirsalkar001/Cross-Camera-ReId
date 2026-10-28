import math
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    CAM1_PATH,
    CAM2_PATH,
    OUT_PATH,
    CALIBRATION_PATH,
    CAM2_OFFSET_FRAMES,
    MIN_TRACK_AGE,
    MIN_APPEARANCE_SAMPLES,
    MIN_REID_SIM,
    MIN_PAIR_SCORE,
    MATCH_MARGIN,
    PREVIEW_MAX_WIDTH,
    PREVIEW_MAX_HEIGHT,
)
from homography import GroundMapper


def fit_preview(frame):
    """Resize only the display copy; saved output keeps its original resolution."""
    h, w = frame.shape[:2]
    scale = min(PREVIEW_MAX_WIDTH / w, PREVIEW_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        return cv2.resize(
            frame,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def preview_size(width, height):
    scale = min(PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / (height * 2), 1.0)
    return round(width * scale), round(height * 2 * scale)


def margin_ok(matrix, row, col, margin):
    row_other = np.delete(matrix[row, :], col)
    col_other = np.delete(matrix[:, col], row)
    row_best = float(row_other.max()) if row_other.size else -1.0
    col_best = float(col_other.max()) if col_other.size else -1.0
    return matrix[row, col] - row_best >= margin and matrix[row, col] - col_best >= margin


def main():
    mapper = GroundMapper(CALIBRATION_PATH)

    # Model imports are deferred so missing calibration is reported first.
    from models import yolo_world_cam1, yolo_world_cam2
    from tracking import TrackManager, detect_and_track
    from matching import tracklet_score
    from global_ids import GlobalIDs, draw, draw_local

    cap1 = cv2.VideoCapture(CAM1_PATH)
    cap2 = cv2.VideoCapture(CAM2_PATH)
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError(f"Could not open both inputs: {CAM1_PATH} and {CAM2_PATH}")

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam2_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam2_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS) or 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {CAM1_PATH}")
    if mapper.cam1_size and mapper.cam1_size != (width, height):
        raise RuntimeError(
            f"Calibration was made for Camera 1 size {mapper.cam1_size}, but the video is {(width, height)}. "
            "Run calibrate_cameras.py again."
        )
    if mapper.cam2_size and mapper.cam2_size != (cam2_width, cam2_height):
        raise RuntimeError(
            f"Calibration was made for Camera 2 size {mapper.cam2_size}, but the video is "
            f"{(cam2_width, cam2_height)}. Run calibrate_cameras.py again."
        )
    frame_diagonal = math.hypot(width, height)

    for _ in range(max(CAM2_OFFSET_FRAMES, 0)):
        cap2.read()
    for _ in range(max(-CAM2_OFFSET_FRAMES, 0)):
        cap1.read()

    managers = {1: TrackManager(1), 2: TrackManager(2)}
    frame_detections = []
    print("Pass 1/2: YOLO-World detecting people/robots and building camera-local tracks...")
    preview_window = "Cross-Camera Re-ID | Camera 1 over Camera 2"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window, *preview_size(width, height))
    frame_idx = 0
    while True:
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        if not ok1 or not ok2:
            break
        dets1 = detect_and_track(yolo_world_cam1, frame1)
        dets2 = detect_and_track(yolo_world_cam2, frame2)
        managers[1].update(frame1, frame_idx, dets1, fps)
        managers[2].update(frame2, frame_idx, dets2, fps)
        frame_detections.append((dets1, dets2))

        # Show each newly processed pair immediately. Global IDs are decided
        # after all tracklets are available, so this first-pass view shows local IDs.
        preview1 = frame1.copy()
        preview2 = cv2.resize(frame2, (width, height))
        for cam, preview, detections in ((1, preview1, dets1), (2, preview2, dets2)):
            for detection in detections:
                box = detection["box"]
                if cam == 2:
                    sx, sy = width / cam2_width, height / cam2_height
                    box = tuple(round(v * (sx if i % 2 == 0 else sy))
                                for i, v in enumerate(box))
                label = "PERSON" if detection["cls"] == 0 else "HUMANOID ROBOT"
                draw_local(preview, box, cam, detection["id"], label)
        preview = cv2.vconcat([preview1, preview2])
        cv2.putText(
            preview,
            f"DETECTION PASS | frame {frame_idx + 1} | local IDs; global matching comes after the pass",
            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
        )
        preview = fit_preview(preview)
        cv2.imshow(preview_window, preview)
        cv2.waitKey(1)

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  Processed {frame_idx} synchronized frames")
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    tracks1 = [t for t in managers[1].tracks.values()
               if t.n_obs >= MIN_TRACK_AGE and t.appearance_samples >= MIN_APPEARANCE_SAMPLES]
    tracks2 = [t for t in managers[2].tracks.values()
               if t.n_obs >= MIN_TRACK_AGE and t.appearance_samples >= MIN_APPEARANCE_SAMPLES]
    scores = np.full((len(tracks1), len(tracks2)), -1.0, dtype=np.float32)
    reid_scores = np.full_like(scores, -1.0)
    errors = np.full_like(scores, np.inf)
    overlap_counts = np.zeros_like(scores, dtype=np.int32)
    rejections = Counter()
    valid_candidates = []

    print(f"Comparing {len(tracks1)} camera-1 and {len(tracks2)} camera-2 tracklets...")
    for i, track1 in enumerate(tracks1):
        for j, track2 in enumerate(tracks2):
            result, reason = tracklet_score(track1, track2, mapper, frame_diagonal)
            if result is None:
                if reason != "different classes":
                    rejections[reason] += 1
                continue
            score, appearance, ground_error, overlap = result
            scores[i, j] = score
            reid_scores[i, j] = appearance
            errors[i, j] = ground_error
            overlap_counts[i, j] = overlap
            valid_candidates.append((
                score, appearance, ground_error, overlap, track1.tid, track2.tid
            ))

    gids = GlobalIDs()
    if scores.size:
        rows, cols = linear_sum_assignment(-scores)
        for row, col in zip(rows, cols):
            if scores[row, col] < MIN_PAIR_SCORE or reid_scores[row, col] < MIN_REID_SIM:
                rejections["below appearance/combined-score threshold"] += 1
                continue
            if not margin_ok(scores, row, col, MATCH_MARGIN):
                rejections["ambiguous combined-score margin"] += 1
                continue
            if not margin_ok(reid_scores, row, col, MATCH_MARGIN):
                rejections["ambiguous appearance margin"] += 1
                continue
            gid = gids.link(tracks1[row].tid, tracks2[col].tid)
            if gid is not None:
                print(
                    f"Locked global ID {gid}: cam1 track {tracks1[row].tid} <-> "
                    f"cam2 track {tracks2[col].tid}; OSNet {reid_scores[row, col]:.2f}, "
                    f"floor error {errors[row, col]:.0f}px across "
                    f"{overlap_counts[row, col]} synchronized samples"
                )

    if gids.next_id == 1:
        print("No pair passed the ReID, floor-position, and ambiguity checks; output will show local IDs only.")
        if rejections:
            print("Pair rejection summary: " + "; ".join(
                f"{reason}: {count}" for reason, count in rejections.most_common(5)
            ))
        if valid_candidates:
            print("Best candidates that passed the synchronized floor-position check:")
            for score, appearance, error, overlap, tid1, tid2 in sorted(
                valid_candidates, reverse=True
            )[:5]:
                print(
                    f"  cam1 track {tid1} <-> cam2 track {tid2}: "
                    f"score {score:.2f}, appearance {appearance:.2f}, "
                    f"floor error {error:.0f}px, overlap {overlap} samples"
                )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width, height * 2))
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {OUT_PATH}")

    cap1 = cv2.VideoCapture(CAM1_PATH)
    cap2 = cv2.VideoCapture(CAM2_PATH)
    for _ in range(max(CAM2_OFFSET_FRAMES, 0)):
        cap2.read()
    for _ in range(max(-CAM2_OFFSET_FRAMES, 0)):
        cap1.read()

    print("Pass 2/2: writing vertically stacked output with locked identities...")
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window, *preview_size(width, height))
    for idx, (dets1, dets2) in enumerate(frame_detections):
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        if not ok1 or not ok2:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for cam, frame, detections in ((1, frame1, dets1), (2, frame2, dets2)):
            for detection in detections:
                tid = detection["id"]
                gid = gids.get(cam, tid)
                box = detection["box"]
                if cam == 2:
                    sx, sy = width / cam2_width, height / cam2_height
                    box = tuple(round(v * (sx if i % 2 == 0 else sy))
                                for i, v in enumerate(box))
                label = "PERSON" if detection["cls"] == 0 else "HUMANOID ROBOT"
                if gid is None:
                    draw_local(frame, box, cam, tid, label)
                else:
                    draw(frame, box, gid, f"{label} | GLOBAL ID:{gid}")
        combined = cv2.vconcat([frame1, frame2])
        out.write(combined)
        preview = fit_preview(combined)
        cv2.imshow(preview_window, preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if (idx + 1) % 300 == 0:
            print(f"  Rendered {idx + 1}/{len(frame_detections)} frames")

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished: {gids.next_id - 1} cross-camera identities. Output: {OUT_PATH}")


if __name__ == "__main__":
    main()

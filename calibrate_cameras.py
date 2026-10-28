import json

import cv2
import numpy as np

from config import CALIBRATION_PATH, CAM1_PATH, CAM2_PATH


def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame from {path}")
    return frame


frame1 = read_first_frame(CAM1_PATH)
frame2 = read_first_frame(CAM2_PATH)
display_width = 640
display_height = round(frame1.shape[0] * display_width / frame1.shape[1])
view1 = cv2.resize(frame1, (display_width, display_height))
view2 = cv2.resize(frame2, (display_width, display_height))
points1, points2 = [], []
canvas = None


def redraw():
    global canvas
    left, right = view1.copy(), view2.copy()
    for i, (x, y) in enumerate(points1):
        p = (round(x * display_width / frame1.shape[1]),
             round(y * display_height / frame1.shape[0]))
        cv2.circle(left, p, 6, (0, 255, 255), -1)
        cv2.putText(left, str(i + 1), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for i, (x, y) in enumerate(points2):
        p = (round(x * display_width / frame2.shape[1]),
             round(y * display_height / frame2.shape[0]))
        cv2.circle(right, p, 6, (0, 255, 255), -1)
        cv2.putText(right, str(i + 1), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(left, f"CAMERA 1  points: {len(points1)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(right, f"CAMERA 2  points: {len(points2)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    canvas = cv2.hconcat([left, right])
    cv2.putText(canvas, "Click matching floor points in both views | Enter=save (4+ pairs) | r=reset | q=quit",
                (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 0), 2)


def on_mouse(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN or y >= display_height:
        return
    if x < display_width:
        source = frame1
        px = x * source.shape[1] / display_width
        py = y * source.shape[0] / display_height
        points1.append((float(px), float(py)))
    elif x < display_width * 2:
        source = frame2
        px = (x - display_width) * source.shape[1] / display_width
        py = y * source.shape[0] / display_height
        points2.append((float(px), float(py)))
    else:
        return
    redraw()


redraw()
cv2.namedWindow("Floor calibration: Camera 1 | Camera 2", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Floor calibration: Camera 1 | Camera 2", on_mouse)
print("Click a floor landmark in Camera 1, then the same physical landmark in Camera 2.")
print("Repeat for at least 4 pairs (8 or more are preferable). Press Enter to save, r to reset, q to quit.")
while True:
    cv2.imshow("Floor calibration: Camera 1 | Camera 2", canvas)
    key = cv2.waitKey(20) & 0xFF
    if key in (ord("q"), 27):
        cv2.destroyAllWindows()
        raise SystemExit("Calibration cancelled.")
    if key == ord("r"):
        points1.clear()
        points2.clear()
        redraw()
    if key in (10, 13) and len(points1) >= 4 and len(points1) == len(points2):
        break
cv2.destroyAllWindows()

src = np.asarray(points2, dtype=np.float32).reshape(-1, 1, 2)
dst = np.asarray(points1, dtype=np.float32).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 20.0)
if H is None or mask is None or int(mask.sum()) < 4:
    raise RuntimeError("Could not estimate a floor mapping. Re-run and click clearer shared points.")

projected = cv2.perspectiveTransform(src, H)
errors = np.linalg.norm(projected[:, 0] - dst[:, 0], axis=1)
inlier_errors = errors[mask.ravel().astype(bool)]
with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "cam2_to_cam1": H.tolist(),
        "inliers": int(mask.sum()),
        "cam1_size": [int(frame1.shape[1]), int(frame1.shape[0])],
        "cam2_size": [int(frame2.shape[1]), int(frame2.shape[0])],
    }, f, indent=2)
print(f"Saved calibration to {CALIBRATION_PATH}")
print(f"RANSAC inliers: {int(mask.sum())}/{len(points1)}; median reprojection error: {float(np.median(inlier_errors)):.1f} px")

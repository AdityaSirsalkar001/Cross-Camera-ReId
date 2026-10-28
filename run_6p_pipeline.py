"""Run the Cross-Camera-ReId pipeline on the downloaded 6p camera clips.

This standalone runner overrides paths and matching weights at runtime. It does
not modify pipeline.py or the project's existing config/calibration files.
"""

import json
import runpy
from pathlib import Path

import cv2
import config


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_6p_crosscamera"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config.CAM1_PATH = str(ROOT / "6p-c1.avi")
config.CAM2_PATH = str(ROOT / "6p-c0.avi")
config.OUT_PATH = str(OUTPUT_DIR / "reid_output_6p.mp4")
config.CALIBRATION_PATH = OUTPUT_DIR / "runtime_calibration.json"

# This pair has different dimensions from the project's existing manual
# calibration (360x288 vs 1920x1080). Use synchronized appearance/time matching
# for this run and leave the project's saved calibration untouched.
calibration = {
    "cam2_to_cam1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "inliers": 0,
    "cam1_size": [360, 288],
    "cam2_size": [360, 288],
}
config.CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

config.W_A, config.W_S, config.W_T = 0.95, 0.0, 0.05
config.MAX_GROUND_ERROR_FRAC = 1_000_000.0
config.DETECTION_IMGSZ = 640

# The pipeline's live preview is optional; keep this run usable in a
# non-interactive session while still saving its full-resolution output.
cv2.namedWindow = lambda *args, **kwargs: None
cv2.resizeWindow = lambda *args, **kwargs: None
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: -1
cv2.destroyAllWindows = lambda *args, **kwargs: None

runpy.run_path(str(ROOT / "pipeline.py"), run_name="__main__")

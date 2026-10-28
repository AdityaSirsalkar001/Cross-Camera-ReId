import json

import cv2
import numpy as np


class GroundMapper:
    """Fixed cam2-to-cam1 floor mapping from manually paired floor points."""

    def __init__(self, calibration_path):
        with open(calibration_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.H = np.asarray(data["cam2_to_cam1"], dtype=np.float64)
        self.cam1_size = tuple(data.get("cam1_size", ()))
        self.cam2_size = tuple(data.get("cam2_size", ()))
        if self.H.shape != (3, 3) or not np.isfinite(self.H).all():
            raise ValueError(f"Invalid floor homography in {calibration_path}")
        self.inliers = int(data.get("inliers", 0))

    @property
    def ready(self):
        return True

    def project(self, pt):
        p = cv2.perspectiveTransform(
            np.asarray([[pt]], dtype=np.float32), self.H.astype(np.float32)
        )[0, 0]
        return float(p[0]), float(p[1])

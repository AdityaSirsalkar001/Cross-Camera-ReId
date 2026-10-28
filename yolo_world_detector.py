"""YOLO-World open-vocabulary detection plus independent per-camera ByteTrack."""

import hashlib
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from config import (
    DETECTION_CONF,
    DETECTION_IMGSZ,
    YOLO_WORLD_CLASSES,
    YOLO_WORLD_WEIGHTS,
)

_WEIGHT_REPO = "Bingsu/yolo-world-mirror"
_WEIGHT_REVISION = "414d0eef6ce72b3375f87d9c33ebc8db1e7a97ef"
_WEIGHT_SHA256 = "7c951d3baac2de906e3deda859b2ff965350fbfa14f51b81ea52f0e3d98519c8"


def _ensure_world_weights():
    """Fetch a pinned HF mirror copy and verify it before loading the PyTorch file."""
    if YOLO_WORLD_WEIGHTS.is_file():
        digest = hashlib.sha256(YOLO_WORLD_WEIGHTS.read_bytes()).hexdigest()
        if digest != _WEIGHT_SHA256:
            raise RuntimeError(
                f"YOLO-World weights at {YOLO_WORLD_WEIGHTS} failed SHA-256 validation. "
                "Move the file aside and rerun to download a verified copy."
            )
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading YOLO-World through the Hugging Face mirror requires "
            "huggingface_hub. Install it with: python -m pip install huggingface_hub"
        ) from exc

    cache_dir = YOLO_WORLD_WEIGHTS.parent / "huggingface-cache"
    print("Downloading the pinned YOLO-World checkpoint from Hugging Face...")
    cached_file = hf_hub_download(
        repo_id=_WEIGHT_REPO,
        filename=YOLO_WORLD_WEIGHTS.name,
        revision=_WEIGHT_REVISION,
        cache_dir=str(cache_dir),
    )
    temp_file = YOLO_WORLD_WEIGHTS.with_suffix(".pt.download")
    shutil.copyfile(cached_file, temp_file)
    digest = hashlib.sha256(temp_file.read_bytes()).hexdigest()
    if digest != _WEIGHT_SHA256:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded YOLO-World checkpoint failed SHA-256 validation; "
            "it was not installed. Check the network response and try again."
        )
    temp_file.replace(YOLO_WORLD_WEIGHTS)


class YOLOWorldCore:
    """One shared, promptable detector for the person and robot classes."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        YOLO_WORLD_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        _ensure_world_weights()
        print(
            f"Loading YOLO-World ({YOLO_WORLD_WEIGHTS.name}) on {self.device}. "
            "The weights download into this project's models folder if needed."
        )
        self.model = YOLOWorld(str(YOLO_WORLD_WEIGHTS), verbose=False)
        self.model.set_classes(YOLO_WORLD_CLASSES)

    def detect(self, frame):
        result = self.model.predict(
            source=frame,
            imgsz=DETECTION_IMGSZ,
            conf=DETECTION_CONF,
            device=self.device,
            half=self.device == "cuda",
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.detach().cpu().numpy().reshape(-1, 1)
        detections = np.concatenate((xyxy, conf, cls), axis=1).astype(np.float32)
        return detections


class YOLOWorldVideoTracker:
    """Keep tracker state separate for each camera while sharing detector weights."""

    def __init__(self, core):
        self.core = core
        args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        # Keep brief occlusions from fragmenting a person's or robot's local track.
        args.track_high_thresh = 0.20
        args.track_low_thresh = 0.05
        args.new_track_thresh = 0.20
        args.track_buffer = 45
        self.tracker = BYTETracker(args)

    def detect_and_track(self, frame):
        detections = self.core.detect(frame)
        boxes = Boxes(detections, orig_shape=frame.shape[:2])
        tracked = self.tracker.update(boxes, img=frame)

        height, width = frame.shape[:2]
        output = []
        for row in tracked:
            x1, y1, x2, y2 = map(int, row[:4])
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                output.append({
                    "id": int(row[4]),
                    "box": (x1, y1, x2, y2),
                    "cls": int(row[6]),
                })
        return output

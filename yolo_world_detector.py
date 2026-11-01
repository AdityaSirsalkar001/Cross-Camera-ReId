"""YOLO-World open-vocabulary detection plus independent per-camera ByteTrack.

Fix (2026-11-01): ultralytics 8.4+ BYTETracker.update() requires a Boxes
object whose internal tensors live on CPU.  The previous code reconstructed a
Boxes object from a NumPy array, but that conversion path silently failed in
the version now installed, returning zero tracked objects even when YOLO-World
found detections.  The fix is to call result.boxes.cpu() and pass that
directly to tracker.update(), which is the canonical API for this version.
"""

import hashlib
import shutil

import numpy as np
import torch
from ultralytics import YOLOWorld
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from config import (
    DETECTION_CONF,
    DETECTION_IMGSZ,
    YOLO_WORLD_CLASSES,
    YOLO_WORLD_WEIGHTS,
    MIN_BOX,
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
        """Run YOLO-World and return a CPU-resident Boxes object (or None if empty)."""
        result = self.model.predict(
            source=frame,
            imgsz=DETECTION_IMGSZ,
            conf=DETECTION_CONF,
            device=self.device,
            # NOTE: do NOT pass half=True here; it triggers a FutureWarning in
            # ultralytics 8.4+ and the 'quantize' equivalent is version-dependent.
            # FP32 inference on a GTX 1650 is fine for this model size.
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None  # Signal "no detections" without allocating empty tensors.
        # Move to CPU so BYTETracker.update() can call np.concatenate() on the
        # underlying tensors without a "can't convert cuda tensor to numpy" error.
        return boxes.cpu().numpy()


class YOLOWorldVideoTracker:
    """Keep tracker state separate for each camera while sharing detector weights."""

    def __init__(self, core):
        self.core = core
        args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        # Keep brief occlusions from fragmenting a person's or robot's local track.
        args.track_high_thresh = 0.01
        args.track_low_thresh = 0.001
        args.new_track_thresh = 0.01
        args.track_buffer = 60   # ~2 s at 30 fps; helps bridge occlusions
        self.tracker = BYTETracker(args)
        # Diagnostic snapshot for the most recent frame. Counters are read by
        # pipeline.py and do not affect detections or tracker decisions.
        self.last_diagnostics = {
            "raw_detections": 0,
            "bytetrack_outputs": 0,
            "small_box_rejections": 0,
        }

    def detect_and_track(self, frame):
        """Detect with YOLO-World, track with per-camera BYTETracker.

        Returns a list of dicts: {id, box:(x1,y1,x2,y2), cls}.

        The BYTETracker output shape is (N, 8):
            [x1, y1, x2, y2, track_id, score, cls, detection_idx]
        """
        cpu_boxes = self.core.detect(frame)
        raw_detection_count = len(cpu_boxes) if cpu_boxes is not None else 0

        if cpu_boxes is None or raw_detection_count == 0:
            # Advance the tracker with an empty update so Kalman states are
            # propagated and lost-track counters are incremented correctly.
            self.tracker.frame_id += 1
            self.last_diagnostics = {
                "raw_detections": 0,
                "bytetrack_outputs": 0,
                "small_box_rejections": 0,
            }
            return []

        # cpu_boxes is a Boxes object with tensors on CPU — safe for ByteTrack.
        tracked = self.tracker.update(cpu_boxes, img=frame)

        height, width = frame.shape[:2]
        output = []
        small_box_rejections = 0
        for row in tracked:
            x1, y1, x2, y2 = map(int, row[:4])
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX:
                output.append({
                    "id": int(row[4]),
                    "box": (x1, y1, x2, y2),
                    "cls": int(row[6]),
                })
            else:
                small_box_rejections += 1
        self.last_diagnostics = {
            "raw_detections": raw_detection_count,
            "bytetrack_outputs": len(tracked),
            "small_box_rejections": small_box_rejections,
        }
        return output

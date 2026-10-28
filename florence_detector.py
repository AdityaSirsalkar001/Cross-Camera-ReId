"""Florence-2 open-vocabulary detection with a separate ByteTrack per camera."""

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from config import (
    FLORENCE_CACHE_DIR,
    FLORENCE_DETECT_EVERY,
    FLORENCE_MODEL_ID,
    FLORENCE_TASK_PROMPT,
    FLORENCE_TEXT_PROMPT,
)


class FlorenceCore:
    """Loads the shared Florence-2 model and detects people and humanoid robots."""

    def __init__(self):
        try:
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Florence-2 requires transformers, timm, and einops. Install them in .venv; "
                "the exact command is in README.md."
            ) from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        FLORENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Loading {FLORENCE_MODEL_ID} on {self.device}; first run downloads its weights...")
        self.processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID,
            cache_dir=str(FLORENCE_CACHE_DIR),
        )
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_MODEL_ID,
            torch_dtype=dtype,
            cache_dir=str(FLORENCE_CACHE_DIR),
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, frame):
        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            text=FLORENCE_TASK_PROMPT + FLORENCE_TEXT_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
        )
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text,
            task=FLORENCE_TASK_PROMPT,
            image_size=(width, height),
        )
        result = parsed.get(FLORENCE_TASK_PROMPT, {})
        boxes, labels = result.get("bboxes", []), result.get("labels", [])
        detections = []
        for coords, label in zip(boxes, labels):
            name = str(label).lower()
            if "robot" in name or "humanoid" in name:
                cls = 1
            elif "person" in name or "human" in name:
                cls = 0
            else:
                continue
            x1, y1, x2, y2 = map(int, coords)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                # Florence's standard postprocessor returns boxes and labels, but no
                # calibrated confidence. Keep a fixed detector score for ByteTrack.
                detections.append([x1, y1, x2, y2, 0.80, cls])
        return np.asarray(detections, dtype=np.float32).reshape(-1, 6)


class FlorenceVideoTracker:
    """Per-camera ByteTrack state around shared Florence-2 detections."""

    def __init__(self, core):
        self.core = core
        tracker_args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        self.tracker = BYTETracker(tracker_args)
        self.frame_index = 0

    def detect_and_track(self, frame):
        self.frame_index += 1
        is_detection_frame = (self.frame_index - 1) % FLORENCE_DETECT_EVERY == 0
        if is_detection_frame:
            self._expire_old_tracks()
            detections = self.core.detect(frame)
            boxes = Boxes(detections, orig_shape=frame.shape[:2])
            tracked = self.tracker.update(boxes.cpu().numpy(), img=frame)
        else:
            # Advance ByteTrack's Kalman state without running Florence.
            self.tracker.frame_id += 1
            pool = self.tracker.tracked_stracks + self.tracker.lost_stracks
            self.tracker.multi_predict(pool)
            self._expire_old_tracks()
            return self._current_track_boxes(frame.shape[:2])

        output = []
        height, width = frame.shape[:2]
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

    def _expire_old_tracks(self):
        stale = [
            track for track in self.tracker.tracked_stracks + self.tracker.lost_stracks
            if self.tracker.frame_id - track.frame_id > self.tracker.max_frames_lost
        ]
        if stale:
            stale_ids = {id(track) for track in stale}
            self.tracker.tracked_stracks = [
                track for track in self.tracker.tracked_stracks if id(track) not in stale_ids
            ]
            self.tracker.lost_stracks = [
                track for track in self.tracker.lost_stracks if id(track) not in stale_ids
            ]
            self.tracker.removed_stracks.extend(stale)

    def _current_track_boxes(self, shape):
        height, width = shape
        output = []
        for track in self.tracker.tracked_stracks:
            if not track.is_activated:
                continue
            x1, y1, x2, y2 = map(int, track.xyxy)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                output.append({
                    "id": int(track.track_id),
                    "box": (x1, y1, x2, y2),
                    "cls": int(track.cls),
                })
        return output

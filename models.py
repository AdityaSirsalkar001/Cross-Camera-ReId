import hashlib
import shutil

import torch

from config import REID_WEIGHTS
from yolo_world_detector import YOLOWorldCore, YOLOWorldVideoTracker

print("Loading vision models...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Share one detector model, but keep independent tracker state for each camera.
_yolo_world = YOLOWorldCore()
yolo_world_cam1 = YOLOWorldVideoTracker(_yolo_world)
yolo_world_cam2 = YOLOWorldVideoTracker(_yolo_world)

_REID_REPO = "kaiyangzhou/osnet"
_REID_FILENAME = "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
_REID_REVISION = "01af85e82a9db4f3a4f6ed3a72ed9150bd416d04"
_REID_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"


def _ensure_reid_weights():
    """Download the pinned OSNet-AIN checkpoint into this project and verify it."""
    if REID_WEIGHTS.is_file():
        digest = hashlib.sha256(REID_WEIGHTS.read_bytes()).hexdigest()
        if digest != _REID_SHA256:
            raise RuntimeError(
                f"OSNet checkpoint at {REID_WEIGHTS} failed SHA-256 validation. "
                "Move the file aside and rerun to download a verified copy."
            )
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading OSNet weights requires huggingface_hub. "
            "Install project dependencies using the README setup instructions."
        ) from exc

    cache_dir = REID_WEIGHTS.parent / "models" / "huggingface-cache"
    print("Downloading the pinned OSNet-AIN ReID checkpoint from Hugging Face...")
    cached_file = hf_hub_download(
        repo_id=_REID_REPO,
        filename=_REID_FILENAME,
        revision=_REID_REVISION,
        cache_dir=str(cache_dir),
    )
    REID_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    temp_file = REID_WEIGHTS.with_suffix(".pth.download")
    shutil.copyfile(cached_file, temp_file)
    digest = hashlib.sha256(temp_file.read_bytes()).hexdigest()
    if digest != _REID_SHA256:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded OSNet checkpoint failed SHA-256 validation; "
            "it was not installed. Check the network response and try again."
        )
    temp_file.replace(REID_WEIGHTS)


_ensure_reid_weights()
try:
    from torchreid.utils import FeatureExtractor
except ImportError as exc:
    raise RuntimeError(
        "The official Torchreid FeatureExtractor is required for cross-camera matching. "
        "See the setup command in README.md."
    ) from exc

reid_extractor = FeatureExtractor(
    model_name="osnet_ain_x1_0",
    model_path=str(REID_WEIGHTS),
    image_size=(256, 128),
    device=DEVICE,
    verbose=False,
)

# Distinct colors for different matched pairs (BGR format)
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (0, 165, 255),
    (128, 0, 128),
]

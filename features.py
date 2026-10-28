import cv2
import numpy as np
from models import reid_extractor

def deep_embedding(crop_bgr):
    """L2-normalised OSNet person-ReID embedding of a BGR crop."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    e = reid_extractor(rgb).flatten().detach().cpu().numpy()
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


def body_shape(crop_bgr):
    """Return a normalized HOG shape descriptor and the person's box ratio."""
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 12:
        return None, None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 128), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    angle %= np.pi  # unsigned gradient orientation
    bins = np.minimum((angle * (9.0 / np.pi)).astype(np.int32), 8)
    descriptor = np.zeros((16, 8, 9), dtype=np.float32)
    for cy in range(16):
        for cx in range(8):
            ys, xs = slice(cy * 8, (cy + 1) * 8), slice(cx * 8, (cx + 1) * 8)
            descriptor[cy, cx] = np.bincount(
                bins[ys, xs].ravel(), weights=magnitude[ys, xs].ravel(), minlength=9
            )
    descriptor = descriptor.ravel()
    descriptor /= np.linalg.norm(descriptor) + 1e-8
    return descriptor, w / float(h)

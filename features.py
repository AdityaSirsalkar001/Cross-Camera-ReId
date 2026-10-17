import cv2
import numpy as np
import torch
from models import DEVICE, preprocess, resnet

def deep_embedding(crop_bgr):
    """L2-normalised ResNet50 embedding of a BGR crop."""
    tensor = preprocess(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        e = resnet(tensor).flatten().cpu().numpy()
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

import cv2
import numpy as np
from config import USE_HOMOGRAPHY, H_INTERVAL, H_MIN_INLIERS

class GroundMapper:
    def __init__(self):
        self.H = None
        self.inliers = 0
        try:
            self.det = cv2.SIFT_create(2000)
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        except AttributeError:
            self.det = cv2.ORB_create(3000)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    @property
    def ready(self):
        return USE_HOMOGRAPHY and self.H is not None

    def update(self, f1, f2, boxes1, boxes2, frame_idx):
        if not USE_HOMOGRAPHY or (frame_idx % H_INTERVAL != 0 and self.H is not None):
            return
        g1, g2 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        m1 = np.full(g1.shape, 255, np.uint8)
        m2 = np.full(g2.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes1:      # ignore moving objects -> use background only
            m1[y1:y2, x1:x2] = 0
        for x1, y1, x2, y2 in boxes2:
            m2[y1:y2, x1:x2] = 0
        kp1, d1 = self.det.detectAndCompute(g1, m1)
        kp2, d2 = self.det.detectAndCompute(g2, m2)
        if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
            return
        good = []
        for pair in self.bf.knnMatch(d2, d1, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < H_MIN_INLIERS:
            return
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return
        inl = int(mask.sum())
        if inl >= H_MIN_INLIERS:
            self.H, self.inliers = H, inl

    def project(self, pt):
        p = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), self.H)[0, 0]
        return float(p[0]), float(p[1])

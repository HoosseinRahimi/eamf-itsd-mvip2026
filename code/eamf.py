from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion


@dataclass
class ExposureStats:
    mean: float
    std: float
    score: float


def luminance_stats(img_bgr: np.ndarray, target_mean: float = 0.48, target_std: float = 0.18) -> ExposureStats:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32) / 255.0
    mean = float(l.mean())
    std = float(l.std())
    exposure_term = math.exp(-((mean - target_mean) ** 2) / (2 * 0.22 ** 2))
    contrast_term = min(1.0, std / max(target_std, 1e-6))
    score = float(0.75 * exposure_term + 0.25 * contrast_term)
    return ExposureStats(mean, std, score)


def adaptive_gamma(img_bgr: np.ndarray, target_mean: float = 0.48) -> np.ndarray:
    stats = luminance_stats(img_bgr)
    m = min(max(stats.mean, 1e-3), 0.999)
    gamma = math.log(target_mean) / math.log(m)
    gamma = float(np.clip(gamma, 0.35, 1.0))
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, table)


def clahe_luminance(img_bgr: np.ndarray, clip_limit: float = 2.0, grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def candidate_views(
    img_bgr: np.ndarray,
    mean_threshold: float = 0.38,
    contrast_threshold: float = 0.12,
) -> Tuple[List[np.ndarray], List[float], bool]:
    """Return views, fusion weights, and whether enhancement was activated."""
    s0 = luminance_stats(img_bgr)
    if s0.mean >= mean_threshold and s0.std >= contrast_threshold:
        return [img_bgr], [1.0], False

    gamma = adaptive_gamma(img_bgr)
    clahe = clahe_luminance(img_bgr)
    views = [img_bgr, gamma, clahe]
    scores = [max(luminance_stats(v).score, 0.05) for v in views]
    # Keep original view influential so enhancement artifacts cannot fully dominate.
    scores[0] = max(scores[0], 0.65)
    return views, scores, True


def fuse_detections(
    detections: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    image_shape: Tuple[int, int],
    view_weights: List[float],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
):
    """Fuse same-detector predictions from multiple photometric views.

    Each detection tuple is (xyxy[N,4], conf[N], cls[N]).
    Returns xyxy, conf, cls arrays.
    """
    h, w = image_shape[:2]
    boxes_list, scores_list, labels_list = [], [], []
    for boxes, scores, labels in detections:
        if len(boxes) == 0:
            boxes_list.append([]); scores_list.append([]); labels_list.append([])
            continue
        b = boxes.astype(np.float32).copy()
        b[:, [0, 2]] /= float(w)
        b[:, [1, 3]] /= float(h)
        b = np.clip(b, 0.0, 1.0)
        boxes_list.append(b.tolist())
        scores_list.append(scores.astype(float).tolist())
        labels_list.append(labels.astype(int).tolist())

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=view_weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    if len(boxes) == 0:
        return np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)
    boxes = np.asarray(boxes, np.float32)
    boxes[:, [0, 2]] *= float(w)
    boxes[:, [1, 3]] *= float(h)
    return boxes, np.asarray(scores, np.float32), np.asarray(labels, np.int32)

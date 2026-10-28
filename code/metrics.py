from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def xywhn_to_xyxy(boxes, w, h):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0,4), dtype=np.float32)
    out = np.empty_like(boxes)
    x, y, bw, bh = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    out[:,0] = (x - bw/2) * w; out[:,1] = (y - bh/2) * h
    out[:,2] = (x + bw/2) * w; out[:,3] = (y + bh/2) * h
    return out


def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), np.float32)
    x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a1 = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    a2 = np.maximum(0, boxes[:,2]-boxes[:,0]) * np.maximum(0, boxes[:,3]-boxes[:,1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def ap_from_pr(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def evaluate_map(records, num_classes: int, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)
    aps_by_iou = []
    per_class_ap50 = {}

    for thr in iou_thresholds:
        class_aps = []
        for cls in range(num_classes):
            preds = []
            gt_by_image = {}
            total_gt = 0
            for rec in records:
                gmask = rec['gt_cls'] == cls
                pmask = rec['pred_cls'] == cls
                gboxes = rec['gt_boxes'][gmask]
                gt_by_image[rec['image_id']] = {'boxes': gboxes, 'used': np.zeros(len(gboxes), dtype=bool)}
                total_gt += len(gboxes)
                for b, s in zip(rec['pred_boxes'][pmask], rec['pred_scores'][pmask]):
                    preds.append((float(s), rec['image_id'], b))
            if total_gt == 0:
                continue
            preds.sort(key=lambda x: x[0], reverse=True)
            tp = np.zeros(len(preds), np.float32); fp = np.zeros(len(preds), np.float32)
            for i, (_, image_id, box) in enumerate(preds):
                gt = gt_by_image[image_id]
                ious = box_iou_one_to_many(box, gt['boxes'])
                if len(ious):
                    j = int(np.argmax(ious))
                    if ious[j] >= thr and not gt['used'][j]:
                        tp[i] = 1; gt['used'][j] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            if len(preds) == 0:
                ap = 0.0
            else:
                tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
                recall = tp_c / max(total_gt, 1)
                precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
                ap = ap_from_pr(recall, precision)
            class_aps.append(ap)
            if abs(float(thr) - 0.50) < 1e-6:
                per_class_ap50[cls] = ap
        aps_by_iou.append(float(np.mean(class_aps)) if class_aps else 0.0)

    return {
        'mAP50': aps_by_iou[0],
        'mAP50_95': float(np.mean(aps_by_iou)),
        'aps_by_iou': [float(x) for x in aps_by_iou],
        'per_class_ap50': per_class_ap50,
    }

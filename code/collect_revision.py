"""Collect resumable full-fold per-view predictions; never replace old evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from eamf import adaptive_gamma, clahe_luminance, luminance_stats
from metrics import xywhn_to_xyxy
from provenance_utils import build_manifest, environment_record, sha256_file, write_json

CONDITIONS = ['clean', 'gamma1.8', 'gamma2.4', 'gamma3.0', 'linear_noise', 'overexposed', 'haze']

def working_image(image, longside):
    if not longside:
        return image
    h, w = image.shape[:2]
    ratio = longside / max(h, w)
    shape = (round(w * ratio), round(h * ratio))
    return cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)

def degradation(image, condition, image_id):
    if condition == 'clean':
        return image
    if condition.startswith('gamma'):
        g = float(condition[5:])
        lut = np.array([255 * (v / 255) ** g for v in range(256)], np.uint8)
        return cv2.LUT(image, lut)
    x = image.astype(np.float32) / 255
    linear = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    if condition == 'linear_noise':
        seed = int(hashlib.sha256(('20261027:' + image_id).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        linear = rng.poisson(linear * .10 * 1024).astype(np.float32) / 1024
        linear += rng.normal(0, 2 / 1024, linear.shape).astype(np.float32)
    elif condition == 'overexposed':
        linear *= 2.0
    elif condition == 'haze':
        # Constant transmission and achromatic airlight: synthetic stress only.
        return np.round(np.clip(.6 * x + .4 * .9, 0, 1) * 255).astype(np.uint8)
    else:
        raise ValueError(condition)
    linear = np.clip(linear, 0, 1)
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    return np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8)

def ground_truth(path, w, h):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [list(map(float, x.split())) for x in path.read_text().splitlines() if x.strip()]
    if any(len(x) != 5 for x in rows):
        raise ValueError(f'Invalid detection label {path}')
    classes = np.array([int(x[0]) for x in rows], np.int32)
    return classes.tolist(), xywhn_to_xyxy([x[1:] for x in rows], w, h).tolist()

def predict(model, image, device, size=640, flip=False):
    if flip:
        image = np.ascontiguousarray(image[:, ::-1])
    result = model.predict(image, imgsz=size, conf=.001, iou=.70,
                           device=device, verbose=False, half=False, max_det=300)[0]
    b = result.boxes
    boxes = b.xyxy.cpu().numpy().astype(np.float32)
    if flip and len(boxes):
        left, right = boxes[:, 0].copy(), boxes[:, 2].copy()
        boxes[:, 0], boxes[:, 2] = image.shape[1] - right, image.shape[1] - left
    return {'boxes': boxes.tolist(), 'scores': b.conf.cpu().tolist(), 'classes': b.cls.int().cpu().tolist()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--work-longside', type=int, default=0)
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261027)
    np.random.seed(20261027)
    if args.device != 'cpu':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    paths = sorted(x for x in args.images.iterdir() if x.suffix.lower() in {'.jpg', '.png', '.jpeg'})
    if args.limit:
        paths = paths[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'weights_sha256': sha256_file(args.weights), 'images': str(args.images),
        'labels': str(args.labels), 'n_images': len(paths), 'selection': 'entire official Fold 1; limit=0',
        'imgsz': 640, 'conf': .001, 'detector_iou': .70, 'max_det': 300,
        'device': args.device, 'half': False, 'threads': args.threads,
        'work_longside': args.work_longside,
        'common_resize': 'INTER_LINEAR, aspect-preserving; same working input and normalized labels for all methods; corruptions after common resize',
        'seed': 20261027, 'code_sha256': sha256_file(Path(__file__)),
        'eamf_sha256': sha256_file(Path(__file__).with_name('eamf.py')),
        'parameters_frozen': 'Original published constants; no test-label tuning',
        'noise': {'linear_exposure': .10, 'photon_scale': 1024, 'read_std': 2 / 1024,
                  'quantization': '8-bit sRGB after shot/read noise; per-image SHA-derived seed'},
        'tta': 'original 640, horizontal flip 640 mapped back, unflipped scale 512; 3 calls, not equal FLOPs',
        'timing': 'Collection times are diagnostic only; use standalone benchmark for paper latency',
    }
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    old = args.output / 'protocol.json'
    if old.exists() and json.loads(old.read_text())['signature'] != signature:
        raise RuntimeError('Output protocol differs; select a new output folder')
    write_json(old, {**protocol, 'signature': signature, 'environment': environment_record()})
    write_json(args.output / 'manifest.json', build_manifest(paths, args.labels, args.images))
    model = YOLO(str(args.weights))
    # Warm up before diagnostic timings.
    for _ in range(3):
        predict(model, np.zeros((1080, 1920, 3), np.uint8), args.device)
    started = time.perf_counter()
    for condition in args.conditions:
        folder = args.output / condition
        folder.mkdir(exist_ok=True)
        for i, path in enumerate(paths):
            dest = folder / (path.stem + '.json')
            if dest.exists():
                cached = json.loads(dest.read_text())
                if cached.get('signature') != signature or cached.get('image') != path.name:
                    raise RuntimeError(f'Cache mismatch {dest}')
                continue
            raw = cv2.imread(str(path))
            if raw is None:
                raise RuntimeError(f'Unreadable {path}')
            image = degradation(working_image(raw, args.work_longside), condition, path.name)
            h, w = image.shape[:2]
            labels, boxes = ground_truth(args.labels / (path.stem + '.txt'), w, h)
            t = time.perf_counter()
            original_stats = luminance_stats(image)
            router_ms = 1000 * (time.perf_counter() - t)
            t = time.perf_counter()
            gamma = adaptive_gamma(image)
            gamma_ms = 1000 * (time.perf_counter() - t)
            t = time.perf_counter()
            clahe = clahe_luminance(image)
            clahe_ms = 1000 * (time.perf_counter() - t)
            stats = [original_stats, luminance_stats(gamma), luminance_stats(clahe)]
            views = {}
            inference_ms = {}
            for name, view, size, flip in [('original', image, 640, False),
                                           ('gamma', gamma, 640, False),
                                           ('clahe', clahe, 640, False),
                                           ('flip', image, 640, True),
                                           ('scale', image, 512, False)]:
                t = time.perf_counter()
                views[name] = predict(model, view, args.device, size, flip)
                if args.device != 'cpu':
                    torch.cuda.synchronize()
                inference_ms[name] = 1000 * (time.perf_counter() - t)
            mu = min(max(original_stats.mean, .001), .999)
            effective_gamma = float(np.clip(math.log(.48) / math.log(mu), .35, 1))
            write_json(dest, {
                'signature': signature, 'image': path.name, 'condition': condition,
                'shape': [h, w], 'original_shape': list(raw.shape[:2]),
                'gt_cls': labels, 'gt_boxes': boxes, 'views': views,
                'stats': [s.__dict__ for s in stats], 'gamma_exponent': effective_gamma,
                'gamma_duplicates_original': bool(np.array_equal(image, gamma)),
                'router_active': original_stats.mean < .38 or original_stats.std < .12,
                'diagnostic_ms': {'router': router_ms, 'gamma': gamma_ms, 'clahe': clahe_ms,
                                  'inference': inference_ms},
            })
            if (i + 1) % 50 == 0 or i + 1 == len(paths):
                elapsed = time.perf_counter() - started
                print(f'{condition}: {i+1}/{len(paths)}, total_elapsed={elapsed:.1f}s', flush=True)
        print(f'COMPLETE {condition}', flush=True)
    print('ALL REQUESTED CONDITIONS COMPLETE', flush=True)

if __name__ == '__main__':
    main()

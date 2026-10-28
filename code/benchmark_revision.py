"""One warmed, randomized standalone timing protocol for all paper tables."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from eamf import adaptive_gamma, clahe_luminance, luminance_stats, fuse_detections
from collect_revision import degradation, predict, working_image, CONDITIONS
from analyze_revision import arrays, fuse
from provenance_utils import environment_record, sha256_file, write_json

METHODS = ['Baseline', 'Gamma', 'CLAHE', 'RoutedGamma', 'Unconditional', 'EAMF', 'TTA3', 'TTAms3']

def sync(device):
    if device != 'cpu':
        torch.cuda.synchronize()

def timed_method(model, image, method, device):
    sync(device)
    start = time.perf_counter()
    stage = {'router': 0., 'enhancement_and_weights': 0., 'inference': 0., 'fusion': 0.}
    active = True
    if method in {'EAMF', 'RoutedGamma'}:
        t = time.perf_counter()
        s = luminance_stats(image)
        active = s.mean < .38 or s.std < .12
        stage['router'] = time.perf_counter() - t
    t = time.perf_counter()
    if method in {'Baseline', 'TTA3', 'TTAms3'} or (not active and method in {'EAMF', 'RoutedGamma'}):
        views, weights = [image], [1.]
    elif method in {'Gamma', 'RoutedGamma'}:
        views, weights = [adaptive_gamma(image)], [1.]
    elif method == 'CLAHE':
        views, weights = [clahe_luminance(image)], [1.]
    else:
        views = [image, adaptive_gamma(image), clahe_luminance(image)]
        weights = [max(luminance_stats(v).score, .05) for v in views]
        weights[0] = max(weights[0], .65)
    stage['enhancement_and_weights'] = time.perf_counter() - t
    t = time.perf_counter()
    if method == 'TTA3':
        preds = [predict(model, image, device), predict(model, image, device, flip=True),
                 predict(model, image, device, size=512)]
    elif method == 'TTAms3':
        preds = [predict(model, image, device), predict(model, image, device, size=512),
                 predict(model, image, device, size=768)]
    else:
        preds = [predict(model, v, device) for v in views]
    sync(device)
    stage['inference'] = time.perf_counter() - t
    t = time.perf_counter()
    if method in {'TTA3', 'TTAms3'}:
        keys = ['original', 'flip', 'scale'] if method == 'TTA3' else ['original', 'scale512', 'scale768']
        row = {'shape': image.shape[:2], 'views': dict(zip(keys, preds))}
        fuse(row, keys, [1, 1, 1])
    elif len(preds) > 1:
        fuse_detections([arrays(x) for x in preds], image.shape, weights)
    stage['fusion'] = time.perf_counter() - t
    sync(device)
    total = time.perf_counter() - start
    return {**{k: 1000 * v for k, v in stage.items()}, 'total': 1000 * total}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--samples', type=int, default=48)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--work-longside', type=int, default=640)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(20261027)
    paths = sorted(x for x in args.images.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    selected = sorted(rng.choice(len(paths), min(args.samples, len(paths)), replace=False).tolist())
    model = YOLO(str(args.weights))
    for _ in range(5):
        predict(model, np.zeros((1080, 1920, 3), np.uint8), args.device)
    result = {'protocol': {'seed': 20261027, 'selected_images': [paths[i].name for i in selected],
                          'repeats': args.repeats, 'warmup': 5, 'method_order': 'randomized per image/repetition',
                          'disk_loading_and_stress_generation': 'excluded, identical preloaded input for every method',
                          'included': 'router, enhancement, quality scores, detector preprocessing/inference/NMS/CPU transfer, fusion',
                          'synchronization': 'CUDA synchronized before/after measured method and inference stage',
                          'weights_sha256': sha256_file(args.weights), 'threads': 4, 'opencv_threads': 1,
                          'device': args.device, 'code_sha256': sha256_file(Path(__file__))},
              'environment': environment_record(), 'rows': []}
    result['protocol']['work_longside'] = args.work_longside
    result['protocol']['timing_input'] = 'Common resized working input; native-image decode, common resize and stress generation excluded'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        if saved['protocol'] != result['protocol']:
            raise RuntimeError('Benchmark protocol changed; use a new output path')
        result = saved
        print(f'Resume {len(result["rows"])} completed standalone timings', flush=True)
    completed = {(r['condition'], r['image'], r['repeat'], r['method']) for r in result['rows']}
    for condition in args.conditions:
        for i in selected:
            image = degradation(working_image(cv2.imread(str(paths[i])), args.work_longside), condition, paths[i].name)
            for repeat in range(args.repeats):
                for method in rng.permutation(METHODS).tolist():
                    key = (condition, paths[i].name, repeat, method)
                    if key in completed:
                        continue
                    row = timed_method(model, image, method, args.device)
                    result['rows'].append({'condition': condition, 'image': paths[i].name,
                                           'repeat': repeat, 'method': method, 'ms': row})
                    completed.add(key)
            write_json(args.output, result)
        print(f'BENCHMARK COMPLETE {condition}', flush=True)
    summary = {}
    for condition in args.conditions:
        summary[condition] = {}
        for method in METHODS:
            rows = [r['ms'] for r in result['rows'] if r['condition'] == condition and r['method'] == method]
            summary[condition][method] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    result['summary_ms'] = summary
    write_json(args.output, result)
    print(json.dumps(summary, indent=2), flush=True)

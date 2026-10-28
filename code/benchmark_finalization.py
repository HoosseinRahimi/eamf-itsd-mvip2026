"""One standalone randomized benchmark covering original and added comparators."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from benchmark_revision import timed_method, sync
from collect_revision import degradation, working_image, predict
from provenance_utils import environment_record, sha256_file, write_json
from zero_dce_adapter import ZeroDCE, EXPECTED_WEIGHT

METHODS = ['Baseline', 'Gamma', 'EAMF', 'TTAms3', 'ZeroDCE', 'FineTuneGamma', 'YOLOv9Baseline', 'YOLOv9EAMF']
CONDITIONS = ['clean', 'gamma3.0', 'linear_noise']


def measure(models, enhancer, image, method):
    if method == 'YOLOv9Baseline':
        return timed_method(models['yolov9'], image, 'Baseline', '0')
    if method == 'YOLOv9EAMF':
        return timed_method(models['yolov9'], image, 'EAMF', '0')
    if method == 'FineTuneGamma':
        return timed_method(models['finetune'], image, 'Baseline', '0')
    if method != 'ZeroDCE':
        return timed_method(models['original'], image, method, '0')
    sync('0')
    start = time.perf_counter()
    enhanced = enhancer.enhance(image)
    sync('0')
    enhanced_end = time.perf_counter()
    predict(models['original'], enhanced, '0')
    sync('0')
    end = time.perf_counter()
    return {'router': 0., 'enhancement_and_weights': 1000 * (enhanced_end - start),
            'inference': 1000 * (end - enhanced_end), 'fusion': 0., 'total': 1000 * (end - start)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--finetune-weights', type=Path, required=True)
    p.add_argument('--yolov9-weights', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(20261027)
    paths = sorted(args.images.glob('*.jpg'))
    assert len(paths) == 2352
    selected = sorted(rng.choice(len(paths), 48, replace=False).tolist())
    protocol = {'seed': 20261027, 'selected_images': [paths[i].name for i in selected], 'repeats': 3,
        'warmup': '5 invocations per method on common working inputs before measurement',
        'methods': METHODS, 'conditions': CONDITIONS, 'work_longside': 640, 'device': '0',
        'half': False, 'tf32': False, 'threads': 4, 'opencv_threads': 1,
        'method_order': 'randomized per image/repetition',
        'timing_input': 'preloaded common 640 working image; native decoding, shared resize and common corruption excluded',
        'included': 'router, enhancement and its GPU transfer, quality weights, detector preprocessing/inference/NMS/CPU box transfer, fusion',
        'code_sha256': sha256_file(Path(__file__)),
        'timing_helper_sha256': sha256_file(Path(__file__).with_name('benchmark_revision.py')),
        'adapter_sha256': sha256_file(Path(__file__).with_name('zero_dce_adapter.py')),
        'weights_sha256': sha256_file(args.weights), 'finetune_weights_sha256': sha256_file(args.finetune_weights),
        'yolov9_weights_sha256': sha256_file(args.yolov9_weights), 'enhancer_weights_sha256': EXPECTED_WEIGHT}
    models = {'original': YOLO(str(args.weights)), 'finetune': YOLO(str(args.finetune_weights)),
              'yolov9': YOLO(str(args.yolov9_weights))}
    enhancer = ZeroDCE(args.upstream, '0')
    for i in selected[:5]:
        image = working_image(cv2.imread(str(paths[i])), 640)
        for method in METHODS:
            measure(models, enhancer, image, method)
    result = {'protocol': protocol, 'environment': environment_record(), 'rows': []}
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        assert saved['protocol'] == protocol, 'Benchmark protocol differs'
        result = saved
    completed = {(r['condition'], r['image'], r['repeat'], r['method']) for r in result['rows']}
    for condition in CONDITIONS:
        for i in selected:
            image = degradation(working_image(cv2.imread(str(paths[i])), 640), condition, paths[i].name)
            for repeat in range(3):
                for method in rng.permutation(METHODS).tolist():
                    key = condition, paths[i].name, repeat, method
                    if key in completed:
                        continue
                    result['rows'].append({'condition': condition, 'image': paths[i].name,
                        'repeat': repeat, 'method': method, 'ms': measure(models, enhancer, image, method)})
                    completed.add(key)
            write_json(args.output, result)
        print('BENCHMARK COMPLETE ' + condition, flush=True)
    assert len(result['rows']) == len(completed) == 3456
    result['summary_ms'] = {condition: {method: {key: float(np.mean([
        row['ms'][key] for row in result['rows'] if row['condition'] == condition and row['method'] == method]))
        for key in ['router', 'enhancement_and_weights', 'inference', 'fusion', 'total']} for method in METHODS} for condition in CONDITIONS}
    write_json(args.output, result)
    print('BENCHMARK ALL METHODS COMPLETE: 3456 measured rows', flush=True)

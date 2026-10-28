"""Fresh-environment CPU smoke reproduction of sampled recorded GPU inference paths."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from analyze_revision import arrays, method_prediction
from collect_revision import degradation, working_image, predict
from eamf import adaptive_gamma, clahe_luminance, luminance_stats
from provenance_utils import environment_record, sha256_file, write_json
from zero_dce_adapter import ZeroDCE


def compare(actual, reference):
    a, a_score, a_cls = actual
    b, b_score, b_cls = reference
    am, bm = a_score >= .25, b_score >= .25
    a, a_score, a_cls = a[am], a_score[am], a_cls[am]
    b, b_score, b_cls = b[bm], b_score[bm], b_cls[bm]
    assert len(a) == len(b), 'Operating-point detection count differs'
    if not len(a):
        return {'detections': 0, 'min_iou': None, 'max_score_error': 0., 'max_coordinate_error': 0.}
    overlaps = box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    i, j = linear_sum_assignment(-(overlaps - 100 * (a_cls[:, None] != b_cls[None, :])))
    assert np.array_equal(a_cls[i], b_cls[j]), 'Class mismatch'
    iou, score_error, coordinate_error = overlaps[i, j].min(), np.abs(a_score[i] - b_score[j]).max(), np.abs(a[i] - b[j]).max()
    assert iou >= .99 and score_error <= .001 and coordinate_error <= .05, 'CPU/GPU tolerance exceeded'
    return {'detections': len(a), 'min_iou': float(iou), 'max_score_error': float(score_error), 'max_coordinate_error': float(coordinate_error)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--extension', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    paths = sorted((args.primary / 'full_gpu640' / 'clean').glob('*.json'))
    rng = np.random.default_rng(20261028)
    selected = sorted(rng.choice(len(paths), 12, replace=False).tolist())
    models = {'original': YOLO(str(args.weights)),
        'finetune': YOLO(str(args.extension / 'gamma_finetune' / 'run' / 'weights' / 'last.pt')),
        'yolov9': YOLO(str(args.extension / 'official_extra' / 'ITSD-dataset' / 'yolov9_weights' / 'best_Fold1.pt'))}
    enhancer = ZeroDCE(args.extension / 'external' / 'zero_dce', 'cpu')
    report = {'status': 'RUNNING', 'environment': environment_record(), 'code_sha256': sha256_file(Path(__file__)),
        'selection': '12 of 2352 images without replacement, fixed seed 20261028; same IDs across three conditions',
        'tolerances_predeclared': {'display_conf': .25, 'minimum_same_class_IoU': .99,
            'maximum_score_error': .001, 'maximum_coordinate_error_pixels': .05},
        'scope': 'Sampled CPU/GPU inference consistency, not a fresh full-dataset inference run or a performance estimate.', 'rows': []}
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        for i in selected:
            filename = paths[i].name
            original = json.loads((args.primary / 'full_gpu640' / condition / filename).read_text())
            image = degradation(working_image(cv2.imread(str(args.images / original['image'])), 640), condition, original['image'])
            z = json.loads((args.extension / 'zero_dce_predictions' / condition / filename).read_text())
            ft = json.loads((args.extension / 'finetune_predictions' / condition / filename).read_text())
            y9 = json.loads((args.extension / 'yolov9_predictions' / condition / filename).read_text())
            assert hashlib.sha256(image.tobytes()).hexdigest() == z['common_input_pixel_sha256'] == ft['common_input_pixel_sha256'] == y9['common_input_pixel_sha256']
            views = {'original': image, 'gamma': adaptive_gamma(image), 'clahe': clahe_luminance(image)}
            for index, view in enumerate(views.values()):
                stats = luminance_stats(view)
                assert np.allclose([stats.mean, stats.std, stats.score], [original['stats'][index][k] for k in ['mean','std','score']], atol=1e-12, rtol=0)
            current, current_y9 = dict(original), dict(y9)
            current['views'] = {key: predict(models['original'], value, 'cpu') for key, value in views.items()}
            current_y9['views'] = {key: predict(models['yolov9'], value, 'cpu') for key, value in views.items()}
            comparisons = {
                'Baseline': (arrays(current['views']['original']), arrays(original['views']['original'])),
                'EAMF': (method_prediction(current, 'EAMF'), method_prediction(original, 'EAMF')),
                'ZeroDCE': (arrays(predict(models['original'], enhancer.enhance(image), 'cpu')), arrays(z['prediction'])),
                'FineTuneGamma': (arrays(predict(models['finetune'], image, 'cpu')), arrays(ft['views']['original'])),
                'YOLOv9Baseline': (arrays(current_y9['views']['original']), arrays(y9['views']['original'])),
                'YOLOv9EAMF': (method_prediction(current_y9, 'EAMF'), method_prediction(y9, 'EAMF'))}
            for method, (actual, reference) in comparisons.items():
                report['rows'].append({'image': original['image'], 'condition': condition, 'method': method,
                    'consistency': compare(actual, reference), 'prediction': {'boxes': actual[0].tolist(),
                    'scores': actual[1].tolist(), 'classes': actual[2].tolist()}})
        print('CPU REPRODUCTION COMPLETE ' + condition, flush=True)
    assert len(report['rows']) == 216
    report['status'] = 'PASS'
    write_json(args.output, report)
    print('PASS: 216 sampled method/condition/image comparisons in clean CPU environment', flush=True)

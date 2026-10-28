"""Prospective bounded validation calibration; preserve the historical primary method."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from eamf import adaptive_gamma, clahe_luminance, luminance_stats
from collect_revision import degradation, working_image, predict, ground_truth
from analyze_revision import aggregate, arrays, fuse, match
from provenance_utils import build_manifest, environment_record, sha256_file, write_json

DEFAULT = {'mu_threshold': .38, 'std_threshold': .12, 'target': .48, 'gamma_min': .35,
           'clahe_clip': 2., 'quality_mean_width': .22, 'quality_std_reference': .18,
           'quality_alpha': .75, 'original_floor': .65, 'wbf_iou': .55}
AXES = {'mu_threshold': [.32, .38, .44], 'std_threshold': [.08, .12, .16],
        'target': [.42, .48, .54], 'gamma_min': [.25, .35, .45], 'clahe_clip': [1., 2., 3.],
        'quality_mean_width': [.16, .22, .28], 'quality_std_reference': [.12, .18, .24],
        'quality_alpha': [.5, .75, 1.], 'original_floor': [.5, .65, .8], 'wbf_iou': [.45, .55, .65]}
CONDITIONS = ['clean', 'gamma3.0', 'linear_noise']


def gamma_view(image, target, minimum):
    mean = np.clip(luminance_stats(image).mean, .001, .999)
    exponent = float(np.clip(math.log(target) / math.log(mean), minimum, 1.))
    return cv2.LUT(image, np.array([((i / 255.) ** exponent) * 255 for i in range(256)], np.uint8))


def keys(params):
    return ['original', f'gamma_{params["target"]:g}_{params["gamma_min"]:g}', f'clahe_{params["clahe_clip"]:g}']


def calibrated_prediction(row, params):
    stat = row['stats']['original']
    if stat['mean'] >= params['mu_threshold'] and stat['std'] >= params['std_threshold']:
        return arrays(row['views']['original'])
    selected = keys(params)
    weights = []
    for name in selected:
        s = row['stats'][name]
        score = params['quality_alpha'] * math.exp(-((s['mean'] - params['target']) ** 2) / (2 * params['quality_mean_width'] ** 2))
        score += (1 - params['quality_alpha']) * min(1., s['std'] / params['quality_std_reference'])
        weights.append(max(score, .05))
    weights[0] = max(weights[0], params['original_floor'])
    return fuse(row, selected, weights, iou=params['wbf_iou'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['validation', 'test'], required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--val-list', type=Path)
    p.add_argument('--labels', type=Path)
    p.add_argument('--selection', type=Path)
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(20261028)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    parent = json.loads((args.primary / 'full_gpu640' / 'protocol.json').read_text())
    assert sha256_file(args.weights) == parent['weights_sha256']
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'stage': args.stage, 'code_sha256': sha256_file(Path(__file__)),
                'metric_helper_sha256': sha256_file(Path(__file__).with_name('analyze_revision.py')),
                'parent_signature': parent['signature'], 'weights_sha256': parent['weights_sha256'],
                'conditions': CONDITIONS, 'default_parameters': DEFAULT, 'axes': AXES,
                'selection_criterion': 'maximum unrounded equal-condition mean mAP50:95 on validation only; earliest configuration within 1e-12 of maximum wins; default is first',
                'scope': 'Prospective exploratory revision; Fold1 previously inspected, not an untouched confirmation set; original detector training exposure not reconstructed.'}
    model = YOLO(str(args.weights))
    if args.stage == 'validation':
        assert args.val_list and args.labels
        paths = [Path(line) for line in args.val_list.read_text().splitlines() if line.strip()]
        assert len(paths) == 234
        manifest = build_manifest(paths, args.labels, args.images)
        test_manifest = json.loads((args.primary / 'full_gpu640' / 'manifest.json').read_text())
        assert not ({row['image_sha256'] for row in manifest} & {row['image_sha256'] for row in test_manifest})
        assert not ({row['image'] for row in manifest} & {row['image'] for row in test_manifest})
        write_json(args.output / 'validation_manifest.json', manifest)
        protocol.update(n_images=234, validation_manifest_sha256=sha256_file(args.output / 'validation_manifest.json'),
                        test_image_hash_overlap=0, configurations='one-factor-at-a-time: default plus two alternatives for each of ten axes; 21 total')
        configurations = [('default', dict(DEFAULT))]
        for axis, values in AXES.items():
            for value in values:
                if value != DEFAULT[axis]:
                    configurations.append((f'{axis}_{value:g}', {**DEFAULT, axis: value}))
        assert len(configurations) == 21
        gamma_pairs = sorted({(c['target'], c['gamma_min']) for _, c in configurations})
        clahe_clips = sorted({c['clahe_clip'] for _, c in configurations})
    else:
        assert args.selection
        selection = json.loads(args.selection.read_text())
        params = selection['selected_parameters']
        protocol.update(n_images=2352, selection_sha256=sha256_file(args.selection), selected_parameters=params)
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())['signature'] == signature, 'Protocol changed; select new output'
    else:
        write_json(protocol_path, {**protocol, 'signature': signature, 'environment': environment_record()})
    for _ in range(5):
        predict(model, np.zeros((360, 640, 3), np.uint8), '0')
    for condition in CONDITIONS:
        sources = paths if args.stage == 'validation' else sorted((args.primary / 'full_gpu640' / condition).glob('*.json'))
        for i, source_path in enumerate(sources):
            destination = args.output / condition / (source_path.stem + '.json')
            if destination.exists():
                assert json.loads(destination.read_text())['signature'] == signature
                continue
            if args.stage == 'validation':
                original = degradation(working_image(cv2.imread(str(source_path)), 640), condition, source_path.name)
                assert np.array_equal(gamma_view(original, .48, .35), adaptive_gamma(original)), 'Default gamma incompatibility'
                views = {'original': original}
                views.update({f'gamma_{t:g}_{m:g}': gamma_view(original, t, m) for t, m in gamma_pairs})
                views.update({f'clahe_{clip:g}': clahe_luminance(original, clip_limit=clip) for clip in clahe_clips})
                gt_cls, gt_boxes = ground_truth(args.labels / (source_path.stem + '.txt'), original.shape[1], original.shape[0])
                row = {'image': source_path.name, 'shape': list(original.shape[:2]), 'gt_cls': gt_cls, 'gt_boxes': gt_boxes,
                       'views': {name: predict(model, view, '0') for name, view in views.items()},
                       'stats': {name: vars(luminance_stats(view)) for name, view in views.items()}}
            else:
                source = json.loads(source_path.read_text())
                selected = keys(params)
                row = {key: source[key] for key in ['image', 'shape', 'gt_cls', 'gt_boxes']}
                row.update(parent_sha256=sha256_file(source_path), views={'original': source['views']['original']},
                           stats={'original': source['stats'][0]})
                image = None
                for j, name in enumerate(selected[1:], 1):
                    reusable = (j == 1 and params['target'] == .48 and params['gamma_min'] == .35) or (j == 2 and params['clahe_clip'] == 2.)
                    if reusable:
                        row['views'][name] = source['views']['gamma' if j == 1 else 'clahe']
                        row['stats'][name] = source['stats'][j]
                    else:
                        if image is None:
                            image = degradation(working_image(cv2.imread(str(args.images / source['image'])), 640), condition, source['image'])
                        view = gamma_view(image, params['target'], params['gamma_min']) if j == 1 else clahe_luminance(image, clip_limit=params['clahe_clip'])
                        row['views'][name] = predict(model, view, '0')
                        row['stats'][name] = vars(luminance_stats(view))
                boxes, scores, classes = calibrated_prediction(row, params)
                row['prediction'] = {'boxes': boxes.tolist(), 'scores': scores.tolist(), 'classes': classes.tolist()}
            row.update(signature=signature, condition=condition)
            write_json(destination, row)
            if (i + 1) % 100 == 0:
                print(f'{args.stage} {condition}: {i+1}/{len(sources)}', flush=True)
        print('COMPLETE ' + args.stage + ' ' + condition, flush=True)
    if args.stage == 'validation':
        rows = {condition: [json.loads(path.read_text()) for path in sorted((args.output / condition).glob('*.json'))] for condition in CONDITIONS}
        assert all(len(items) == 234 for items in rows.values())
        scores = []
        for name, parameters in configurations:
            metrics = {condition: aggregate([match(row, calibrated_prediction(row, parameters)) for row in items]) for condition, items in rows.items()}
            score = float(np.mean([metric['mAP50_95'] for metric in metrics.values()]))
            scores.append({'configuration': name, 'parameters': parameters, 'metrics': metrics, 'mean_mAP50_95': score})
            print(f'VALIDATION {name}: {score:.9f}', flush=True)
        maximum = max(score['mean_mAP50_95'] for score in scores)
        selected = next(score for score in scores if maximum - score['mean_mAP50_95'] <= 1e-12)
        write_json(args.output / 'selection.json', {'selected_configuration': selected['configuration'],
            'selected_parameters': selected['parameters'], 'scores': scores, 'validation_images': 234,
            'selection_criterion': protocol['selection_criterion'], 'code_sha256': sha256_file(Path(__file__)),
            'validation_protocol_sha256': sha256_file(protocol_path), 'test_labels_used_for_selection': False,
            'limitation': protocol['scope']})
        print('SELECTED ON VALIDATION ONLY: ' + selected['configuration'], flush=True)
    print('CALIBRATION STAGE COMPLETE', flush=True)

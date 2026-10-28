"""Derive matched comparisons from cached predictions with official Ultralytics AP."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou, compute_ap
from provenance_utils import sha256_file, write_json

METHODS = ['Baseline', 'Gamma', 'CLAHE', 'RoutedGamma', 'Unconditional', 'EAMF',
           'OriginalGamma', 'OriginalCLAHE', 'GammaCLAHE', 'UniformWBF', 'DeduplicatedWBF',
           'NoOriginalFloor', 'NMS', 'SoftNMS', 'TTA3']

def arrays(pred):
    return (np.asarray(pred['boxes'], np.float32).reshape(-1, 4),
            np.asarray(pred['scores'], np.float32), np.asarray(pred['classes'], np.int32))

def fuse(row, selected, weights, kind='wbf', iou=.55):
    h, w = row['shape']
    scale = np.array([w, h, w, h], np.float32)
    predictions = [arrays(row['views'][key]) for key in selected]
    if not any(len(pred[0]) for pred in predictions):
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32)
    # Preserve (0,4) for an empty view: NMS concatenates arrays across views.
    boxes = [np.clip(v[0] / scale, 0, 1).reshape(-1, 4) for v in predictions]
    scores = [v[1].tolist() for v in predictions]
    labels = [v[2].tolist() for v in predictions]
    if kind == 'wbf':
        b, s, c = weighted_boxes_fusion(boxes, scores, labels, weights=weights,
                    iou_thr=iou, skip_box_thr=.001, conf_type='avg', allows_overflow=False)
    elif kind == 'nms':
        b, s, c = nms(boxes, scores, labels, weights=weights, iou_thr=iou)
    else:
        b, s, c = soft_nms(boxes, scores, labels, weights=weights, method=2, iou_thr=iou, sigma=.5, thresh=.001)
    return np.asarray(b, np.float32).reshape(-1, 4) * scale, np.asarray(s, np.float32), np.asarray(c, np.int32)

def method_prediction(row, method, mu=.38, sigma=.12, wbf_iou=.55):
    active = row['stats'][0]['mean'] < mu or row['stats'][0]['std'] < sigma
    if method == 'Baseline':
        return arrays(row['views']['original'])
    if method in {'Gamma', 'CLAHE'}:
        return arrays(row['views'][method.lower()])
    if method == 'RoutedGamma':
        return arrays(row['views']['gamma' if active else 'original'])
    if method == 'TTA3':
        return fuse(row, ['original', 'flip', 'scale'], [1, 1, 1])
    if not active and method != 'Unconditional':
        return arrays(row['views']['original'])
    selected = ['original', 'gamma', 'clahe']
    q = [max(s['score'], .05) for s in row['stats']]
    q[0] = max(q[0], .65)
    if method in {'OriginalGamma', 'OriginalCLAHE', 'GammaCLAHE'}:
        indices = {'OriginalGamma': [0, 1], 'OriginalCLAHE': [0, 2], 'GammaCLAHE': [1, 2]}[method]
        selected = [selected[i] for i in indices]
        q = [q[i] for i in indices]
    if method == 'UniformWBF':
        q = [1.0] * 3
    if method == 'NoOriginalFloor':
        q[0] = max(row['stats'][0]['score'], .05)
    if method == 'DeduplicatedWBF' and row['gamma_duplicates_original']:
        selected, q = ['original', 'clahe'], [q[0], q[2]]
    return fuse(row, selected, q, 'nms' if method == 'NMS' else 'softnms' if method == 'SoftNMS' else 'wbf', wbf_iou)

def match(row, pred):
    b, s, c = pred
    gt_c = np.asarray(row['gt_cls'], np.int32)
    gt_b = np.asarray(row['gt_boxes'], np.float32).reshape(-1, 4)
    validator = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    tp = DetectionValidator.match_predictions(validator, torch.from_numpy(c), torch.from_numpy(gt_c),
                    box_iou(torch.from_numpy(gt_b), torch.from_numpy(b))).numpy()
    return tp, s, c, gt_c

def aggregate(items, indices=None):
    chosen = items if indices is None else [items[i] for i in indices]
    t = np.concatenate([x[0] for x in chosen], axis=0)
    s = np.concatenate([x[1] for x in chosen])
    c = np.concatenate([x[2] for x in chosen])
    gt = np.concatenate([x[3] for x in chosen])
    if not len(gt):
        return {'mAP50': None, 'mAP50_95': None, 'class_ap50': {}, 'classes': 0}
    if not len(s):
        classes = np.unique(gt)
        return {'mAP50': 0.0, 'mAP50_95': 0.0, 'class_ap50': {str(c): 0. for c in classes}, 'classes': len(classes)}
    result = ap_per_class(t, s, c, gt)
    ap = result[5]
    return {'mAP50': float(ap[:, 0].mean()), 'mAP50_95': float(ap.mean()),
            'class_ap50': {str(int(c)): float(v) for c, v in zip(result[6], ap[:, 0])},
            'classes': len(result[6])}

def false_positives(items, threshold=.25):
    tp = sum(int(x[0][x[1] >= threshold, 0].sum()) for x in items)
    det = sum(int((x[1] >= threshold).sum()) for x in items)
    return {'threshold': threshold, 'TP50': tp, 'FP50': det - tp,
            'FN50': sum(len(x[3]) for x in items) - tp}

def bootstrap_ap(items, indices):
    # The official AP computation, without generating unused 1000-point F1 curves.
    chosen = [items[i] for i in indices]
    tp = np.concatenate([x[0] for x in chosen], axis=0)
    scores = np.concatenate([x[1] for x in chosen])
    pred_classes = np.concatenate([x[2] for x in chosen])
    gt = np.concatenate([x[3] for x in chosen])
    order = np.argsort(-scores)
    tp, pred_classes = tp[order], pred_classes[order]
    classes, counts = np.unique(gt, return_counts=True)
    ap = np.zeros((len(classes), 10))
    for i, (cls, count) in enumerate(zip(classes, counts)):
        mask = pred_classes == cls
        if not mask.any():
            continue
        tpc = tp[mask].cumsum(0)
        fpc = (1 - tp[mask]).cumsum(0)
        recall = tpc / (count + 1e-16)
        precision = tpc / (tpc + fpc)
        for j in range(10):
            ap[i, j] = compute_ap(recall[:, j], precision[:, j])[0]
    return float(ap[:, 0].mean()), float(ap.mean())

def bootstrap(a, b, count, seed=20261027):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(count):
        ix = rng.integers(0, len(a), len(a))
        am, bm = bootstrap_ap(a, ix), bootstrap_ap(b, ix)
        samples.append([100 * (bm[0] - am[0]), 100 * (bm[1] - am[1])])
        if (i + 1) % 100 == 0:
            print(f'bootstrap {i+1}/{count}', flush=True)
    vals = np.array(samples)
    return {'resamples': count, 'seed': seed, 'unit': 'paired images with replacement',
            'class_rule': 'same GT-present classes in each paired replicate; absent classes excluded',
            'delta_mAP50_pp_CI95': np.quantile(vals[:, 0], [.025, .975]).tolist(),
            'delta_mAP50_95_pp_CI95': np.quantile(vals[:, 1], [.025, .975]).tolist(),
            'bootstrap_deltas_pp': samples}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--bootstrap', type=int, default=0)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.input / 'protocol.json').read_text())
    table = []
    for condition in args.conditions:
        existing_path = args.output / (condition + '_results.json')
        if existing_path.exists() and not args.force:
            existing = json.loads(existing_path.read_text())
            if (existing.get('n_images') == protocol['n_images']
                    and existing.get('source_signature') == protocol['signature']
                    and set(existing.get('metrics', {})) == set(METHODS)
                    and existing.get('paired_bootstrap_eamf_vs_baseline', {}).get('resamples', 0) >= args.bootstrap):
                for method, metric in existing['metrics'].items():
                    table.append({'condition': condition, 'method': method, 'n_images': existing['n_images'],
                                  'mAP50': metric['mAP50'], 'mAP50_95': metric['mAP50_95'],
                                  'activation': existing['activation'], 'gt_present_classes': metric['classes']})
                print(f'Reuse complete {condition}', flush=True)
                continue
        paths = sorted((args.input / condition).glob('*.json'))
        if len(paths) != protocol['n_images'] and not args.allow_partial:
            raise RuntimeError(f'{condition}: {len(paths)}/{protocol["n_images"]}; refusing partial results')
        rows = [json.loads(p.read_text()) for p in paths]
        if not rows:
            raise RuntimeError(f'No rows for {condition}')
        if any(r['signature'] != protocol['signature'] for r in rows):
            raise RuntimeError('Protocol mismatch')
        print(f'Analyze {condition}: {len(rows)} images', flush=True)
        stats = {}
        cache = {}
        for method in METHODS:
            predictions = [method_prediction(r, method) for r in rows]
            items = [match(r, p) for r, p in zip(rows, predictions)]
            cache[method] = items
            metric = aggregate(items)
            stats[method] = {**metric, 'operating_point': false_positives(items)}
            table.append({'condition': condition, 'method': method, 'n_images': len(rows),
                          'mAP50': metric['mAP50'], 'mAP50_95': metric['mAP50_95'],
                          'activation': sum(r['router_active'] for r in rows) / len(rows),
                          'gt_present_classes': metric['classes']})
            print(f'{condition} {method}: {metric["mAP50"]:.6f}/{metric["mAP50_95"]:.6f}', flush=True)
        payload = {'condition': condition, 'n_images': len(rows), 'partial': len(rows) != protocol['n_images'],
                   'metrics': stats, 'source_signature': protocol['signature'],
                   'metric_code': 'Ultralytics 8.3.233 DetectionValidator.match_predictions + ap_per_class',
                   'metric_code_sha256': sha256_file(Path(__file__)),
                   'activation': sum(r['router_active'] for r in rows) / len(rows),
                   'gamma_duplicates': sum(r['gamma_duplicates_original'] for r in rows),
                   'gamma_duplicate_active': sum(r['gamma_duplicates_original'] and r['router_active'] for r in rows)}
        active_rows = [r for r in rows if r['router_active']]
        payload['quality_diagnostics'] = {
            'active_images_with_original_weight_floor': sum(r['stats'][0]['score'] < .65 for r in active_rows),
            'active_original_mean_q': float(np.mean([max(r['stats'][0]['score'], .65) for r in active_rows])) if active_rows else None,
            'active_gamma_mean_q': float(np.mean([r['stats'][1]['score'] for r in active_rows])) if active_rows else None,
            'active_clahe_mean_q': float(np.mean([r['stats'][2]['score'] for r in active_rows])) if active_rows else None,
            'gamma_mean_absolute_error_to_nominal_L_target': float(np.mean([abs(r['stats'][1]['mean'] - .48) for r in active_rows])) if active_rows else None,
        }
        if condition in {'clean', 'gamma3.0', 'linear_noise'}:
            sensitivity = []
            for mu in [.32, .38, .44]:
                for sigma in [.08, .12, .16]:
                    metrics = aggregate([match(r, method_prediction(r, 'EAMF', mu=mu, sigma=sigma)) for r in rows])
                    activation = sum(r['stats'][0]['mean'] < mu or r['stats'][0]['std'] < sigma for r in rows) / len(rows)
                    sensitivity.append({'mu_threshold': mu, 'sigma_threshold': sigma,
                                        'mAP50': metrics['mAP50'], 'mAP50_95': metrics['mAP50_95'],
                                        'activation': activation, 'expected_detector_calls': 1 + 2 * activation})
            payload['descriptive_router_sensitivity'] = {
                'note': 'Test-set descriptive diagnostic only; no parameter selection; original .38/.12 retained',
                'rows': sensitivity}
        low_ix = [i for i, r in enumerate(rows) if r['stats'][0]['mean'] < .38]
        if condition == 'clean' and low_ix:
            payload['natural_low_luminance_proxy'] = {
                'definition': 'Lab L mean < .38; brightness proxy, NOT a verified nighttime label',
                'n_images': len(low_ix),
                'baseline': aggregate(cache['Baseline'], low_ix),
                'eamf': aggregate(cache['EAMF'], low_ix)}
        if args.bootstrap:
            payload['paired_bootstrap_eamf_vs_baseline'] = bootstrap(cache['Baseline'], cache['EAMF'], args.bootstrap)
            payload['paired_bootstrap_eamf_vs_tta'] = bootstrap(cache['TTA3'], cache['EAMF'], args.bootstrap)
        write_json(args.output / (condition + '_results.json'), payload)
        # Save matched TP flags and scores for independent paired reanalysis.
        with gzip.open(args.output / (condition + '_matched.jsonl.gz'), 'wt', encoding='utf8') as f:
            for i, r in enumerate(rows):
                d = {'image': r['image'], 'gt_cls': r['gt_cls'], 'methods': {}}
                for method, items in cache.items():
                    t, s, c, _ = items[i]
                    d['methods'][method] = {'tp': t.tolist(), 'scores': s.tolist(), 'classes': c.tolist()}
                f.write(json.dumps(d) + '\n')
        write_json(args.output / (condition + '_source_hashes.json'),
                   [{'file': p.name, 'sha256': sha256_file(p)} for p in paths])
    with (args.output / 'all_methods.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (args.output / 'per_class_AP50.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'method', 'class_id', 'AP50'])
        for condition in args.conditions:
            payload = json.loads((args.output / (condition + '_results.json')).read_text())
            for method, metric in payload['metrics'].items():
                for cls, ap in metric['class_ap50'].items():
                    writer.writerow([condition, method, cls, ap])

if __name__ == '__main__':
    main()

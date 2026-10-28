"""Analyze complete comparator records with the same official metric and paired AP bootstrap."""
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
import torch
from analyze_revision import aggregate, arrays, bootstrap, match, method_prediction, false_positives
from provenance_utils import sha256_file, write_json


def matched_items(rows, method):
    return [(np.asarray(r['methods'][method]['tp'], bool).reshape(-1, 10),
             np.asarray(r['methods'][method]['scores'], np.float32),
             np.asarray(r['methods'][method]['classes'], np.int32), np.asarray(r['gt_cls'], np.int32)) for r in rows]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-analysis', type=Path, required=True)
    p.add_argument('--method', choices=['ZeroDCE', 'YOLOv9', 'FineTuneGamma'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        output = args.output / (condition + '_results.json')
        if output.exists():
            saved = json.loads(output.read_text())
            assert saved['code_sha256'] == sha256_file(Path(__file__)) and saved['n_images'] == 2352
            print('Reuse complete ' + condition, flush=True)
            continue
        while True:
            paths = sorted((args.input / condition).glob('*.json'))
            if len(paths) == 2352:
                break
            if time.monotonic() - started >= args.wait_seconds:
                raise RuntimeError(f'Incomplete {condition}: {len(paths)}/2352')
            time.sleep(15)
        protocol = json.loads((args.input / 'protocol.json').read_text())
        rows = [json.loads(path.read_text()) for path in paths]
        lineage = []
        for path, row in zip(paths, rows):
            assert row['signature'] == protocol['signature']
            assert row['parent_sha256'] == sha256_file(args.parent / condition / path.name)
            lineage.append({'file': path.name, 'sha256': sha256_file(path), 'parent_sha256': row['parent_sha256']})
        cache = {}
        if args.method == 'YOLOv9':
            for method in ['Baseline', 'Gamma', 'EAMF']:
                cache[method] = [match(row, method_prediction(row, method)) for row in rows]
            comparison = ('Baseline', 'EAMF')
        else:
            cache[args.method] = [match(row, arrays(row['prediction'] if args.method == 'ZeroDCE' else row['views']['original'])) for row in rows]
            with gzip.open(args.parent_analysis / (condition + '_matched.jsonl.gz'), 'rt', encoding='utf-8') as stream:
                parent_rows = [json.loads(line) for line in stream]
            assert [r['image'] for r in rows] == [r['image'] for r in parent_rows]
            assert [r['gt_cls'] for r in rows] == [r['gt_cls'] for r in parent_rows]
            for method in ['Baseline', 'EAMF']:
                cache[method] = matched_items(parent_rows, method)
            comparison = (args.method, 'EAMF')
        metrics = {method: {**aggregate(items), 'operating_point': false_positives(items)} for method, items in cache.items()}
        print(args.method + ' ' + condition + ': ' + json.dumps({m: [v['mAP50'], v['mAP50_95']] for m, v in metrics.items()}), flush=True)
        result = {'condition': condition, 'extension': args.method, 'n_images': len(rows), 'metrics': metrics,
                  'paired_comparison': comparison[1] + '_minus_' + comparison[0],
                  'paired_bootstrap': bootstrap(cache[comparison[0]], cache[comparison[1]], 1000),
                  'code_sha256': sha256_file(Path(__file__)),
                  'metric_helper_sha256': sha256_file(Path(__file__).with_name('analyze_revision.py')),
                  'source_protocol_sha256': sha256_file(args.input / 'protocol.json'),
                  'interpretation': 'Exploratory additional comparator; same full-fold inputs and GT; no test-set parameter selection.'}
        write_json(output, result)
        write_json(args.output / (condition + '_source_hashes.json'), lineage)
        with gzip.open(args.output / (condition + '_matched.jsonl.gz'), 'wt', encoding='utf-8') as stream:
            for i, row in enumerate(rows):
                methods = {}
                for method, items in cache.items():
                    tp, scores, classes, _ = items[i]
                    methods[method] = {'tp': tp.tolist(), 'scores': scores.tolist(), 'classes': classes.tolist()}
                stream.write(json.dumps({'image': row['image'], 'gt_cls': row['gt_cls'], 'methods': methods}) + '\n')
        print('COMPLETE ' + condition, flush=True)
    print('ALL EXTENSION ANALYSES COMPLETE', flush=True)

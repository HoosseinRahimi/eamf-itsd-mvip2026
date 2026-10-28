"""Analyze the separately selected validation-calibrated control on the official fold."""
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
import torch
from analyze_revision import arrays, match, aggregate, bootstrap, false_positives
from analyze_extension import matched_items
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--primary', type=Path, required=True)
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
            continue
        while len(list((args.input / condition).glob('*.json'))) != 2352:
            if time.monotonic() - started >= args.wait_seconds:
                raise RuntimeError('Incomplete calibrated collection ' + condition)
            time.sleep(15)
        paths = sorted((args.input / condition).glob('*.json'))
        protocol = json.loads((args.input / 'protocol.json').read_text())
        rows = [json.loads(path.read_text()) for path in paths]
        lineage = []
        for path, row in zip(paths, rows):
            assert row['signature'] == protocol['signature']
            assert row['parent_sha256'] == sha256_file(args.primary / 'full_gpu640' / condition / path.name)
            lineage.append({'file': path.name, 'sha256': sha256_file(path), 'parent_sha256': row['parent_sha256']})
        with gzip.open(args.primary / 'full_gpu640_analysis_v2' / (condition + '_matched.jsonl.gz'), 'rt', encoding='utf-8') as stream:
            parents = [json.loads(line) for line in stream]
        assert [row['image'] for row in rows] == [row['image'] for row in parents]
        cache = {'CalibratedEAMF': [match(row, arrays(row['prediction'])) for row in rows],
                 'Baseline': matched_items(parents, 'Baseline'), 'EAMF': matched_items(parents, 'EAMF')}
        metrics = {method: {**aggregate(items), 'operating_point': false_positives(items)} for method, items in cache.items()}
        print('CALIBRATED ' + condition + ': ' + json.dumps({m: [v['mAP50'], v['mAP50_95']] for m,v in metrics.items()}), flush=True)
        result = {'condition': condition, 'n_images': 2352, 'metrics': metrics,
            'paired_comparison': 'CalibratedEAMF_minus_EAMF', 'paired_bootstrap': bootstrap(cache['EAMF'], cache['CalibratedEAMF'], 1000),
            'code_sha256': sha256_file(Path(__file__)), 'metric_helper_sha256': sha256_file(Path(__file__).with_name('analyze_revision.py')),
            'source_protocol_sha256': sha256_file(args.input / 'protocol.json'), 'selection_sha256': protocol['selection_sha256'],
            'interpretation': 'Selection used separate Fold2 validation only; Fold1 had previously been observed, so this remains exploratory.'}
        write_json(output, result)
        write_json(args.output / (condition + '_source_hashes.json'), lineage)
        with gzip.open(args.output / (condition + '_matched.jsonl.gz'), 'wt', encoding='utf-8') as stream:
            for i, row in enumerate(rows):
                methods = {}
                for method, items in cache.items():
                    tp, scores, classes, _ = items[i]
                    methods[method] = {'tp': tp.tolist(), 'scores': scores.tolist(), 'classes': classes.tolist()}
                stream.write(json.dumps({'image': row['image'], 'gt_cls': row['gt_cls'], 'methods': methods}) + '\n')
        print('CALIBRATED ANALYSIS COMPLETE ' + condition, flush=True)
    print('ALL CALIBRATED ANALYSES COMPLETE', flush=True)

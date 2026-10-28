"""Add paired EAMF-minus-gamma uncertainty from unchanged matched predictions."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from analyze_revision import aggregate, bootstrap
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', default=['clean', 'gamma3.0', 'linear_noise'])
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        output = args.output / (condition + '_gamma_comparison.json')
        source = args.analysis / (condition + '_matched.jsonl.gz')
        if output.exists():
            old = json.loads(output.read_text())
            if old['source_sha256'] == sha256_file(source) and old['code_sha256'] == sha256_file(Path(__file__)):
                print('Reuse complete ' + condition, flush=True)
                continue
            raise RuntimeError('Select new output for changed analysis')
        with gzip.open(source, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == 2352
        caches = {}
        for method in ['Gamma', 'EAMF']:
            caches[method] = [(np.asarray(r['methods'][method]['tp'], bool).reshape(-1, 10),
                              np.asarray(r['methods'][method]['scores'], np.float32),
                              np.asarray(r['methods'][method]['classes'], np.int32),
                              np.asarray(r['gt_cls'], np.int32)) for r in rows]
        print('Paired EAMF-minus-gamma: ' + condition, flush=True)
        summary = {method: aggregate(items) for method, items in caches.items()}
        reference = json.loads((args.analysis / (condition + '_results.json')).read_text())
        for method in caches:
            assert np.allclose([summary[method]['mAP50'], summary[method]['mAP50_95']],
                               [reference['metrics'][method]['mAP50'], reference['metrics'][method]['mAP50_95']], atol=1e-12)
        result = {'condition': condition, 'n_images': len(rows), 'metrics': summary,
                  'CI_EAMF_minus_Gamma': bootstrap(caches['Gamma'], caches['EAMF'], 1000),
                  'source_sha256': sha256_file(source), 'code_sha256': sha256_file(Path(__file__)),
                  'analysis_code_sha256': sha256_file(Path(__file__).with_name('analyze_revision.py')),
                  'interpretation': 'Exploratory paired comparison; no parameter selection or retrospective non-inferiority test.'}
        write_json(output, result)
        print('COMPLETE ' + condition, flush=True)

"""Prove that the parameterized default reproduces the frozen primary implementation."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_revision import method_prediction
from calibrate_eamf import DEFAULT, calibrated_prediction, keys
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    n = 0
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        paths = sorted((args.validation / condition).glob('*.json'))
        assert len(paths) == 234
        for path in paths:
            row = json.loads(path.read_text())
            selected = keys(DEFAULT)
            reference = {key: row[key] for key in ['image', 'shape', 'gt_cls', 'gt_boxes']}
            reference['views'] = {name: row['views'][key] for name, key in zip(['original', 'gamma', 'clahe'], selected)}
            reference['stats'] = [row['stats'][key] for key in selected]
            actual, expected = calibrated_prediction(row, DEFAULT), method_prediction(reference, 'EAMF')
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            n += 1
    write_json(args.output, {'status': 'PASS', 'condition_image_cases': n,
        'claim': 'Parameterized default has identical frozen-reference fused boxes, scores and classes on all 702 validation records.',
        'calibration_source_sha256': sha256_file(Path(__file__).with_name('calibrate_eamf.py')),
        'reference_source_sha256': sha256_file(Path(__file__).with_name('analyze_revision.py'))})
    print(f'PASS: {n} actual validation records match the frozen default implementation', flush=True)

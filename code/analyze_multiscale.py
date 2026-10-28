"""Compare a class-preserving, three-scale baseline with EAMF."""
import argparse
import json
import time
from pathlib import Path
from analyze_revision import aggregate, bootstrap, false_positives, fuse, match, method_prediction
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--extra', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.cache / 'protocol.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        result_path = args.output / (condition + '_multiscale.json')
        if result_path.exists():
            continue
        folder = args.extra / condition
        while len(list(folder.glob('*.json'))) < protocol['n_images']:
            time.sleep(15)
        items, eamf, lineage = [], [], []
        for path in sorted((args.cache / condition).glob('*.json')):
            row = json.loads(path.read_text())
            ext_path = folder / path.name
            extra = json.loads(ext_path.read_text())
            if extra['parent_record_sha256'] != sha256_file(path):
                raise RuntimeError('Additional view is not bound to this input record')
            row['views']['scale768'] = extra['scale768']
            pred = fuse(row, ['original', 'scale', 'scale768'], [1, 1, 1])
            items.append(match(row, pred))
            eamf.append(match(row, method_prediction(row, 'EAMF')))
            lineage.append({'image': row['image'], 'parent_sha256': sha256_file(path),
                            'extension_sha256': sha256_file(ext_path)})
        result = {'condition': condition, 'n_images': len(items), 'parent_signature': protocol['signature'],
                  'TTAms3': aggregate(items), 'operating_point': false_positives(items),
                  'EAMF': aggregate(eamf), 'CI_EAMF_minus_TTAms3': bootstrap(items, eamf, 1000),
                  'note': '512/640/768 without flips, three calls, approximately matched pixel-area budget; no equality of measured latency implied',
                  'source_lineage': lineage, 'code_sha256': sha256_file(Path(__file__))}
        write_json(result_path, result)
        print(f'COMPLETE MULTISCALE ANALYSIS {condition}', flush=True)

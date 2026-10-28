"""Audit exact encoded-image duplication across official partitions; preserve all inputs."""
import argparse
import gzip
import json
import subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_revision import aggregate
from provenance_utils import sha256_file, write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--fold1-manifest', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    official = args.work / 'archives'
    official.mkdir(exist_ok=True)
    test = json.loads(args.fold1_manifest.read_text())
    hashes = defaultdict(list)
    for row in test:
        hashes[row['image_sha256']].append(row['image'])
    cross = []
    fold_records = {}
    for fold in range(2, 7):
        output = args.work / f'fold{fold}_image_hashes.json'
        if output.exists():
            records = json.loads(output.read_text())
        else:
            nested = official / 'ITSD-dataset' / f'Fold-{fold}.rar'
            if not nested.exists():
                subprocess.run(['tar.exe', '-xf', str(args.archive), '-C', str(official), f'ITSD-dataset/Fold-{fold}.rar'], check=True)
            extracted = args.work / 'extracted'
            extracted.mkdir(exist_ok=True)
            images = extracted / f'Fold-{fold}' / 'images'
            if not images.exists():
                subprocess.run(['tar.exe', '-xf', str(nested), '-C', str(extracted)], check=True)
            paths = sorted(images.glob('*.jpg'))
            assert len(paths) > 2000
            records = [{'image': path.name, 'sha256': sha256_file(path)} for path in paths]
            write_json(output, records)
        fold_records[str(fold)] = {'images': len(records), 'hash_manifest_sha256': sha256_file(output)}
        for row in records:
            if row['sha256'] in hashes:
                cross.append({'source_fold': fold, 'source_image': row['image'], 'fold1_images': hashes[row['sha256']],
                              'image_sha256': row['sha256']})
        print(f'AUDITED Fold {fold}: {len(records)} images; cumulative matching records {len(cross)}', flush=True)
    leaked_ids = {image for row in cross for image in row['fold1_images']}
    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]
    redundant_ids = {name for ids in duplicate_groups for name in sorted(ids)[1:]}
    excluded = leaked_ids | redundant_ids
    keep = [row['image'] for row in test if row['image'] not in excluded]
    results = {}
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        path = args.analysis / (condition + '_matched.jsonl.gz')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        selected = [row for row in rows if row['image'] in set(keep)]
        metrics = {}
        for method in ['Baseline', 'Gamma', 'EAMF', 'TTA3']:
            items = [(np.asarray(r['methods'][method]['tp'], bool).reshape(-1, 10),
                      np.asarray(r['methods'][method]['scores'], np.float32),
                      np.asarray(r['methods'][method]['classes'], np.int32), np.asarray(r['gt_cls'], np.int32)) for r in selected]
            metrics[method] = aggregate(items)
        results[condition] = {'n_images': len(selected), 'metrics': metrics, 'matched_source_sha256': sha256_file(path)}
    write_json(args.work / 'fold_overlap_audit.json', {
        'code_sha256': sha256_file(Path(__file__)), 'fold1_manifest_sha256': sha256_file(args.fold1_manifest),
        'official_fold1_images': len(test), 'fold1_unique_encoded_images': len(hashes),
        'fold1_internal_duplicate_groups': duplicate_groups, 'cross_fold_matching_records': cross,
        'fold1_images_matching_other_folds': sorted(leaked_ids), 'other_fold_manifests': fold_records,
        'diagnostic_filter': 'Exclude every Fold1 image with exact encoded-file SHA256 match in folds 2--6; also remove redundant exact copies within Fold1.',
        'diagnostic_kept_images': keep, 'diagnostic_excluded_images': sorted(excluded), 'diagnostic_metrics': results,
        'limitation': 'Encoded-file duplicates only; no near-duplicate/session audit. Official weights training history is not independently reconstructed. Filtered sensitivity is exploratory, not a new untouched test set.'})
    print(f'COMPLETE: {len(leaked_ids)} Fold1 images match other folds; diagnostic keeps {len(keep)} images', flush=True)

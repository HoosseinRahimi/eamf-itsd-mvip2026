"""Verify added raw-output lineage, official AP, timing, and training records."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from analyze_revision import aggregate
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--extension', type=Path, required=True)
    p.add_argument('--manuscript', type=Path)
    p.add_argument('--baseline-delivery', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    code = Path(__file__).parent
    torch.set_num_threads(1)
    checks = []
    def require(name, condition):
        if not condition:
            raise RuntimeError(name)
        checks.append({'check': name, 'passed': True})
    if args.baseline_delivery:
        baseline = json.loads((args.baseline_delivery / 'DELIVERY_MANIFEST.json').read_text())
        require('previous delivery remains byte-identical', all(sha256_file(args.baseline_delivery / name) == item['sha256'] for name, item in baseline['files'].items()))
    parent = json.loads((args.primary / 'full_gpu640' / 'protocol.json').read_text())
    require('unchanged primary collection code', parent['code_sha256'] == sha256_file(code / 'collect_revision.py'))
    require('unchanged primary EAMF code', parent['eamf_sha256'] == sha256_file(code / 'eamf.py'))
    sources = [('zero_dce_predictions', 'zero_dce_analysis', 'collect_zero_dce.py', 'analyze_extension.py'),
               ('yolov9_predictions', 'yolov9_analysis', 'collect_detector_extension.py', 'analyze_extension.py'),
               ('finetune_predictions', 'finetune_analysis', 'collect_detector_extension.py', 'analyze_extension.py'),
               ('calibrated_predictions', 'calibrated_analysis', 'calibrate_eamf.py', 'analyze_calibrated.py')]
    raw_total = 0
    for collection_name, analysis_name, collector, analyzer in sources:
        root = args.extension / collection_name
        protocol = json.loads((root / 'protocol.json').read_text())
        require(collection_name + ' executed collector', protocol['code_sha256'] == sha256_file(code / collector))
        require(collection_name + ' parent protocol', protocol['parent_signature'] == parent['signature'])
        for condition in ['clean', 'gamma3.0', 'linear_noise']:
            paths = sorted((root / condition).glob('*.json'))
            parent_paths = sorted((args.primary / 'full_gpu640' / condition).glob('*.json'))
            require(collection_name + '/' + condition + ' full fold', len(paths) == 2352 and [p.name for p in paths] == [p.name for p in parent_paths])
            result_path = args.extension / analysis_name / (condition + '_results.json')
            result = json.loads(result_path.read_text())
            require(analysis_name + '/' + condition + ' executed analyzer', result['code_sha256'] == sha256_file(code / analyzer))
            require(analysis_name + '/' + condition + ' metric helper', result['metric_helper_sha256'] == sha256_file(code / 'analyze_revision.py'))
            lineage = json.loads((args.extension / analysis_name / (condition + '_source_hashes.json')).read_text())
            for path, parent_path, record in zip(paths, parent_paths, lineage):
                assert path.name == record['file'] and sha256_file(path) == record['sha256']
                assert sha256_file(parent_path) == record['parent_sha256']
                raw = json.loads(path.read_text())
                assert raw['signature'] == protocol['signature'] and raw['parent_sha256'] == record['parent_sha256']
            require(analysis_name + '/' + condition + ' 2352 raw and parent hashes', len(lineage) == len(paths))
            matched = args.extension / analysis_name / (condition + '_matched.jsonl.gz')
            with gzip.open(matched, 'rt', encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream]
            require(analysis_name + '/' + condition + ' matched image coverage', len(rows) == 2352 and [r['image'] for r in rows] == [p.stem + '.jpg' for p in paths])
            for method, metric in result['metrics'].items():
                items = [(np.asarray(r['methods'][method]['tp'], bool).reshape(-1, 10),
                          np.asarray(r['methods'][method]['scores'], np.float32),
                          np.asarray(r['methods'][method]['classes'], np.int32), np.asarray(r['gt_cls'], np.int32)) for r in rows]
                rebuilt = aggregate(items)
                require(analysis_name + '/' + condition + '/' + method + ' official AP reaggregation',
                        np.allclose([rebuilt['mAP50'], rebuilt['mAP50_95']], [metric['mAP50'], metric['mAP50_95']], atol=1e-12, rtol=0))
            ci = result['paired_bootstrap']
            draws = np.asarray(ci['bootstrap_deltas_pp'])
            require(analysis_name + '/' + condition + ' paired bootstrap quantiles', draws.shape == (1000, 2) and
                np.allclose(np.quantile(draws, [.025, .975], axis=0).T,
                            [ci['delta_mAP50_pp_CI95'], ci['delta_mAP50_95_pp_CI95']], atol=1e-12, rtol=0))
            raw_total += len(paths)
    training_root = args.extension / 'gamma_finetune'
    training = json.loads((training_root / 'training_protocol.json').read_text())
    completed = json.loads((training_root / 'training_complete.json').read_text())
    require('executed training source preserved', training['code_sha256'] == sha256_file(code / 'train_gamma_comparator.py'))
    checkpoint = Path(completed['checkpoint'])
    if not checkpoint.is_file():
        checkpoint = training_root / 'run' / 'weights' / 'last.pt'
    require('final checkpoint matches actual completed training', completed['last_checkpoint_sha256'] == sha256_file(checkpoint) and completed['training_seconds'] > 0)
    with (training_root / 'run' / 'results.csv').open() as stream:
        epochs = list(csv.DictReader(stream))
    require('three epochs recorded', [int(r['epoch']) for r in epochs] == [1, 2, 3])
    training_manifest = json.loads((training_root / 'fold2_manifest.json').read_text())
    test_manifest = json.loads((args.primary / 'full_gpu640' / 'manifest.json').read_text())
    require('fine-tuning source excludes exact test-image overlap', not ({r['image_sha256'] for r in training_manifest} & {r['image_sha256'] for r in test_manifest}))
    require('fine-tuning source is deduplicated', len(training_manifest) == len({r['image_sha256'] for r in training_manifest}) == training['train_images'] + training['validation_images'])
    calibration = json.loads((args.extension / 'calibration_validation' / 'selection.json').read_text())
    require('validation calibration executed source', calibration['code_sha256'] == sha256_file(code / 'calibrate_eamf.py'))
    require('predeclared 21 configurations and validation-only selection', len(calibration['scores']) == 21 and calibration['validation_images'] == 234 and not calibration['test_labels_used_for_selection'])
    maximum = max(row['mean_mAP50_95'] for row in calibration['scores'])
    chosen = next(row for row in calibration['scores'] if maximum - row['mean_mAP50_95'] <= 1e-12)
    require('recorded validation argmax and tie rule', chosen['configuration'] == calibration['selected_configuration'] and chosen['parameters'] == calibration['selected_parameters'])
    benchmark = json.loads((args.extension / 'benchmark_finalization.json').read_text())
    require('executed standalone benchmark source', benchmark['protocol']['code_sha256'] == sha256_file(code / 'benchmark_finalization.py'))
    require('executed benchmark helper', benchmark['protocol']['timing_helper_sha256'] == sha256_file(code / 'benchmark_revision.py'))
    keys = {(r['condition'], r['image'], r['repeat'], r['method']) for r in benchmark['rows']}
    expected = {(c, image, repeat, method) for c in benchmark['protocol']['conditions'] for image in benchmark['protocol']['selected_images']
                for repeat in range(3) for method in benchmark['protocol']['methods']}
    require('all 3456 standalone timing measurements unique and complete', len(benchmark['rows']) == len(keys) == 3456 and keys == expected)
    require('positive finite measured totals', all(np.isfinite(r['ms']['total']) and r['ms']['total'] > 0 for r in benchmark['rows']))
    for condition, methods in benchmark['summary_ms'].items():
        for method, values in methods.items():
            rows = [r['ms'] for r in benchmark['rows'] if r['condition'] == condition and r['method'] == method]
            require(condition + '/' + method + ' measured timing aggregation', len(rows) == 144 and all(
                np.isclose(value, np.mean([r[key] for r in rows]), atol=1e-12, rtol=0) for key, value in values.items()))
    if args.manuscript:
        for filename in ['manuscript_table_lineage.json', 'final_extension_lineage.json']:
            lineage = json.loads((args.manuscript / filename).read_text())
            require(filename + ' final TeX hashes', all(sha256_file(args.manuscript / name) == digest for name, digest in lineage['outputs'].items()))
        extended = json.loads((args.manuscript / 'final_extension_lineage.json').read_text())
        require('extension table input hashes', all(sha256_file(args.extension / name) == digest for name, digest in extended['inputs'].items()))
        require('official conference template class', sha256_file(args.manuscript / 'IEEEtran.cls') == 'c972aca108fda004c3514d63658e02816da2e54d9a1451e870b9bd970e003f55')
    write_json(args.output, {'status': 'PASS', 'checks': checks, 'new_raw_records': raw_total, 'verifier_sha256': sha256_file(Path(__file__)),
        'scope': 'Actual extension outputs, official AP, training and timing lineage. Does not certify author declarations, private portal state, deadline resolution, publication access, or visual layout.'})
    print(f'PASS: {len(checks)} final evidence checks; {raw_total} additional raw records verified', flush=True)

"""Fail closed on incomplete or changed quantitative evidence before delivery."""
import argparse
import json
from pathlib import Path
import numpy as np
from collect_revision import CONDITIONS
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='D: revision evidence directory')
    p.add_argument('--manuscript', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    code = Path(__file__).parent
    checks = []
    def require(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise RuntimeError(name)
    primary = args.root / 'full_gpu640'
    analyses = args.root / 'full_gpu640_analysis_v2'
    protocol = json.loads((primary / 'protocol.json').read_text())
    require('full 2352-image working-resolution protocol', protocol['n_images'] == 2352 and protocol['work_longside'] == 640)
    require('executed collector source hash', protocol['code_sha256'] == sha256_file(code / 'collect_revision.py'))
    require('EAMF source hash', protocol['eamf_sha256'] == sha256_file(code / 'eamf.py'))
    require('official Fold 1 weight hash', protocol['weights_sha256'] == 'fc92a7c61edbbd7edc4dea6ec1fd6992825e230e55e9bd22286285810aeb8b54')
    for condition in CONDITIONS:
        result = json.loads((analyses / (condition + '_results.json')).read_text())
        require(condition + ' complete and matched', result['n_images'] == 2352 and not result['partial'] and result['source_signature'] == protocol['signature'])
        require(condition + ' analyzer source hash', result['metric_code_sha256'] == sha256_file(code / 'analyze_revision.py'))
        require(condition + ' metric classes and finite values', all(
            row['classes'] == 116 and np.isfinite(row['mAP50']) and np.isfinite(row['mAP50_95'])
            for row in result['metrics'].values()))
        for key in ['paired_bootstrap_eamf_vs_baseline', 'paired_bootstrap_eamf_vs_tta']:
            ci = result[key]
            draws = np.asarray(ci['bootstrap_deltas_pp'])
            require(condition + ' ' + key + ' 1000 draws', draws.shape == (1000, 2) and np.isfinite(draws).all())
            require(condition + ' ' + key + ' quantiles', np.allclose(np.quantile(draws[:, 0], [.025, .975]), ci['delta_mAP50_pp_CI95'], atol=1e-12))
        sources = json.loads((analyses / (condition + '_source_hashes.json')).read_text())
        require(condition + ' raw-record hashes', len(sources) == 2352 and all(
            sha256_file(primary / condition / record['file']) == record['sha256'] for record in sources))
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        result = json.loads((args.root / 'multiscale_analysis' / (condition + '_multiscale.json')).read_text())
        require(condition + ' three-scale analysis source', result['code_sha256'] == sha256_file(code / 'analyze_multiscale.py'))
        require(condition + ' three-scale full fold', result['n_images'] == 2352 and result['CI_EAMF_minus_TTAms3']['resamples'] == 1000)
        for record in result['source_lineage']:
            name = Path(record['image']).with_suffix('.json')
            require_hash = (sha256_file(primary / condition / name) == record['parent_sha256']
                and sha256_file(args.root / 'multiscale_extra' / condition / name) == record['extension_sha256'])
            if not require_hash:
                raise RuntimeError(f'Extension lineage mismatch: {condition}/{name}')
        checks.append({'check': condition + ' 2352 extension record hashes', 'passed': True})
    ms_protocol = json.loads((args.root / 'multiscale_extra' / 'protocol.json').read_text())
    require('three-scale collection source', ms_protocol['code_sha256'] == sha256_file(code / 'collect_multiscale.py'))
    benchmark = json.loads((args.root / 'benchmark_gpu640.json').read_text())
    require('benchmark executed source', benchmark['protocol']['code_sha256'] == sha256_file(code / 'benchmark_revision.py'))
    require('benchmark 3456 unique measured rows', len(benchmark['rows']) == 3456 and len({
        (r['condition'], r['image'], r['repeat'], r['method']) for r in benchmark['rows']}) == 3456)
    require('benchmark totals positive and finite', all(np.isfinite(r['ms']['total']) and r['ms']['total'] > 0 for r in benchmark['rows']))
    figure = json.loads((args.manuscript / 'figure_lineage.json').read_text())
    require('regenerated figure hash', figure['output_sha256'] == sha256_file(args.manuscript / 'revision_examples.png'))
    lineage = json.loads((args.manuscript / 'manuscript_table_lineage.json').read_text())
    require('all generated TeX output hashes', all(sha256_file(args.manuscript / name) == digest for name, digest in lineage['outputs'].items()))
    native = json.loads((args.root / 'full_cpu' / 'protocol.json').read_text())
    require('native-resolution control source preserved', native['code_sha256'] == sha256_file(code / 'collect_native_reference.py'))
    write_json(args.output, {'status': 'PASS', 'checks': checks,
                            'native_control': '2352 unmodified images only; not seven-condition primary experiment',
                            'primary_raw_records': 7 * 2352, 'extra_scale_records': 3 * 2352,
                            'remaining_limits': ['one fold and detector', 'unverified real-night labels',
                              'no learned-enhancer/fine-tuning comparison', 'no independently documented historical tuning',
                              'public code URL unassigned']})
    print(f'PASS: {len(checks)} evidence checks; raw records and all executed-source hashes verified')

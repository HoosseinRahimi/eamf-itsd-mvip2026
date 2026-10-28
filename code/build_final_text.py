"""Generate camera-ready extensions only from complete, frozen real-run outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from provenance_utils import sha256_file, write_json

CONDITIONS = ['clean', 'gamma3.0', 'linear_noise']
LABELS = ['Unmodified', '$g=3.0$', 'Linear-light noise']


def pair(metric):
    return f"{100*metric['mAP50']:.2f} / {100*metric['mAP50_95']:.2f}"


def interval(values):
    return f'[{values[0]:+.2f}, {values[1]:+.2f}]'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--extension', type=Path, required=True)
    p.add_argument('--manuscript', type=Path, required=True)
    args = p.parse_args()
    data, inputs = {}, {}
    for kind in ['zero_dce_analysis', 'yolov9_analysis', 'finetune_analysis', 'calibrated_analysis']:
        data[kind] = {}
        for condition in CONDITIONS:
            path = args.extension / kind / (condition + '_results.json')
            result = json.loads(path.read_text())
            assert result['n_images'] == 2352
            assert result['paired_bootstrap']['resamples'] == 1000
            assert len(result['paired_bootstrap']['bootstrap_deltas_pp']) == 1000
            data[kind][condition] = result
            inputs[path.relative_to(args.extension).as_posix()] = sha256_file(path)
    gamma = {}
    for condition in CONDITIONS:
        path = args.extension / 'gamma_comparison' / (condition + '_gamma_comparison.json')
        gamma[condition] = json.loads(path.read_text())
        assert gamma[condition]['n_images'] == 2352
        inputs[path.relative_to(args.extension).as_posix()] = sha256_file(path)
    train_path = args.extension / 'gamma_finetune' / 'training_protocol.json'
    train = json.loads(train_path.read_text())
    completion = json.loads((args.extension / 'gamma_finetune' / 'training_complete.json').read_text())
    assert completion['completed_epochs'] == train['epochs'] == 3
    assert train['test_overlap_by_filename_and_sha256'] == 0
    checkpoint = Path(completion['checkpoint'])
    if not checkpoint.is_file():
        checkpoint = args.extension / 'gamma_finetune' / 'run' / 'weights' / 'last.pt'
    assert sha256_file(checkpoint) == completion['last_checkpoint_sha256']
    overlap_path = args.extension / 'fold_overlap' / 'fold_overlap_audit.json'
    overlap = json.loads(overlap_path.read_text())
    benchmark_path = args.extension / 'benchmark_finalization.json'
    benchmark = json.loads(benchmark_path.read_text())
    assert len(benchmark['rows']) == 3456 and 'summary_ms' in benchmark
    selection_path = args.extension / 'calibration_validation' / 'selection.json'
    selection = json.loads(selection_path.read_text())
    assert len(selection['scores']) == 21 and not selection['test_labels_used_for_selection']
    for path in [train_path, args.extension / 'gamma_finetune' / 'training_complete.json', overlap_path, benchmark_path, selection_path]:
        inputs[path.relative_to(args.extension).as_posix()] = sha256_file(path)
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_measured_text.py')),
        '--analysis', str(args.primary / 'full_gpu640_analysis_v2'), '--cache', str(args.primary / 'full_gpu640'),
        '--benchmark', str(benchmark_path), '--multiscale', str(args.primary / 'multiscale_analysis'),
        '--manuscript', str(args.manuscript)], check=True)
    primary = {condition: json.loads((args.primary / 'full_gpu640_analysis_v2' / (condition + '_results.json')).read_text()) for condition in CONDITIONS}
    rows = [
        ('YOLOv12n baseline', [primary[c]['metrics']['Baseline'] for c in CONDITIONS]),
        ('YOLOv12n + EAMF', [primary[c]['metrics']['EAMF'] for c in CONDITIONS]),
        ('YOLOv12n + gamma', [primary[c]['metrics']['Gamma'] for c in CONDITIONS]),
        ('YOLOv12n + Zero-DCE', [data['zero_dce_analysis'][c]['metrics']['ZeroDCE'] for c in CONDITIONS]),
        ('YOLOv12n, gamma fine-tune', [data['finetune_analysis'][c]['metrics']['FineTuneGamma'] for c in CONDITIONS]),
        ('Validation-calibrated EAMF', [data['calibrated_analysis'][c]['metrics']['CalibratedEAMF'] for c in CONDITIONS]),
        ('YOLOv9t baseline', [data['yolov9_analysis'][c]['metrics']['Baseline'] for c in CONDITIONS]),
        ('YOLOv9t + EAMF', [data['yolov9_analysis'][c]['metrics']['EAMF'] for c in CONDITIONS]),
        ('YOLOv9t + gamma', [data['yolov9_analysis'][c]['metrics']['Gamma'] for c in CONDITIONS])]
    table = [r'\begin{table*}[t]', r'\centering\small\setlength{\tabcolsep}{7pt}',
        r'\caption{Additional full-Fold-1 controls: mAP@50 / mAP@50:95 (percent), all 2,352 images per cell. Zero-DCE uses a fixed released checkpoint; the detector fine-tune uses three fixed epochs. Calibration uses separate Fold 2 validation labels and preserves detector weights.}',
        r'\label{tab:additional}', r'\begin{tabular}{lccc}', r'\toprule',
        r'Method & Unmodified & $g=3.0$ & Linear-light noise \\', r'\midrule']
    for label, metrics in rows:
        table.append(label + ' & ' + ' & '.join(pair(metric) for metric in metrics) + r' \\')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (args.manuscript / 'final_extension_tables.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
    protocol = (r'\paragraph{Additional controls.} We evaluate the official Fold~1 YOLOv9t checkpoint~\cite{wang2024yolov9} with the same class mapping, inputs, thresholds, and unchanged EAMF constants. '
        r'For a learned preprocessor, Zero-DCE~\cite{guo2020zerodce} uses the original authors\textquotesingle{} released Epoch99 weights without ITSD training; RGB values are normalized to $[0,1]$, the final enhanced output is quantized to eight bits without JPEG re-encoding, then passed to the unchanged YOLOv12n detector. '
        f'A bounded training comparator fine-tunes YOLOv12n for three fixed epochs on {train["train_images"]:,} deduplicated Fold~2 images, with {train["validation_images"]} separately allocated validation images; all exact Fold~1 matches are excluded before training. '
        r'Gamma augmentation uses probability 0.5 and exponent $U[1,3]$. SGD uses learning rate 0.001, final multiplier 0.1, momentum 0.9, weight decay 0.0005, one warm-up epoch, batch four, FP32, and seed 20261028; other color/geometric augmentations are disabled. '
        r'The last epoch is selected by budget, not Fold~1 score. These comparisons cover the complete unmodified, $g=3.0$, and linear-noise conditions; they are exploratory rather than an untouched confirmation of earlier choices.\n')
    protocol = protocol.replace(r'\textquotesingle{}', "'").replace(r'\n', '\n')
    protocol += (r'A separate labeled-validation control evaluates 21 one-factor configurations: the default plus two alternatives for each of ten routing, gamma, CLAHE, quality-weight, and WBF parameters. '
        r'Selection maximizes the unrounded equal-condition mean mAP@50:95 on the 234 disjoint Fold~2 validation images, with fixed tie-breaking, before generating this control\textquotesingle{}s Fold~1 outputs. '
        r'It changes no detector weights and does not replace the historical primary configuration. The previously inspected Fold~1 is not an untouched confirmation set; the original detector\textquotesingle{}s training exposure is not reconstructed.\n').replace(r'\textquotesingle{}', "'").replace(r'\n', '\n')
    (args.manuscript / 'final_extension_protocol.tex').write_text(protocol, encoding='utf-8')
    results = [r'Table~\ref{tab:additional} adds a fixed learned enhancer, a bounded training comparator, and a second detector.']
    ci_gamma = gamma['gamma3.0']['CI_EAMF_minus_Gamma']
    results.append('For YOLOv12n at $g=3.0$, the EAMF-minus-gamma mAP@50 interval is ' + interval(ci_gamma['delta_mAP50_pp_CI95']) +
        ' percentage points, while the mAP@50:95 interval is ' + interval(ci_gamma['delta_mAP50_95_pp_CI95']) +
        ' points. The stricter-IoU metric therefore favors gamma alone in this paired comparison; full fusion has no demonstrated universal advantage over its simpler component.')
    z = data['zero_dce_analysis']
    for condition in ['gamma3.0', 'linear_noise']:
        result = z[condition]
        delta = 100 * (result['metrics']['EAMF']['mAP50'] - result['metrics']['ZeroDCE']['mAP50'])
        label = '$g=3.0$' if condition == 'gamma3.0' else 'linear-light noise'
        results.append(f'Against the fixed Zero-DCE preprocessor under {label}, EAMF changes mAP@50 by {delta:+.2f} percentage points (paired 95\\% interval {interval(result["paired_bootstrap"]["delta_mAP50_pp_CI95"])}).')
    y9 = data['yolov9_analysis']
    for condition in ['gamma3.0', 'linear_noise']:
        result = y9[condition]
        base, eamf = result['metrics']['Baseline'], result['metrics']['EAMF']
        label = '$g=3.0$' if condition == 'gamma3.0' else 'linear-light noise'
        results.append(f'On YOLOv9t under {label}, baseline/EAMF mAP@50 are {100*base["mAP50"]:.2f}\\%/{100*eamf["mAP50"]:.2f}\\%, with paired difference interval {interval(result["paired_bootstrap"]["delta_mAP50_pp_CI95"])} points.')
    results.append('The noisy-input outcome thus depends on the detector; two tested configurations do not establish architecture-wide generality.')
    ft = data['finetune_analysis']['gamma3.0']
    delta = 100 * (ft['metrics']['EAMF']['mAP50'] - ft['metrics']['FineTuneGamma']['mAP50'])
    results.append(f'The three-epoch gamma fine-tune reaches {100*ft["metrics"]["FineTuneGamma"]["mAP50"]:.2f}\\% mAP@50 at $g=3.0$; EAMF minus this trained comparator is {delta:+.2f} points (95\\% interval {interval(ft["paired_bootstrap"]["delta_mAP50_pp_CI95"])}). '
        f'Training took {completion["training_seconds"] / 60:.1f} minutes on the stated GPU. This bounded run is not a search for the best trained model; all outcomes are retained.')
    default = selection['scores'][0]
    chosen = next(row for row in selection['scores'] if row['configuration'] == selection['selected_configuration'])
    changed = {key: value for key, value in chosen['parameters'].items() if value != default['parameters'][key]}
    changed_text = ', '.join(key.replace('_', ' ') + f'={value:g}' for key, value in changed.items()) or 'the original parameters'
    results.append(f'The separate validation study selects {changed_text}; validation mean mAP@50:95 is {100*chosen["mean_mAP50_95"]:.2f}\\% versus {100*default["mean_mAP50_95"]:.2f}\\% for the default. '
        'This is bounded one-factor calibration, not global optimization or evidence of historical selection independence. '
        'Table~\\ref{tab:additional} reports the selected control separately; the timing table refers to default EAMF.')
    (args.manuscript / 'final_extension_results.tex').write_text('\n\n'.join(results) + '\n', encoding='utf-8')
    filtered = overlap['diagnostic_metrics']
    n = len(overlap['diagnostic_kept_images'])
    gain = 100 * (filtered['gamma3.0']['metrics']['EAMF']['mAP50'] - filtered['gamma3.0']['metrics']['Baseline']['mAP50'])
    noise_gain = 100 * (filtered['linear_noise']['metrics']['EAMF']['mAP50'] - filtered['linear_noise']['metrics']['Baseline']['mAP50'])
    split = (f'An encoded-file SHA-256 audit found {overlap["official_fold1_images"]-overlap["fold1_unique_encoded_images"]} redundant exact copies within Fold~1 and {len(overlap["fold1_images_matching_other_folds"])} Fold~1 images with exact copies in other official folds. '
        f'For comparability, primary tables retain the official fold. A diagnostic removing cross-fold matches and redundant internal copies leaves {n:,} images: EAMF-minus-baseline mAP@50 is {gain:+.2f} points at $g=3.0$ and {noise_gain:+.2f} under linear noise. '
        'These filtered point estimates preserve the main qualitative tradeoff. The audit does not establish the released checkpoints\' exact original training exposure or remove near-duplicate/session dependence.\n')
    (args.manuscript / 'split_audit_text.tex').write_text(split, encoding='utf-8')
    runtime = (r'Accuracy and timing use Python 3.12.3, Ultralytics 8.3.233, PyTorch 2.2.2, FP32, and an NVIDIA GeForce RTX 3060 Laptop GPU. '
        r'The final standalone benchmark uses 48 uniformly selected Fold~1 images, three repetitions, five warm-up invocations per method, and randomized method order per image. '
        r'All timing-table values come from this one run. Starting from a common preloaded 640-pixel working input, timers include routing, enhancement and its transfers, quality weights, detector preprocessing/inference/NMS/CPU box transfer, and fusion, with CUDA synchronization. '
        r'Native decoding, shared resize, and common stress construction are excluded. FPS is reciprocal mean wrapper latency, not camera-to-output speed or batched throughput.\n')
    (args.manuscript / 'runtime_protocol.tex').write_text(runtime.replace(r'\n', '\n'), encoding='utf-8')
    lineage_path = args.manuscript / 'manuscript_table_lineage.json'
    lineage = json.loads(lineage_path.read_text())
    lineage['outputs']['runtime_protocol.tex'] = sha256_file(args.manuscript / 'runtime_protocol.tex')
    lineage['final_runtime_generator_sha256'] = sha256_file(Path(__file__))
    write_json(lineage_path, lineage)
    names = ['final_extension_tables.tex', 'final_extension_protocol.tex', 'final_extension_results.tex', 'split_audit_text.tex']
    write_json(args.manuscript / 'final_extension_lineage.json', {'inputs': inputs,
        'generator_sha256': sha256_file(Path(__file__)), 'outputs': {name: sha256_file(args.manuscript / name) for name in names}})
    print('COMPLETE: final measured text, additional tables, split audit, and common timing lineage generated', flush=True)
